from functools import partial

import torch
import torch.nn as nn
import torch.distributions as td

from timm.models.vision_transformer import PatchEmbed, Block
from timm.models.vision_transformer import CrossAttention, Attention, DropPath, Mlp

from util.pos_embed import get_2d_sincos_pos_embed
import copy

import torch.nn.functional as F


def ema_model(modelA, modelB, m):
    with torch.no_grad():
        for paramA, paramB in zip(modelA.parameters(), modelB.parameters()):
            paramA.data = m * paramA.data + (1 - m) * paramB.data
    return modelA


class CSABlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm_kv1 = norm_layer(dim)
        self.cattn = CrossAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        mlp_hidden_dim = int(dim * mlp_ratio)

        '''self.mlp1 = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )'''
        self.norm3 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.norm4 = norm_layer(dim)
        self.mlp2 = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, kvx, src_mask=None):
        x = x + self.drop_path(
            self.cattn(self.norm1(x), self.norm_kv1(kvx), src_mask=src_mask)
        )
        #x = x + self.mlp1(self.norm2(x))
        x = x + self.drop_path(self.attn(self.norm3(x)))
        x = x + self.mlp2(self.norm4(x))
        return x


class PhiNetv2(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        stoch=32,
        discrete=32,
        decoder_embed_dim=384,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        norm_pix_loss=True,
        kl_scale=0.01,
        kl_balance=0.2,
        kl_freebit=0.1,
        mask_ratio=0.75,
        noise_scale=0.5,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False
        )  # fixed sin-cos embedding

        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    qk_scale=None,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        # Determine the size of stochastic variable
        # For discrete latents:
        # - It consists of M N-dimensional one-hot vectors (M: stoch, N: discrete)
        # For continuous latents:
        # - It is M-dimenisonal gaussian. Thus it has M * 2 for mean and std
        stoch_size = stoch * discrete if discrete != 0 else stoch * 2

        # Posterior takes both src_h and tgt_h
        # Thus it has embed_dim * 2 as an input dimension
        self.to_posterior = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, stoch_size),
        )

        # Prior only takes src_h
        # Thus it has embed_dim as an input dimension
        self.to_prior = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, stoch_size),
        )

        self.ca3 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
        )

        self.decoder_embed_mae = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.decoder_embed_deter = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.decoder_embed_stoch = nn.Linear(stoch_size, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False
        )  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList(
            [
                CSABlock(
                    decoder_embed_dim,
                    decoder_num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    qk_scale=None,
                    norm_layer=norm_layer,
                )
                for i in range(decoder_depth)
            ]
        )

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, patch_size**2 * in_chans, bias=True
        )  # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss
        self.kl_scale = kl_scale
        self.kl_balance = kl_balance
        self.kl_freebit = kl_freebit

        self.stoch = stoch
        self.discrete = discrete

        self.noise_scale = noise_scale
        self.mask_ratio = mask_ratio

        self.initialize_weights()

        self.beta=0.99
        self.ema_model = copy.deepcopy(self)
        self.ema_model.train()

        # Do not update EMA model
        for param in self.ema_model.parameters():
            param.requires_grad = False  

    @torch.no_grad()
    def update_ema_model(self):
        if isinstance(self, torch.nn.parallel.DistributedDataParallel):
            model = self.module 
        else:
            model = self

        for ema_param, param in zip(model.ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(model.beta).add_(param.data, alpha=1 - model.beta)


    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.patch_embed.num_patches**0.5),
            cls_token=True,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            int(self.patch_embed.num_patches**0.5),
            cls_token=True,
        )
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(decoder_pos_embed).float().unsqueeze(0)
        )

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.mask_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum("nchpwq->nhwpqc", x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def perturb(self, x):
        noise = torch.randn_like(x) * self.noise_scale
        return x + noise

    def forward_encoder(self, imgs, mask_ratio=0.0):
        x = self.patch_embed(imgs)
        x = x + self.pos_embed[:, 1:, :]

        if mask_ratio != 0.0:
            # masking: length -> length * mask_ratio
            x, mask, ids_restore = self.random_masking(x, mask_ratio)
        else:
            mask, ids_restore = None, None

        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x, mask, ids_restore

    def forward_decoder_linear(self, h):
        #kvx_h = self.get_feat(h, z)

        x = self.ca1(h)
        
        return x
    
    def forward_decoder_fut_latent(self, h, z):
        kvx_h = self.get_feat(h, z)

        mask_tokens = self.mask_token.repeat(h.shape[0], h.shape[1], 1)
        x = mask_tokens + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x, kvx=kvx_h)
        x = self.decoder_norm(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def get_feat(self, h, z):
        h = self.decoder_embed_deter(h) + self.decoder_pos_embed
        if self.discrete != 0:
            z = z.reshape(*z.shape[:-2], 1, self.stoch * self.discrete)
        z = self.decoder_embed_stoch(z)
        feat = torch.cat([z, h], dim=1)
        return feat

    def make_dist(self, logits):
        if self.discrete != 0:
            logits = logits.reshape([-1, self.stoch, self.discrete])
            dist = td.Independent(td.OneHotCategoricalStraightThrough(logits=logits), 1)
        else:
            mean, std = torch.split(logits, 2, -1)
            dist = td.Normal(mean, std)
        return dist

    def kl_loss(self, post_logits, prior_logits):
        balance = self.kl_balance
        freebit = self.kl_freebit
        post_to_prior_kl = td.kl_divergence(
            self.make_dist(post_logits), self.make_dist(prior_logits.detach())
        )
        prior_to_post_kl = td.kl_divergence(
            self.make_dist(post_logits.detach()), self.make_dist(prior_logits)
        )
        kl_value = (
            post_to_prior_kl * balance + prior_to_post_kl * (1.0 - balance)
        ).mean()
        kl_loss = torch.maximum(kl_value, torch.ones_like(kl_value) * freebit)
        return kl_loss, kl_value

    #Phinet V2 with symmetric loss
    def forward(self, src_imgs, tgt_imgs, epoch):
        #Encoders
        src_h, _, _ = self.forward_encoder(src_imgs, mask_ratio=0)
        tgt_h, _, _ = self.forward_encoder(tgt_imgs, mask_ratio=0)

        src_p, _, _ = self.forward_encoder(self.perturb(src_imgs), mask_ratio=0)
        tgt_p, _, _ = self.forward_encoder(self.perturb(tgt_imgs), mask_ratio=0)
        
        #f_long encoders
        src_z, _, _ = self.ema_model.forward_encoder(src_imgs, mask_ratio=0)
        tgt_z, _, _ = self.ema_model.forward_encoder(tgt_imgs, mask_ratio=0)
            
        
        #CA3    
        src_h_ca3_cls = self.ca3(src_h[:, 0])
        tgt_h_ca3_cls = self.ca3(tgt_h[:, 0])
        src_h_ca3 = self.ca3(src_h)
        tgt_h_ca3 = self.ca3(tgt_h)

        #KL (symmetric loss1)
        # Posterior distribution from both images
        post_h1 = torch.cat([src_h_ca3_cls, tgt_p[:, 0]], -1)
        post_logits1 = self.to_posterior(post_h1)
            
        post_dist1 = self.make_dist(post_logits1)
        post_z1 = post_dist1.rsample()

        # Prior distribution only from current images
        prior_h1 = src_h_ca3_cls

        prior_logits1 = self.to_prior(prior_h1.detach())
        prior_dist1 = self.make_dist(prior_logits1)
        prior_z1 = prior_dist1.rsample()

        #Predictor g
        tgt_pred = self.forward_decoder_fut_latent(src_h_ca3, post_z1)

        #KL (symmetric loss2)
        # Posterior distribution from both images
        post_h2 = torch.cat([tgt_h_ca3_cls, src_p[:, 0]], -1)
        post_logits2 = self.to_posterior(post_h2)
        post_dist2 = self.make_dist(post_logits2)
        post_z2 = post_dist2.rsample()

        # Prior distribution only from current images
        prior_h2 = tgt_h_ca3_cls
        prior_logits2 = self.to_prior(prior_h2.detach())
            
        prior_dist2 = self.make_dist(prior_logits2)
        prior_z2 = prior_dist2.rsample()

        #Predictor g
        src_pred = self.forward_decoder_fut_latent(tgt_h_ca3, post_z2)
    
        kl_loss1, kl_value1 = self.kl_loss(post_logits1, prior_logits1) 
        kl_loss2, kl_value2 = self.kl_loss(post_logits2, prior_logits2) 
        loss_sim1 = (kl_loss1 + kl_loss2)/2 

        mseloss = nn.MSELoss()
        
        loss_sim2 = mseloss(tgt_pred,tgt_z[:,1:,:].detach())/2 + mseloss(src_pred,src_z[:,1:,:].detach())/2

        loss = loss_sim2 + self.kl_scale*loss_sim1 

        return loss, tgt_pred


def phinetv2_vit_small_patch8_dec512d8b(**kwargs):
    model = PhiNetv2(
        patch_size=8,
        embed_dim=384,
        depth=12,
        num_heads=6,
        decoder_embed_dim=384,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def phinetv2_vit_small_patch16_dec512d8b(**kwargs):
    model = PhiNetv2(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        decoder_embed_dim=384,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model

def phinetv2_vit_base_patch16_dec512d8b(**kwargs):
    model = PhiNetv2(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=12,
        decoder_embed_dim=384,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model

def phinetv2_vit_large_patch16_dec512d8b(**kwargs):
    model = PhiNetv2(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model

phinetv2_vit_small_patch8 = phinetv2_vit_small_patch8_dec512d8b  # decoder: 512 dim, 8 blocks
phinetv2_vit_small_patch16 = phinetv2_vit_small_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
phinetv2_vit_base_patch16 = phinetv2_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
phinetv2_vit_large_patch16 = phinetv2_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
