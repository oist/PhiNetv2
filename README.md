<h1 align="center"> PhiNet v2: A Mask-Free Brain-Inspired <br> Vision Foundation Model from Video</h1>
<div align="center">
  <a href="https://scholar.google.co.jp/citations?user=1cKNu1gAAAAJ&hl=en" target="_blank">Makoto&nbsp;Yamada</a><sup>1</sup> &ensp; <b>&middot;</b> &ensp;
  <a target="_blank">Ayoub&nbsp;Rhim</a><sup>1</sup> &ensp; <b>&middot;</b> &ensp;
  <a target="_blank">Kian Ming A.&nbsp;Chai</a><sup>2</sup>
  <br>
  <a href="https://riverstone496.github.io/" target="_blank">Satoki&nbsp;Ishikawa</a><sup>3</sup>&ensp; <b>&middot;</b> &ensp;
  <a href="https://sabokrou.github.io/" target="_blank">Mohammad&nbsp;Sabokrou</a><sup>1</sup>&ensp; <b>&middot;</b> &ensp;
  <a href="https://yaohungt.github.io/" target="_blank">Yao-Hung Hubert&nbsp;Tsai</a><sup>1</sup>
  <br>
  <sup>1</sup> Okinawa Institute of Science and Technology &emsp; <sup>2</sup>DSO National Laboratories &emsp; <sup>3</sup>Institute of Science Tokyo &emsp; <br>
</div>
<p align="center">
  <a href="https://raw.githubusercontent.com/oist/PhiNetv2/main/image/FIG1_PhinetV2.png">
    <img src="https://raw.githubusercontent.com/oist/PhiNetv2/main/image/FIG1_PhinetV2.png" width="500"/>
  </a>
</p>

# PhiNetv2

PhiNetv2 pretrain code is heavily based on [Visual Representation Learning with Stochastic Frame Prediction](https://github.com/huiwon-jang/RSP). 

## Data (K400)

```sh
cd data_preprocessing
sh k400_downloader.sh
sh k400_extractor.sh
```

We need to change the data structure. Also, there is no category replacement_for_corrupted_k400; we need to add it manually. 

```python
cd data_preprocessing
python3 arrange_by_classes_modified.py ../datasets/k400/
cd ../datasets/k400/videos/train/
cp -r ../../replacement/replacement_for_corrupted_k400/ ./
```

## Data-preprocessing
- We follow the data-preprocessing of RSP method. 
- We resize the data into 256x256 for the efficient loading while training.
- If ffmpeg is not installed in your machine. It needs to install it.

```python
cd data_preprocessing
python3 make_256scale_modified.py --datadir ../datasets/k400/videos
```
- We additionally provide the code to filter out several not-working videos.

Then, we generate a file that is used for training. The link should be direct link. 

```python
cd data_preprocessing
python3 make_labels.py --datadir /home/m/makoto-yamada/work/Python/PhiNetv2/datasets/k400/videos --filedir /home/m/makoto-yamada/work/Python/PhiNetv2/datasets/k400/videos/train2
```

## Pre-training
Our training script is same as the RSP (ICML 2024). 

```sh
python3 -m torch.distributed.launch --nproc_per_node=4 --master_port=29500 main_pretrain_phinetv2.py
--batch_size 32
--accum_iter 6
--model phinetv2_vit_small_patch16
--epochs 400
--warmup_epochs 40
--data_path /home/m/makoto-yamada/work/Python/PhiNetv2/datasets/
--log_dir log
--output_dir output
--norm_pix_loss
--repeated_sampling 2 
```

## Evaluation
We modified the code of RSP (ICML 2024) and CropMAE (ECCV 2024). 

We provide the checkpoint below:  
- ViT-S/16 400 epochs: [Download here](https://example.com/path/to/checkpoint)

### Davis

### VIT

### JHMDB

## Acknowledgement
We thank Mr. Renaud Vandeghen for his support on running experiments on JHMDB.

