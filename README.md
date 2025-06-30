<h1 align="center"> PhiNet v2: A Mask-Free Brain-Inspired <br> Vision Foundation Model from Video</h1>
<div align="center">
  <a href="https://scholar.google.co.jp/citations?user=1cKNu1gAAAAJ&hl=en" target="_blank">Makoto&nbsp;Yamada</a><sup>1</sup> &ensp; <b>&middot;</b> &ensp;
 <a target="_blank">Kian Ming A.&nbsp;Chai</a><sup>2</sup>&ensp; <b>&middot;</b> &ensp;
  <a target="_blank">Ayoub&nbsp;Rhim</a><sup>1</sup> 
  <br>
  <a href="https://riverstone496.github.io/" target="_blank">Satoki&nbsp;Ishikawa</a><sup>3</sup>&ensp; <b>&middot;</b> &ensp;
  <a href="https://sabokrou.github.io/" target="_blank">Mohammad&nbsp;Sabokrou</a><sup>1</sup>&ensp; <b>&middot;</b> &ensp;
  <a href="https://yaohungt.github.io/" target="_blank">Yao-Hung Hubert&nbsp;Tsai</a><sup>1</sup>
  <br>
  <sup>1</sup> Okinawa Institute of Science and Technology &emsp; <sup>2</sup>DSO National Laboratories &emsp; <sup>3</sup>Institute of Science Tokyo &emsp; <br>
</div>
<h3 align="center">[<a href="https://arxiv.org/abs/2505.11129">arXiv</a>]</h3>
<br>
<p align="center">
  <a href="https://raw.githubusercontent.com/oist/PhiNetv2/main/image/FIG1_PhinetV2.png">
    <img src="https://raw.githubusercontent.com/oist/PhiNetv2/main/image/FIG1_PhinetV2.png" width="500"/>
  </a>
</p>


# PhiNetv2

PhiNetv2 pretrain code is heavily based on [Visual Representation Learning with Stochastic Frame Prediction](https://github.com/huiwon-jang/RSP). 

## Environment
All the code was run using Python 3.9.12. 

```sh
mkdir local
mkdir src
mkdir python
cd local/src
wget https://www.python.org/ftp/python/3.9.12/Python-3.9.12.tgz
tar zxvf Python-3.9.12.tgz
./configure --prefix=/home/m/makoto-yamada/local/python/
make
make install

#Add the following in the .bashrc
PATH="$HOME/local/python/bin:$PATH"

source .bashrc

pip3 install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip3 uninstall numpy 
pip3 install numpy==1.26.4
```


## Data (K400)
The original downloader and extractor codes are available at [kinetics-dataset](https://github.com/cvdfoundation/kinetics-dataset).

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
- We follow the data preprocessing procedure of the RSP method.
- The videos are resized to 256×256 for efficient loading during training.
- If ffmpeg is not installed on your machine, please install it beforehand.

```python
cd data_preprocessing
python3 make_256scale_modified.py --datadir ../datasets/k400/videos
```

In our PhiNet v2 experiments, we used a total of 240,355 MP4 files based on our previous preprocessing. However, we found that the above code generates 241,738 videos when preparing the preprocessing pipeline code for publication.
To ensure reproducibility, we provide the following filtering function:

```sh
cd data_preprocessing
sh mv_filtered.sh
```

- We additionally apply the code to filter out several not-working videos (same as RSP code).
Then, we generate a file that is used for training. The link should be direct link. 

```python
cd data_preprocessing
python3 make_labels.py --datadir /home/m/makoto-yamada/work/Python/PhiNetv2/datasets/k400/videos --filedir /home/m/makoto-yamada/work/Python/PhiNetv2/datasets/k400/videos/train2
```

## Pre-training

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

Pretraining takes approximately 2.5 days on 4 A100 GPUs (80GB). By reducing the batch size to 8 or 16, the model can be trained using 4 V100 GPUs instead.

## Pre-trained model

| Dataset  | $J\\&F_m$ (DAVIS)  | mIoU (VIP) | PCK@0.1 (JHMDB) | Download |
| -------- | ------------------ | ---- | ------- | -------- |
| K400     | 60.1               | 33.1 | 45.0    |  [Download](https://huggingface.co/OIST-MLDS-Unit/PhiNetV2/blob/main/phinetv2-vits16.pth)

## Evaluation
To be updated.

### Davis

### VIT

### JHMDB

## License

This project is licensed under the MIT License.  
It includes code originally developed by Huiwon Jang (2024), licensed under the MIT License.
See the [LICENSE](./LICENSE) file for details.

## Citation

If you find this code or model useful in your research, we kindly ask you to cite the following paper:

```bibtex
@article{yamada2025phinet,
  title={PhiNet v2: A Mask-Free Brain-Inspired Vision Foundation Model from Video},
  author={Yamada, Makoto and Chai, Kian Ming A and Rhim, Ayoub and Ishikawa, Satoki and Sabokrou, Mohammad and Tsai, Yao-Hung Hubert},
  journal={arXiv preprint arXiv:2505.11129},
  year={2025}
}
```

## Acknowledgement
We thank Mr. Renaud Vandeghen for his support on running experiments on JHMDB. Chai contributed while on sabbatical leave visiting the Machine Learning and Data Science Unit at Okinawa Institute of Science and Technology, and the Department of Statistics in the University of Oxford. This research was carried out solely for academic purposes using OIST resources.

