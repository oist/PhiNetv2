# PhiNetv2

## Data (K400)

```sh
sh data_preprocessing/k400_downloader.sh
sh data_preprocessing/k400_extractor.sh
```

We need to change the data structure.

```python
python arrange_by_classes_modified.py
```

There is no category replacement_for_corrupted_k400; we need to add it manually. 

## Data-preprocessing
- We resize the data into 256x256 for the efficient loading while training.
- If ffmpeg is not installed in your machine. It needs to install it.
- For our setup, we use $DATA_ROOT=/bucket/YamadaU/Datasets

```python
python data_preprocessing/make_256scale_modified.py --datadir $DATA_ROOT
```

- We additionally provide the code to filter out several not-working videos.

```python
python data_preprocessing/make_labels.py --datadir $DATA_ROOT --filedir train2
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
--data_path /bucket/YamadaU/Datasets/
--log_dir log
--output_dir output_bs32
--norm_pix_loss
--repeated_sampling 2 
```

## Evaluation
We modified the code of RSP (ICML 2024) and CropMAE (ECCV 2024). 

### Davis

### VIT

### JHMDB

## Acknowledgement
We thank Mr. Renaud Vandeghen for his support on running experiments on JHMDB.

