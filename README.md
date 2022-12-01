# RandLA-Net: Efficient Semantic Segmentation of Large-Scale Point Clouds

# Contents

- [RandLA-Net](#randlanet)
    - [Model Architecture](#model-architecture)
    - [Requirements](#requirements)
        - [Install dependencies](#install-dependencies)
    - [Dataset](#dataset)
        - [Preparation](#preparation)
        - [Directory structure of dataset](#directory-structure-of-dataset)
    - [Quick Start](#quick-start)
    - [Script Description](#script-description)
        - [Scripts and Sample Code](#scripts-and-sample-code)
        - [Script Parameter](#script-parameter)
    - [Training](#training)
        - [Training Process](#training-process)
        - [Training Result](#training-result)
    - [Evaluation](#evaluation)
        - [Evaluation Process](#evaluation-process)
        - [Evaluation Result](#evaluation-result)
    - [Performance](#performance)
        - [Training Performance](#training-performance)
        - [Inference Performance](#inference-performance)
        - [S3DIS Area 5](#s3dis-area-5)
        - [S3DIS 6 fold cross validation](#s3dis-6-fold-cross-validation)
    - [Reference](#reference)

# [RandLA-Net](#contents)

Mindspore implementation for ***"RandLA-Net: Efficient Semantic Segmentation of Large-Scale Point Clouds"***

Please read the [original paper](https://arxiv.org/abs/1911.11236)
or [original tensorflow implementation](https://github.com/QingyongHu/RandLA-Net) for more detailed information.

## [Model Architecture](#contents)

![RandLA-Net Framework](./figs/framework.png)

## [Requirements](#contents)

- Hardware
    - For Ascend: Ascend 910.
    - For GPU: cuda==11.1

- Framework
    - Mindspore = 1.7.0

- Third Package
    - Python==3.7.5
    - pandas==1.3.5
    - scikit-learn==0.21.3
    - numpy==1.21.5

### [Install dependencies](#contents)

1. `pip install -r requirements.txt`
2. `cd third_party` & `bash compile_op.sh`

## [Dataset](#contents)

### [Preparation](#contents)

1. Download S3DIS dataset from
   this [link](https://docs.google.com/forms/d/e/1FAIpQLScDimvNMCGhy_rmBA2gHfDu3naktRm6A8BPwAWWDv-Uhm6Shw/viewform?c=0&w=1)
   .
2. Uncompress `Stanford3dDataset_v1.2_Aligned_Version.zip` to `dataset/S3DIS`.
3. run `data_prepare_s3dis.py` (in `src/utils/data_prepare_s3dis.py`) to process data. The processed data will be stored
   in `input_0.040` and `original_ply` folders.

### [Directory structure of dataset](#contents)

```html
dataset
└──S3DIS                                     #  S3DIS dataset
   ├── input_0.040
   │   ├── *.ply
   │   ├── *_proj.pkl
   │   └── *_KDTree.pkl
   ├── original_ply
   │   └── *.ply
   │
   └── Stanford3dDataset_v1.2_Aligned_Version
```

## [Quick Start](#contents)

For GPU:

S3DIS Area 5

```shell
bash scripts/train_s3dis_6fold_gpu.sh
bash scripts/eval_s3dis_6fold_gpu.sh
```

S3DIS 6 fold cross validation

```shell
bash scripts/train_s3dis_6fold_gpu.sh
bash scripts/eval_s3dis_6fold_gpu.sh
```

For Ascend:

S3DIS Area 5

```shell
bash scripts/train_s3dis_area5_ascend.sh
bash scripts/eval_s3dis_area5_ascend.sh
```

S3DIS 6 fold cross validation

```shell
bash scripts/train_s3dis_6fold_ascend.sh
bash scripts/eval_s3dis_6fold_ascend.sh
```

## [Script Description](#contents)

### [Scripts and Sample Code](#contents)

```html
RandLA
├── scripts
│   ├── eval_s3dis_6fold_ascend.sh           # Evaluate: S3DIS 6 fold cross validation on Ascend
│   ├── eval_s3dis_6fold_gpu.sh              # Evaluate: S3DIS 6 fold cross validation on GPU
│   ├── eval_s3dis_area5_ascend.sh           # Evaluate: S3DIS Area 5 on Ascend
│   ├── eval_s3dis_area5_gpu.sh              # Evaluate: S3DIS Area 5 on GPU
│   ├── train_s3dis_6fold_ascend.sh          # Train: S3DIS 6 fold on Ascend
│   ├── train_s3dis_6fold_gpu.sh             # Train: S3DIS 6 fold on GPU
│   ├── train_s3dis_area5_ascend.sh          # Train: S3DIS Area 5 on Ascend
│   └── train_s3dis_area5_gpu.sh             # Train: S3DIS Area 5 on GPU
├── src
|   ├── data                                 # class and functions for Mindspore dataset
│   │   ├── dataset.py                       # dataset class for train
│   ├── model                                # network architecture
│   │   ├── model.py                         # combine loss function with network
│   └── utils
│       ├── data_prepare_s3dis.py            # data processor for s3dis dataset
│       ├── helper_ply.py                    # file utils
│       ├── logger.py                        # logger
│       └── tools.py                         # DataProcessing and Config
├── third_party
|   ├── cpp_wrappers                         # dependency for point cloud subsampling
|   ├── meta                                 # meta information for data processor
|   ├── nearest_neighbors                    # dependency for point cloud nearest_neighbors
|   └── compile_op.sh                        # shell for installing dependencies, including cpp_wrappers and nearest_neighbors
|
├── 6_fold_cv.py
├── README.md
├── eval.py
├── requirements.txt
└── train.py
```

### [Script Parameter](#contents)

we use `train_s3dis_area5_gpu.sh` as an example

```shell
python train.py \
  --epochs 100 \
  --batch_size 6 \
  --val_area Area_5 \
  --device_target GPU \
  --device_id 0 \
  --outputs_dir ./runs \
  --scale \
  --name randla_Area-5-gpu
```

The following table describes the arguments. Some default Arguments are defined in `src/utils/tools.py`. You can change freely as you want.

| Config Arguments  |                         Explanation                          |
| :---------------: | :----------------------------------------------------------: |
| `--epoch`            | number of epochs for training |
| `--batch_size`       | batch size |
| `--val_area`         | which area to validate              |
| `--device_target`    | chose "Ascend" or "GPU" |
| `--device_id`        | which Ascend AI core/GPU to run(default:0) |
| `--outputs_dir`      | where stores log and network weights  |
| `--scale`            | use auto loss scale or not  |
| `--name`             | experiment name, which will combine with outputs_dir. The output files for current experiments will be stores in `outputs_dir/name`  |

## [Training](#contents)

### [Training Process](#contents)

For GPU on S3DIS area 5:

```shell
bash train_s3dis_area5_gpu.sh
```

For Ascend on S3DIS area 5:

```shell
bash train_s3dis_area5_ascend.sh
```

For GPU on S3DIS 6 fold:

```shell
bash train_s3dis_6fold_gpu.sh
```

For Ascend on S3DIS 6 fold:

```shell
bash train_s3dis_6fold_ascend.sh
```

### [Training Result](#contents)

Using `bash scripts/train_s3dis_area5_ascend.sh` as an example:

Training results will be stored in `/runs/randla_Area-5-ascend` , which is determined
by `{args.outputs_dir}/{args.name}/ckpt`. For example:

```html
runs
├── randla_Area-5-ascend
    ├── 2022-10-24_time_11_23_40_rank_0.log
    └── ckpt
         ├── randla_1_500.ckpt
         ├── randla_2_500.ckpt
         └── ....
```

## [Evaluation](#contents)

### [Evaluation Process](#contents)

For GPU on S3DIS area 5:

```shell
bash eval_s3dis_area5_gpu.sh
```

For Ascend on S3DIS area 5:

```shell
bash eval_s3dis_area5_gpu.sh
```

For GPU on S3DIS 6 fold cross validation:

```shell
bash eval_s3dis_6fold_gpu.sh
```

For Ascend on S3DIS 6 fold cross validation:

```shell
bash eval_s3dis_6fold_gpu.sh
```

Note: Before you start eval, please guarantee `--model_path` is equal to
`{args.outputs_dir}/{args.name}` when training.

### [Evaluation Result](#contents)

Ascend S3DIS Area 5 result

```shell
Area_5_office_5 Acc:0.9362713695425551
Area_5_office_6 Acc:0.9065830704297408
Area_5_office_7 Acc:0.8913033421714497
Area_5_office_8 Acc:0.9357954006048742
Area_5_office_9 Acc:0.8978723295191693
Area_5_pantry_1 Acc:0.7361812911462947
Area_5_storage_1 Acc:0.6030527047227359
Area_5_storage_2 Acc:0.5668263063261647
Area_5_storage_3 Acc:0.7194852396632069
Area_5_storage_4 Acc:0.8087958823812531
--------------------------------------------------------------------------------------
62.48 | 92.81 96.89 79.23  0.00 30.78 57.55 35.78 77.12 87.15 60.27 71.66 69.81 53.20
--------------------------------------------------------------------------------------
```

## [Performance](#contents)

### [Training Performance](#contents)

| Parameters                 | Ascend 910                                                   | GPU (3090) |
| -------------------------- | ------------------------------------------------------------ | ----------------------------------------------|
| Model Version              | RandLA-Net                                                   | RandLA-Net                                    |
| Resource                   | Ascend 910; CPU 2.60GHz, 24cores; Memory 96G; OS Euler2.8    | Nvidia GeForce RTX 3090                       |
| uploaded Date              | 11/26/2022 (month/day/year)                                  | 11/26/2022 (month/day/year)                   |
| MindSpore Version          | 1.7.0                                                        | 1.7.0                                         |
| Dataset                    | S3DIS                                                        | S3DIS                                         |
| Training Parameters        | epoch=100, batch_size = 6                                    | epoch=100, batch_size = 6                     |
| Optimizer                  | Adam                                                         | Adam                                          |
| Loss Function              | Softmax Cross Entropy                                        | Softmax Cross Entropy                         |
| outputs                    | feature vector + probability                                 | feature vector + probability                  |
| Speed                      | 1181 ms/step                                                 | 478 ms/step                                   |
| Total time                 | About 16 h 24 mins                                           | About 7 h 30 mins                             |
| Checkpoint                 | 58 MB (.ckpt file)                                           | 58 MB (.ckpt file)                            |

### [Inference Performance](#contents)

| Parameters          | Ascend                      |   GPU                      |
| ------------------- | --------------------------- |--------------------------- |
| Model Version       | RandLA-Net                  | RandLA-Net                 |
| Resource            | Ascend 910; OS Euler2.8     | Nvidia GeForce RTX 3090    |
| Uploaded Date       | 11/26/2022 (month/day/year) | 11/26/2022 (month/day/year)|
| MindSpore Version   | 1.7.0                       | 1.7.0                      |
| Dataset             | S3DIS                       | S3DIS                      |
| batch_size          | 32                          | 32                          |
| outputs             | feature vector + probability| feature vector + probability  |
| Accuracy            | See following tables        | See following tables       |

### [S3DIS Area 5](#contents)

| Metric | Value(Tensorflow)|  Value(Mindspore, Ascend) |    Value(Mindspore, GPU)      |
| :----: | :------------:   |  :-------------------: |       :-------------------:      |
| mIoU |     62.5%          |         62.5%         |               62.7%               |

### [S3DIS 6 fold cross validation](#contents)

| Metric | Value(Tensorflow)|  Value(Mindspore, Ascend) |    Value(Mindspore, GPU)      |
| :----: | :------------:   |  :-------------------: |       :-------------------:      |
| mIoU |     70.0%          |         69.1%         |               68.8%               |

## [Reference](#contents)

Please kindly cite the original paper references in your publications if it helps your research:

```html
@inproceedings{hu2020randla,
  title={Randla-net: Efficient semantic segmentation of large-scale point clouds},
  author={Hu, Qingyong and Yang, Bo and Xie, Linhai and Rosa, Stefano and Guo, Yulan and Wang, Zhihua and Trigoni, Niki and Markham, Andrew},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={11108--11117},
  year={2020}
}
```