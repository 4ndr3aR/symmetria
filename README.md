# Symmetria
Official repository of the paper "Symmetria: A Synthetic Dataset for Learning in Point Clouds"

## 1. Requirements
PyTorch >= 1.7.0; python >= 3.7; CUDA >= 9.0; GCC >= 4.9; torchvision;

```
pip install -r requirements.txt
```

```
# Chamfer Distance & emd
cd ./extensions/chamfer_dist
python setup.py install --user
cd ./extensions/emd
python setup.py install --user
# PointNet++
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
# GPU kNN
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl

```

## 2. Datasets
All the Symmetria datasets can be found in this [link](http://deeplearning.ge.imati.cnr.it/symmetria-neurips/). Alternatively, you can download the datasets for pretraining with the following commands:

```
wget http://deeplearning.ge.imati.cnr.it/symmetria-neurips/dataset/ssl/SymSSL-10K.zip
wget http://deeplearning.ge.imati.cnr.it/symmetria-neurips/dataset/ssl/SymSSL-50K.zip
```
For information about datasets ShapeNet, ModelNet, ScanObjectNN, and ShapeNetPart, please check this [documentation](DATASET.md).

## 3. Models
### Point-MAE with Symmetria-10K


| Task              | Dataset        | Config                                                          | Acc.       | Download                                                                                      |
| ----------------- | -------------- | --------------------------------------------------------------- | ---------- | --------------------------------------------------------------------------------------------- |
| Pre-training      | Symmetria-10K       | [pretrain-symetryshape-10k.yaml](./cfgs/Point-MAE/pretrain-symetryshape-10k.yaml)                           | N.A.       | [here](https://drive.google.com/file/d/141q74bJ_WNycALwITVPRFrYk_KULXBR_/view?usp=sharing) |
| Classification    | ScanObjectNN   | [finetune_scan_hardest.yaml](./cfgs/Point-MAE/finetune_scan_hardest.yaml) | 82.4%      | [here](https://drive.google.com/file/d/14C353FnvbHOSFCsZz-OTHMyi1Q9V1lTY/view?usp=sharing) |
| Classification    | ScanObjectNN   | [finetune_scan_objbg.yaml](./cfgs/Point-MAE/finetune_scan_objbg.yaml)     | 86.1%      | [here](https://drive.google.com/file/d/1HU5amUWGQK_UIvd0rQ3ranPqZInN0JDh/view?usp=sharing) |
| Classification    | ScanObjectNN   | [finetune_scan_objonly.yaml](./cfgs/Point-MAE/finetune_scan_objonly.yaml) | 87.8%      | [here](https://drive.google.com/file/d/1T80brUJCsoTbHtS_JcsHhtfh_DnX2adb/view?usp=sharing) |
| Classification    | ModelNet40(1k) | [finetune_modelnet.yaml](./cfgs/Point-MAE/finetune_modelnet.yaml)         | 92.5%      | [here](https://drive.google.com/file/d/1cHx2OdITxq8XWC-BEr2PGiKe5pcubkcy/view?usp=sharing) |
| Part segmentation | ShapeNetPart   | [segmentation](./segmentation)                                  | 85.9% mIoU | [here]() |

| Task              | Dataset    | Config                              | 5w10s Acc. (%) | 5w20s Acc. (%) | 10w10s Acc. (%) | 10w20s Acc. (%) |
| ----------------- | ---------- | ----------------------------------- | -------------- | -------------- | --------------- | --------------- |
| Few-shot learning | ModelNet40 | [fewshot.yaml](./cfgs/Point-MAE/fewshot.yaml) | 95.0 ± 2.0     | 97.0 ± 2.9     | 90.7 ± 5.1      | 93.7 ± 4.6      |

### Point-MAE with Symmetria-50K


| Task              | Dataset        | Config                                                          | Acc.       | Download                                                                                      |
| ----------------- | -------------- | --------------------------------------------------------------- | ---------- | --------------------------------------------------------------------------------------------- |
| Pre-training      | Symmetria-50K       | [pretrain-symetryshape-50k.yaml](./cfgs/Point-MAE/pretrain-symetryshape-50k.yaml)                           | N.A.       | [here](https://drive.google.com/file/d/1xZAfjto7oc2cEqKHRc1kSt2mDoyenUF9/view?usp=sharing) |
| Classification    | ScanObjectNN   | [finetune_scan_hardest.yaml](./cfgs/Point-MAE/finetune_scan_hardest.yaml) | 83.0%      | [here](https://drive.google.com/file/d/1wtOIvSxm4TnOI7N4IEcVSlGcLr4pwrUW/view?usp=sharing) |
| Classification    | ScanObjectNN   | [finetune_scan_objbg.yaml](./cfgs/Point-MAE/finetune_scan_objbg.yaml)     | 87.8%      | [here](https://drive.google.com/file/d/1xOL7y0qevy2GJV3OxoR-1uiRH7xs0u7i/view?usp=sharing) |
| Classification    | ScanObjectNN   | [finetune_scan_objonly.yaml](./cfgs/Point-MAE/finetune_scan_objonly.yaml) | 86.6%      | [here](https://drive.google.com/file/d/1B4WjpH0ieTiKWt9HZduheQWUW-yn8bHf/view?usp=sharing) |
| Classification    | ModelNet40(1k) | [finetune_modelnet.yaml](./cfgs/PointMAE/finetune_modelnet.yaml)         | 93.4%      | [here](https://drive.google.com/file/d/1Dg3d8cR20djNVCGtBp1YNeVoQe-Ntb9w/view?usp=sharing) |
| Part segmentation | ShapeNetPart   | [segmentation](./segmentation)                                  | 85.9% mIoU | [here]() |

| Task              | Dataset    | Config                              | 5w10s Acc. (%) | 5w20s Acc. (%) | 10w10s Acc. (%) | 10w20s Acc. (%) |
| ----------------- | ---------- | ----------------------------------- | -------------- | -------------- | --------------- | --------------- |
| Few-shot learning | ModelNet40 | [fewshot.yaml](./cfgs/Point-MAE/fewshot.yaml) | 95.0 ± 2.4     | 97.0 ± 2.5     | 89.5 ± 4.8      | 94.0 ± 3.7      |

### PointGPT-S with Symmetria-10K


| Task              | Dataset        | Config                                                          | Acc.       | Download                                                                                      |
| ----------------- | -------------- | --------------------------------------------------------------- | ---------- | --------------------------------------------------------------------------------------------- |
| Pre-training      | Symmetria-10K       | [pretrain-symmetryshape-10k.yaml](./cfgs/PointGPT-S/pretrain-symmetryshape-10k.yaml)                           | N.A.       | [here](https://drive.google.com/file/d/1ReGU1JQEcJ8A5Z8RWD_cyojAUFT8HE8s/view?usp=sharing) |
| Classification    | ScanObjectNN   | [finetune_scan_hardest.yaml](./cfgs/PointGPT-S/finetune_scan_hardest.yaml) | 84.3%      | [here](https://drive.google.com/file/d/1_e20X_1V61oM5s1Fr-9j1xw1DEnfJjLb/view?usp=sharing) |
| Classification    | ScanObjectNN   | [finetune_scan_objbg.yaml](./cfgs/PointGPT-S/finetune_scan_objbg.yaml)     | 73.8%      | [here](https://drive.google.com/file/d/1K2FeoCRcU9TJtVbnKCHk5hMkfsG6hBtc/view?usp=sharing) |
| Classification    | ScanObjectNN   | [finetune_scan_objonly.yaml](./cfgs/PointGPT-S/finetune_scan_objonly.yaml) | 79.0%      | [here](https://drive.google.com/file/d/1S7uV66UfPLyHDjQCo6PciUi_CnO1PIPN/view?usp=sharing) |
| Classification    | ModelNet40(1k) | [finetune_modelnet.yaml](./cfgs/PointGPT-S/finetune_modelnet.yaml)         | 93.0%      | [here](https://drive.google.com/file/d/1g3zzhRMBZZQ9GZwc9k-i2kfdZqp8-NnK/view?usp=sharing) |
| Part segmentation | ShapeNetPart   | [segmentation](./segmentation)                                  | 85.8% mIoU | [here]() |

| Task              | Dataset    | Config                              | 5w10s Acc. (%) | 5w20s Acc. (%) | 10w10s Acc. (%) | 10w20s Acc. (%) |
| ----------------- | ---------- | ----------------------------------- | -------------- | -------------- | --------------- | --------------- |
| Few-shot learning | ModelNet40 | [fewshot.yaml](./cfgs/PointGPT-S/fewshot.yaml) | 94.0 ± 3.5     | 98.0 ± 2.1     | 90.5 ± 5.6      | 94.0 ± 4.2      |

### PointGPT-S with Symmetria-50K


| Task              | Dataset        | Config                                                          | Acc.       | Download                                                                                      |
| ----------------- | -------------- | --------------------------------------------------------------- | ---------- | --------------------------------------------------------------------------------------------- |
| Pre-training      | Symmetria-50K       | [pretrain-symmetryshape-50k.yaml](./cfgs/PointGPT-S/pretrain-symmetryshape-50k.yaml)                           | N.A.       | [here](https://drive.google.com/file/d/1gh9z5kDqVlu6j0HxdmP7zl7jeSgWihbg/view?usp=sharing) |
| Classification    | ScanObjectNN   | [finetune_scan_hardest.yaml](./cfgs/PointGPT-S/finetune_scan_hardest.yaml) | 84.9%      | [here](https://drive.google.com/file/d/1acmpd3LC563uXDX8d7QMDAiwGjoDxuUQ/view?usp=sharing) |
| Classification    | ScanObjectNN   | [finetune_scan_objbg.yaml](./cfgs/PointGPT-S/finetune_scan_objbg.yaml)     | 74.4%      | [here](https://drive.google.com/file/d/1CvKc3asN1a6F4t-XSA9MLzlryeWocsHF/view?usp=sharing) |
| Classification    | ScanObjectNN   | [finetune_scan_objonly.yaml](./cfgs/PointGPT-S/finetune_scan_objonly.yaml) | 79.4%      | [here](https://drive.google.com/file/d/1CODC1IMGwKjfuJRbtwwP_xZpzgGUCnZY/view?usp=sharing) |
| Classification    | ModelNet40(1k) | [finetune_modelnet.yaml](./cfgs/PointGPT-S/finetune_modelnet.yaml)         | 93.2%      | [here](https://drive.google.com/file/d/1y0jhZovfCiMR4WOKO9JM-uBYWMQ_8bRh/view?usp=sharing) |
| Part segmentation | ShapeNetPart   | [segmentation](./segmentation)                                  | 85.8% mIoU | [here](https://drive.google.com/file/d/1WBY7VSBex1tLcRijB77DNfK1p8Mrq056/view?usp=sharing) |

| Task              | Dataset    | Config                              | 5w10s Acc. (%) | 5w20s Acc. (%) | 10w10s Acc. (%) | 10w20s Acc. (%) |
| ----------------- | ---------- | ----------------------------------- | -------------- | -------------- | --------------- | --------------- |
| Few-shot learning | ModelNet40 | [fewshot.yaml](./cfgs/PointGPT-S/fewshot.yaml) | 95.7 ± 3.2     | 98.0 ± 1.5     | 91.1 ± 4.9      | 94.3 ± 4.0      |

## 4. Pre-training
To pretrain a model , run the following command:

```
CUDA_VISIBLE_DEVICES=<GPU> python main.py --config cfgs/<MODEL_NAME>/pretrain.yaml --exp_name <output_file_name>
```

## 5. Fine-tuning
Fine-tuning on ScanObjectNN, run the following command:

```
CUDA_VISIBLE_DEVICES=<GPUs> python main.py --config cfgs/<MODEL_NAME>/finetune_scan_hardest.yaml \
--finetune_model --exp_name <output_file_name> --ckpts <path/to/pre-trained/model>
```

Fine-tuning on ModelNet40, run the following command:

```
CUDA_VISIBLE_DEVICES=<GPUs> python main.py --config cfgs/<MODEL_NAME>/finetune_modelnet.yaml \
--finetune_model --exp_name <output_file_name> --ckpts <path/to/pre-trained/model>
```

Voting on ModelNet40, run the following command:

```
CUDA_VISIBLE_DEVICES=<GPUs> python main.py --test --config cfgs/<MODEL_NAME>/finetune_modelnet.yaml \
--exp_name <output_file_name> --ckpts <path/to/best/fine-tuned/model>
```

Few-shot learning, run the following command:

```
CUDA_VISIBLE_DEVICES=<GPUs> python main.py --config cfgs/<MODEL_NAME>/fewshot.yaml --finetune_model \
--ckpts <path/to/pre-trained/model> --exp_name <output_file_name> --way <5 or 10> --shot <10 or 20> --fold <0-9>
```

Part segmentation on ShapeNetPart, run the following command:

```
cd segmentation
python main.py --ckpts <path/to/pre-trained/model> --root path/to/data --learning_rate 0.0002 --epoch 300 --model_name <MODEL_NAME>
```

## Acknowledgements
Our codes are built upon [PointGPT](https://github.com/CGuangyan-BIT/PointGPT), [Point-MAE](https://github.com/Pang-Yatian/Point-MAE), [Point-BERT](https://github.com/lulutang0608/Point-BERT), [Pointnet2_PyTorch](https://github.com/erikwijmans/Pointnet2_PyTorch) and [Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch).