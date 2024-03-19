# Zero-Reference Low-Light Enhancement via Physical Quadruple Priors (CVPR-24)

This is the official PyTorch code for our paper Zero-Reference Low-Light Enhancement via Physical Quadruple Priors

Authors: Wenjing Wang, Huan Yang, Jianlong Fu, Jiaying Liu

#### 0. Preparation

Create a new conda environment
```
conda env create -f environment.yaml
conda activate quadprior
```

Download the checkpoints from [Google Drive]() or [Baidu Netdisk](https://pan.baidu.com/s/10sKrFyCHBQCVk76Y33wJlw?pwd=j9kv) and save them as follows (required for both training and testing)

- `./checkpoints/COCO-final.ckpt`
- `./checkpoints/main-epoch=00-step=7000.ckpt`
- `./models/cldm_v15.yaml`
- `./models/control_sd15_ini.ckpt`

#### 1. Test

For testing the example images in `./test_data`, simply run:

```
CUDA_VISIBLE_DEVICES=0 python test.py --input_folder ./test_data --same_folder ./output_QuadPrior
```

Then the resulting images can be found in `./output_QuadPrior`.

Expected results can be found in `./output_QuadPrior-reference` for reference.

By default, the inference code uses float16. On NVIDIA GeForce RTX 4090, the inference requires about 13GB of GPU memory.

#### 2. Train

##### 2.1 Data Preparation
Our model is trained solely with the [COCO dataset](https://cocodataset.org/).

In `train.py`, write the path to COCO images as `coco_images`, following the format of [glob.glob](https://docs.python.org/3/library/glob.html).

For example, you may download the train and unlabeled sets and save them as './COCO-2017/train2017/' and './COCO-2017/unlabeled2017/'

##### 2.2 Train
Other parameters can be edited in `train.py`, such as batch size (`batch_size`), number of GPUs (`number_of_gpu`), learning rate (`learning_rate`), how frequently to save visualization (`logger_freq`).

By default, the training uses float16 and deepspeed stage 2, offload optimizer, and CPU checkpointing.

On NVIDIA GeForce RTX 4090, setting 4 batches per GPU takes 20GB memory for each GPU. We use 2 GPUs to train the framework.

If you want to train from scratch, please set `resume_path=''`. Currently it continues training from `checkpoints/COCO-final.ckpt`.

-------

If you find our code useful, please consider citing our paper.

```
@misc{quadprior,
  title={Zero-Reference Low-Light Enhancement via Physical Quadruple Priors}, 
  author={Wenjing Wang and Huan Yang and Jianlong Fu and Jiaying Liu},
  booktitle={IEEE conference on computer vision and pattern recognition (CVPR)}
  year={2024},
}
```

-------

This code is based on [ControlNet](https://github.com/lllyasviel/ControlNet), 
