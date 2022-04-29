# Multi-Camera Vehicle Tracking System for AI City Challenge 2022

The 2nd place solution of track1 (City-Scale Multi-Camera Vehicle Tracking) in the [NVIDIA AI City Challenge](https://www.aicitychallenge.org/) from team 59 ([BOE Technology Group Co., Ltd](https://www.boe.com/en))

## Environment 
- GPU Compute Capability: 7.5
- CUDA: 11.4.2
- Python: 3.8.10
- PyTorch: 1.10.0a0+3fd9dcf
- OpenCV: 4.5.3 (Compilation from the source code [opencv-4.5.3](https://github.com/opencv/opencv/archive/refs/tags/4.5.3.tar.gz), [opencv_contrib-4.5.3](https://github.com/opencv/opencv_contrib/archive/refs/tags/4.5.3.tar.gz))
- Ohter dependencies are in the `requirements.txt`

You can run the command below to get [our docker image](https://hub.docker.com/repository/docker/wangzhen95/deeplearning) that built based on the `pytorch:20.11-py3` from [NVIDIA NGC](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel_21-09.html#rel_21-09).
```
docker pull wangzhen95/deeplearning:v1.3
``` 

## Data Preparation
- Download the datasets [AIC22_Track1_MTMC_Tracking](https://www.aicitychallenge.org/2022-data-and-evaluation/)
and put it under the folder `datasets`.
- Download the pre-trained models from [Google drive](https://drive.google.com/drive/folders/1XRFJyZqJ80z6jv9k70N4ULMypMTZw7wg?usp=sharing).

Make sure the data structure is like:
```
├── AIC22-MTMC
    ├── datasets
    │   └── AIC22_Track1_MTMC_Tracking
    ├── detector
    |   └── yolov5_2022
    |       └── weights
    |           └── yolov5x6.pt
    └── reid
        └── reid_model
            ├── resnet101_ibn_a_2.pth
            ├── resnet101_ibn_a_3.pth
            └── resnext101_ibn_a_2.pth
```
## Running the code

- Modify absolute paths in `config/aic_all.yml`, `config/aic_reid1.yml`, `config/aic_reid2.yml`, `config/aic_reid3.yml`:

```
CHALLENGE_DATA_DIR: '/xxx/AIC22-MCVT/datasets/AIC22_Track1_MTMC_Tracking/'
DET_SOURCE_DIR: '/xxx/AIC22-MCVT/datasets/algorithm_results/detection/images/test/S06/'
DATA_DIR: '/xxx/AIC22-MCVT/datasets/algorithm_results/detect_merge/'
REID_SIZE_TEST: [384, 384]    # 384, 256
ROI_DIR: '/xxx/AIC22-MCVT/datasets/AIC22_Track1_MTMC_Tracking/test/S06/'
CID_BIAS_DIR: '/xxx/AIC22-MCVT/datasets/AIC22_Track1_MTMC_Tracking/cam_timestamp/'
USE_RERANK: True
USE_FF: True
SCORE_THR: 0.1
MCMT_OUTPUT_TXT: 'track1.txt'
```
- Run the docker image:
```
docker run -it --gpus=all --ipc=host -v/xxx/AIC22-MCVT:/xxx/AIC22-MCVT -w /xxx/AIC22-MCVT wangzhen95/deeplearning:v1.3 /bin/bash
```

- Then run:
```
bash ./run_all.sh
```

The final results will locate at path ```./matching/track1.txt```

## Reproduce based on detection and Re-ID results 
If you want rapidly reproduce our results, you can directly download `algorithm_result` from our [google drive](https://drive.google.com/drive/folders/1XRFJyZqJ80z6jv9k70N4ULMypMTZw7wg?usp=sharing).
- Then put it in `AIC22-MCVT/datasets` and modify absolute paths in `config/aic_all.yml`

- Run `bash ./run_mcvt.sh`

The final results will locate at path ```./matching/track1.txt```

## Citation
TBA
