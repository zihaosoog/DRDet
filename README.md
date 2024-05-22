# DRDet

## Results on Visdrone2019
| Method  | Para(M) | mAP    | $AP_{50}$ | $AP_{75}$ | $AP_{s}$  | $AP_{m}$  | $AP_{l}$  |
|----------|---------|--------|---------|---------|---------|---------|---------|
| ATSS     |          | 12.72  | 22.14   | 13.15   | 4.27    | 19.46   | 32.24   |
| Cascade RCNN |          | 14.43  | 22.13   | 15.53   | 3.28    | 25.24   | 37.34   |
| Faster RCNN* |          | 14.61  | 23.21   | 16.24   | 3.53    | 26.36   | 35.51   |
| TOOD      |          | 14.63  | 24.82   | 14.49   | 5.67    | 21.85   | 33.81   |
| FCOS      |          | 16.19  | 28.34   | 16.63   | 6.93    | 22.37   | 31.61   |
| YOLOXs    | 8.9      | 18.50  | 32.7    | 18.4    | 10.0    | 28.1    | 41.4    |
| YOLOv5s   | 7.2      | 19.10  | 34.4    |         |         |         |         |
| YOLOv8s   | 11.2     | 24.21  | 41.43   |         |         |         |         |
| NaNoDet Plus | 7.8      |26.81  |44.10  |27.52  |16.21  |41.23  |51.09  |
| DRDet (ours) | 10.9     |**28.32**  |**45.84**  |**29.03**  |**17.38**  |**42.66**  |**50.91**  |

模型在2080Ti上的推理耗时 FP32/wo TensorRT
| 模型 | 平均推理耗时（秒） |
|---|---|
| baseline | 0.0185 |
| ours | 0.054 |

****
### Requirements

* Linux or MacOS
* CUDA >= 10.0
* Python >= 3.6
* Pytorch >= 1.7
* experimental support Windows (Notice: Windows not support distributed training before pytorch1.7)

* Install requirements

```shell script
pip install -r requirements.txt
```
    
* Setup NanoDet
```shell script
python setup.py develop
```

****

## How to Train

1. **Prepare dataset**
   convert your dataset annotations to MS COCO format
2. **Start training**

   Baseline and DRDet are both now using [pytorch lightning](https://github.com/PyTorchLightning/pytorch-lightning) for training.

   For both single-GPU or multiple-GPUs, run:

   ```shell script
   python tools/train.py CONFIG_FILE_PATH
   ```
3. **SAHI**

   slice dataset with coco format

    ```bash
    >> pip install sahi
    >> sahi coco slice --image_dir dir/to/images --dataset_json_path dataset.json
    ```
    
   slice the given images and COCO formatted annotations and export them to given output folder directory.
    
    Specify slice height/width size as `--slice_size 512`.
    
    Specify slice overlap ratio for height/width size as `--overlap_ratio 0.2`.
    
    If you want to ignore images with annotations set it add `--ignore_negative_samples` argument.

4. **Test and Vis**

   ```bash
   python demo/demo.py --demo image --config ./config/nanodet-plus-m-1.5x_416.yml --model ./nanodet-plus-m-1.5x_416_checkpoint.ckpt --path ./inferimgs

   ```





   
