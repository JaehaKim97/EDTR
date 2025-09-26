1. Download original dataset by following this [instruction](../#datasets).

2. Generate your own degraded version by running the following commands:

```shell
python datasets/val_data_generation/gen_cls-dataset.py --config datasets/val_data_generation/config/cls/cub200-deg-mxb.yaml  # CUB200
python datasets/val_data_generation/gen_seg-dataset.py --config datasets/val_data_generation/config/seg/pascalvoc-deg-mxb.yaml  # VOC2012 for segmentation
python datasets/val_data_generation/gen_det-dataset.py --config datasets/val_data_generation/config/det/pascalvoc-deg-mxb.yaml  # VOC2012 for detection
```

You can also customize the degradation settings by modifying the factors in the configuration file.
