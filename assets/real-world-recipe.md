The differences between the real-world EDTR detection model and the original EDTR detection model used in our main manuscript are:

1. We use a much larger training dataset: [COCO2017](https://cocodataset.org/#download) (117k samples) instead of VOC2012 (6k samples).
2. We use a stronger detection backbone, switching from `FasterRCNN_MobileNet_V3_Large_FPN` to `FasterRCNN_ResNet50_FPN_V2`.
3. We use Real-ESRGAN degradation instead of CodeFormer degradation (we found this crucial for generalizing to real-world inputs).

## Training recipe for the real-world EDTR model

1. Train the EDTR model with CodeFormer degradation for 150k iterations.
2. Re-train the model with Real-ESRGAN degradation for an additional 200k iterations.

The training configuration files can be found [here](../configs/det/coco/train).

You can run training with the following commands:

```shell
# codeformer degradation
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --main_process_port 4177 main/det/train_swinir-pre.py --config configs/det/coco/train/000_swinir-pre.yaml  # pre-restoration
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 4177 main/det/train_edtr.py --config configs/det/coco/train/001_edtr-s4-r50.yaml  # edtr

# real-esrgan degradation
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --main_process_port 4177 main/det/train_swinir-pre.py --config configs/det/coco/train/100_swinir-pre-v2.yaml  # pre-restoration 
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 4177 main/det/train_edtr.py --config configs/det/coco/train/101_edtr-s4-r50v2.yaml  # edtr
```

*NOTE*: You can download the pretrained detection model weights for `FasterRCNN_ResNet50_FPN_V2` by referring [here](https://docs.pytorch.org/vision/main/models.html).
