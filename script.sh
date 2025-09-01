# Inference
## Classification
CUDA_VISIBLE_DEVICES=0 accelerate launch --main_process_port 4177 main/cls/test_edtr.py --config configs/cls/cub200/test/006_edtr-s1.yaml --save-img            # EDTR
CUDA_VISIBLE_DEVICES=0 accelerate launch --main_process_port 4177 main/cls/test_diffbir.py --config configs/cls/cub200/test/005_diffbir.yaml --save-img         # DiffBIR
CUDA_VISIBLE_DEVICES=0 accelerate launch --main_process_port 4177 main/cls/test_cls.py --config configs/cls/cub200/test/000_oracle.yaml --save-img              # Oracle
CUDA_VISIBLE_DEVICES=0 accelerate launch --main_process_port 4177 main/cls/test_cls.py --config configs/cls/cub200/test/001_lq.yaml --save-img                  # No-restoration
CUDA_VISIBLE_DEVICES=0 accelerate launch --main_process_port 4177 main/cls/test_swinir.py --config configs/cls/cub200/test/004_swinir-sr4ir.yaml --save-img     # SwinIR-SR4IR

## Segmentation
CUDA_VISIBLE_DEVICES=0 accelerate launch --main_process_port 4177 main/seg/test_edtr.py --config configs/seg/voc2012/test/007_edtr-s4.yaml --save-img           # EDTR
CUDA_VISIBLE_DEVICES=0 accelerate launch --main_process_port 4177 main/seg/test_diffbir.py --config configs/seg/voc2012/test/005_diffbir.yaml --save-img        # DiffBIR
CUDA_VISIBLE_DEVICES=0 accelerate launch --main_process_port 4177 main/seg/test_seg.py --config configs/seg/voc2012/test/000_oracle.yaml --save-img             # Oracle
CUDA_VISIBLE_DEVICES=0 accelerate launch --main_process_port 4177 main/seg/test_seg.py --config configs/seg/voc2012/test/001_lq.yaml --save-img                 # No-restoration
CUDA_VISIBLE_DEVICES=0 accelerate launch --main_process_port 4177 main/seg/test_swinir.py --config configs/seg/voc2012/test/004_swinir-sr4ir.yaml --save-img    # SwinIR-SR4IR

## Detection
CUDA_VISIBLE_DEVICES=0 accelerate launch --main_process_port 4177 main/det/test_edtr.py --config configs/det/voc2012/test/007_edtr-s4.yaml --save-img           # EDTR
CUDA_VISIBLE_DEVICES=0 accelerate launch --main_process_port 4177 main/det/test_diffbir.py --config configs/det/voc2012/test/005_diffbir.yaml --save-img        # DiffBIR
CUDA_VISIBLE_DEVICES=0 accelerate launch --main_process_port 4177 main/det/test_det.py --config configs/det/voc2012/test/000_oracle.yaml --save-img             # Oracle
CUDA_VISIBLE_DEVICES=0 accelerate launch --main_process_port 4177 main/det/test_det.py --config configs/det/voc2012/test/001_lq.yaml --save-img                 # No-restoration
CUDA_VISIBLE_DEVICES=0 accelerate launch --main_process_port 4177 main/det/test_swinir.py --config configs/det/voc2012/test/004_swinir-sr4ir.yaml --save-img    # SwinIR-SR4IR

# Training
## Classification
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --main_process_port 4177 main/cls/train_swinir-pre.py --config configs/cls/cub200/train/002_swinir-pre.yaml          # SwinIR-Pre
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 4177 main/cls/train_edtr.py --config configs/cls/cub200/train/007_edtr-s4.yaml               # EDTR
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 4177 main/cls/train_diffbir.py --config configs/cls/cub200/train/005_diffbir.yaml            # DiffBIR
CUDA_VISIBLE_DEVICES=0 accelerate launch --main_process_port 4177 main/cls/train_cls.py --config configs/cls/cub200/train/000_oracle.yaml                       # Oracle
CUDA_VISIBLE_DEVICES=0 accelerate launch --main_process_port 4177 main/cls/train_cls.py --config configs/cls/cub200/train/001_lq.yaml                           # No-restoration
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --main_process_port 4177 main/cls/train_swinir-sr4ir.py --config configs/cls/cub200/train/004_swinir-sr4ir.yaml      # SwinIR-SR4IR

## Segmentation
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --main_process_port 4177 main/seg/train_swinir-pre.py --config configs/seg/voc2012/train/002_swinir-pre.yaml         # SwinIR-Pre
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 4177 main/seg/train_edtr.py --config configs/seg/voc2012/train/007_edtr-s4.yaml              # EDTR
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 4177 main/seg/train_diffbir.py --config configs/seg/voc2012/train/005_diffbir.yaml           # DiffBIR
CUDA_VISIBLE_DEVICES=0 accelerate launch --main_process_port 4177 main/seg/train_seg.py --config configs/seg/voc2012/train/000_oracle.yaml                      # Oracle
CUDA_VISIBLE_DEVICES=0 accelerate launch --main_process_port 4177 main/seg/train_seg.py --config configs/seg/voc2012/train/001_lq.yaml                          # No-restoration
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --main_process_port 4177 main/seg/train_swinir-sr4ir.py --config configs/seg/voc2012/train/004_swinir-sr4ir.yaml     # SwinIR-SR4IR

## Detection
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --main_process_port 4177 main/det/train_swinir-pre.py --config configs/det/voc2012/train/002_swinir-pre.yaml         # SwinIR-Pre
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 4177 main/det/train_edtr.py --config configs/det/voc2012/train/007_edtr-s4.yaml              # EDTR
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 4177 main/det/train_diffbir.py --config configs/det/voc2012/train/005_diffbir.yaml           # DiffBIR
CUDA_VISIBLE_DEVICES=0 accelerate launch --main_process_port 4177 main/det/train_det.py --config configs/det/voc2012/train/000_oracle.yaml                      # Oracle
CUDA_VISIBLE_DEVICES=0 accelerate launch --main_process_port 4177 main/det/train_det.py --config configs/det/voc2012/train/001_lq.yaml                          # No-restoration
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --main_process_port 4177 main/det/train_swinir-sr4ir.py --config configs/det/voc2012/train/004_swinir-sr4ir.yaml     # SwinIR-SR4IR
