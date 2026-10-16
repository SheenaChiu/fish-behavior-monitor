"""
train_detector.py
=================
訓練 RF-DETR Nano 魚隻偵測器，使用純 DINOv2-S/14 作為 backbone
(非 RF-DETR Nano 預設的 DINOv2-with-registers + windowed attention)。

架構設計:
  整套 pipeline (detection / tracking / action recognition) 共用同一份
  DINOv2-S/14 frozen backbone，形成 unified foundation model framework，
  使 backbone 的輸出可同時供下游追蹤與動作辨識任務使用，無需重複計算。

Backbone 配置:
  - Encoder:                   dinov2_small (純 DINOv2-S，無 register tokens、
                               無 windowed attention)
  - Patch size:                14
  - Num windows:               1 (不分 window)
  - Resolution:                378 (= 14 × 27)
  - Positional encoding size:  27
  - Pretrain weights:          None
      - Backbone 權重自 HF Hub (facebook/dinov2-small) 載入
      - Projector / decoder / head 隨機初始化
      - 註: RF-DETR Nano 的 pretrain weights 使用 patch_size=16，
            與本設定的 patch_size=14 不相容，故不可直接載入。

執行:
  python train_detector.py \
    --dataset_dir output/coco_fish \
    --output_dir output/detector \
    --epochs 80 \
    --batch_size 8 \
    --lr 1e-4 \
    --early_stopping

  訓練完的 best checkpoint 會存為 {output_dir}/checkpoint_best_total.pth，
  使用前可複製 (或改名) 為 weights/detector.pth 以符合推論流程預設路徑。
"""

import argparse
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset_dir', required=True)
    ap.add_argument('--output_dir', default='output/detector')
    ap.add_argument('--epochs', type=int, default=80,
                    help='比原 RF-DETR Nano (50) 多, 因 decoder 從零訓')
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--grad_accum_steps', type=int, default=1)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--lr_encoder', type=float, default=1.5e-4)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--resolution', type=int, default=378,
                    help='必須 = patch_size × positional_encoding_size, '
                         '預設 378 = 14 × 27. 其他選: 392 = 14×28, 672 = 14×48')
    ap.add_argument('--positional_encoding_size', type=int, default=27,
                    help='= resolution / patch_size')
    ap.add_argument('--freeze_backbone', action='store_true', default=True,
                    help='凍結 DINOv2-S backbone (預設 True，與 pipeline 其他階段的 backbone 使用方式一致)')
    ap.add_argument('--early_stopping', action='store_true')
    ap.add_argument('--early_stopping_patience', type=int, default=15)
    ap.add_argument('--tensorboard', action='store_true')
    args = ap.parse_args()

    # 確認 resolution 跟 pos_encoding_size 一致
    patch_size = 14
    expected_res = patch_size * args.positional_encoding_size
    if args.resolution != expected_res:
        print(f"resolution ({args.resolution}) must equal "
              f"patch_size ({patch_size}) × positional_encoding_size "
              f"({args.positional_encoding_size}) = {expected_res}",
              file=sys.stderr)
        sys.exit(1)

    # 確認 resolution 被 (patch_size × num_windows = 14 × 1 = 14) 整除
    if args.resolution % patch_size != 0:
        print(f"resolution must be divisible by patch_size × num_windows = {patch_size}",
              file=sys.stderr)
        sys.exit(1)

    # 確認 dataset 結構
    for split in ['train', 'valid']:
        ann = os.path.join(args.dataset_dir, split, '_annotations.coco.json')
        if not os.path.exists(ann):
            print(f"Missing {ann}。請先將資料集轉換為 COCO 格式 (需包含 _annotations.coco.json)。",
                  file=sys.stderr)
            sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # === Import rfdetr ===
    try:
        from rfdetr import RFDETRNano
        from rfdetr.config import RFDETRNanoConfig
    except ImportError:
        print("rfdetr not installed", file=sys.stderr)
        sys.exit(1)

    print(f"=== Training RF-DETR Nano with vanilla DINOv2-S/14 backbone ===")
    print(f"  Backbone:    DINOv2-S/14 (no registers, no windowed attn, frozen)")
    print(f"  Patch size:  {patch_size}")
    print(f"  Resolution:  {args.resolution}")
    print(f"  Pos enc size: {args.positional_encoding_size}")
    print(f"  Num windows: 1 (no windowing)")
    print(f"  Dataset: {args.dataset_dir}")
    print(f"  Output:  {args.output_dir}")
    print(f"  Epochs:  {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}")
    print()

    # === Build RFDETRNano ===
    #
    # pretrain_weights 設為 None 的原因:
    #   原版 rf-detr-nano.pth 使用 patch_size=16，與此處使用的 patch_size=14 不相容，
    #   直接載入會失敗。
    #
    # encoder='dinov2_small' 會觸發 rfdetr/models/backbone/dinov2.py 內的
    # use_windowed_attn=False 路徑，自 HF Hub (facebook/dinov2-small) 載入
    # DINOv2-S/14 的預訓練權重。
    #
    # 載入結果:
    #   - DINOv2-S/14 backbone: 自 HF Hub 載入，frozen
    #   - Projector / decoder / head: 隨機初始化，由本腳本訓練

    model = RFDETRNano(
        num_classes=1,
        resolution=args.resolution,
        patch_size=patch_size,
        num_windows=1,                              # 不分 window
        positional_encoding_size=args.positional_encoding_size,
        encoder='dinov2_small',                     # 純 DINOv2-S，無 registers、無 windowed attention
        pretrain_weights=None,                      # 原版 rf-detr-nano.pth 與 patch_size=14 不相容
        freeze_encoder=args.freeze_backbone,        # 與 pipeline 其他階段的 backbone 使用方式一致
    )

    # === Train ===
    train_kwargs = dict(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        lr=args.lr,
        lr_encoder=args.lr_encoder,
        num_workers=args.num_workers,
        resolution=args.resolution,
        early_stopping=args.early_stopping,
        early_stopping_patience=args.early_stopping_patience,
        tensorboard=args.tensorboard,
        weight_decay=1e-4,
        warmup_epochs=3,    # decoder 從零訓, warmup 較長有助穩定
    )

    print(f"=== Training kwargs ===")
    for k, v in train_kwargs.items():
        print(f"  {k} = {v}")
    print()

    model.train(**train_kwargs)

    print()
    print("=== Done ===")
    print(f"  Best ckpt: {args.output_dir}/checkpoint_best_total.pth")


if __name__ == "__main__":
    main()