import argparse
import torch
from ultralytics import YOLO
from ultralytics.nn.modules import Conv, Bottleneck, C2f, Detect
import torch.nn as nn

# Custom C2f_v2 for unpickling pruned model
class C2f_v2(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv0 = Conv(c1, self.c, 1, 1)
        self.cv1 = Conv(c1, self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, k=((3,3),(3,3)), e=1.0)
            for _ in range(n)
        )
    def forward(self, x):
        y = [self.cv0(x), self.cv1(x)]
        for m in self.m:
            y.append(m(y[-1]))
        return self.cv2(torch.cat(y, 1))

# Register for unpickling
import __main__ as main_mod; setattr(main_mod, 'C2f_v2', C2f_v2)

# Default config
DEFAULT_CFG = 'yolov8n-seg.yaml'

def parse_args():
    parser = argparse.ArgumentParser(description='YOLOv8n-seg Fine-tuning Script')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to pruned yolov8n-seg checkpoint')
    parser.add_argument('--cfg', type=str, default=DEFAULT_CFG,
                        help='Model YAML for architecture')
    parser.add_argument('--data', type=str, required=True,
                        help='Dataset YAML (e.g., coco128-seg.yaml)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Total epochs')
    parser.add_argument('--train-batch', type=int, default=16,
                        help='Train batch size')
    parser.add_argument('--val-batch', type=int, default=4,
                        help='Validation batch size')
    parser.add_argument('--workers', type=int, default=2,
                        help='DataLoader workers')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Input size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--device', type=str, default='0',
                        help='CUDA device index (single GPU)')
    parser.add_argument('--project', type=str, default='runs/train',
                        help='Output directory')
    parser.add_argument('--name', type=str, default='finetune',
                        help='Run subfolder')
    parser.add_argument('--save-period', type=int, default=1,
                        help='Checkpoint every N epochs')
    parser.add_argument('--val-period', type=int, default=5,
                        help='Validate every N epochs')
    return parser.parse_args()


def main():
    import time
    args = parse_args()
    model = YOLO(args.model)  # includes custom module
    start_time = time.time()  # track elapsed time
    # open log file for appending evaluation results
    log_f = open('finetune.log', 'a', buffering=1)
    args = parse_args()
    model = YOLO(args.model)  # includes custom module

    # Multi-GPU training: single train() call to initialize once
    if ',' in args.device:
        # Multi-GPU training: single train() call without val_period
        print(f"Multi-GPU mode detected: GPUs {args.device}")
        model.train(
            data=args.data,
            epochs=args.epochs,
            batch=args.train_batch,
            imgsz=args.imgsz,
            lr0=args.lr,
            device=args.device,
            project=args.project,
            name=args.name,
            save_period=args.save_period,
            workers=args.workers,
            exist_ok=True
        )
        # After training, run a final validation
        print("Multi-GPU mode: running final validation...")
        torch.cuda.empty_cache()
        val_ret = model.val(
            data=args.data,
            batch=args.val_batch,
            imgsz=args.imgsz,
            device=args.device,
            workers=args.workers
        )
        # extract metrics
        results = getattr(val_ret, 'results_dict', {})
        current_box_map = results.get('metrics/mAP50-95(B)')
        current_mask_map = results.get('metrics/mAP50-95(M)')
        elapsed_time = time.time() - start_time
        # log to console
        print(f"mask mAP = {current_mask_map}, box mAP = {current_box_map:.4f}, elapsed_time: {int(elapsed_time//3600)} hr {int((elapsed_time//60)%60)} min {int(elapsed_time%60)} sec")
        # write to log file
        log_f.write(
            f"mask mAP={current_mask_map}, box mAP={current_box_map:.4f}, elapsed_time={int(elapsed_time//3600)}h{int((elapsed_time//60)%60)}m{int(elapsed_time%60)}s\n"
        )
        log_f.flush()
    # else:
    #     # Single-GPU or CPU: iterative training per epoch
    #     for epoch in range(1, args.epochs + 1):
    #         print(f"Epoch {epoch}/{args.epochs} - Training")
    #         model.train(
    #             data=args.data,
    #             epochs=1,
    #             batch=args.train_batch,
    #             imgsz=args.imgsz,
    #             lr0=args.lr,
    #             device=args.device,
    #             project=args.project,
    #             name=args.name,
    #             save_period=args.save_period,
    #             workers=args.workers,
    #             exist_ok=True
    #         )
    #         if epoch % args.val_period == 0:
    #             print(f"Epoch {epoch} - Validating")
    #             torch.cuda.empty_cache()
    #             model.val(
    #                 data=args.data,
    #                 batch=args.val_batch,
    #                 imgsz=args.imgsz,
    #                 device=args.device,
    #                 workers=args.workers
    #             )

    print("Fine-tuning complete.")

if __name__ == '__main__':
    main()