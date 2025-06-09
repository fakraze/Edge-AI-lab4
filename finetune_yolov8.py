import argparse
import time
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
    return parser.parse_args()


def main():
    args = parse_args()
    # prepare logging
    start_time = time.time()
    log_f = open('finetune.log', 'a', buffering=1)

    # load model
    model = YOLO(args.model)

    # define callback inside main so it captures start_time & log_f
    def log_save(trainer):
        epoch = trainer.epoch
        metrics = trainer.metrics or {}
        box_map  = metrics.get('metrics/mAP50-95(B)', 0.0)
        mask_map = metrics.get('metrics/mAP50-95(M)', 0.0)
        elapsed  = time.time() - start_time
        # console
        print(f"[Epoch {epoch}] saved → box mAP={box_map:.4f}, mask mAP={mask_map:.4f}, elapsed={int(elapsed//3600)} h {int((elapsed//60)%60)} m {int(elapsed%60)} s")
        # file
        log_f.write(f"[Epoch {epoch}] box mAP={box_map:.4f}, mask mAP={mask_map:.4f}, elapsed={int(elapsed//3600)} h {int((elapsed//60)%60)} m {int(elapsed%60)} s\n")
        log_f.flush()

    model.add_callback('on_model_save', log_save)

    # start training
    if ',' in args.device:
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
        # final validation (optional)
        print("Running final validation...")
        torch.cuda.empty_cache()
        model.val(
            data=args.data,
            batch=args.val_batch,
            imgsz=args.imgsz,
            device=args.device,
            workers=args.workers
        )
    else:
        # single-GPU or CPU: iterative per-epoch
        for epoch in range(1, args.epochs + 1):
            print(f"Epoch {epoch}/{args.epochs} - Training")
            model.train(
                data=args.data,
                epochs=1,
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
            
    print("Fine-tuning complete.")
    log_f.close()

if __name__ == '__main__':
    main()
