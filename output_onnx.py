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



from ultralytics import YOLO

# model_path = 'finetune/step_7/weights/best.pt'
model_path = 'runs/segment/step_0_finetune/weights/best.pt'
# 載入你剪枝並微調後保存的權重檔
model = YOLO(model_path)  

# 導出成 ONNX
# 預設會使用 opset 12 並自動選擇輸入尺寸（通常為 640×640）
model.export(
    format='onnx',
    imgsz=(640, 640)    # (height, width)
)