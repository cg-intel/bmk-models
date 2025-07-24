Notes:
- Swin-Tiny & EfficientNet, please set
```
HF_ENDPOINT=https://hf-mirror.com
```
- MobileSAM, please download weights from
```
https://github.com/ChaoningZhang/MobileSAM/blob/master/weights/mobile_sam.pt
```

Requirements: 
```
torch==2.7.1
torchvision==0.22.1
onnx==1.18.0
numpy==2.2.6
opencv-python==4.12.0.88
ultralytics==8.3.169
ultralytics-thop==2.0.14
tqdm==4.67.1
mobile_sam @ git+https://github.com/ChaoningZhang/MobileSAM.git@34bbbfdface3c18e5221aa7de6032d7220c6c6a1
```