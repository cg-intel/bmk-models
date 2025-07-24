import torch
from ultralytics import YOLO

def export_yolo_onnx(model_name, output_path, input_height=640, input_width=640, batch_size=1):
    if model_name.startswith('yolov5'):
        model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)
    else:
        model = YOLO(f'{model_name}.pt').model

    model.eval()

    dummy_input = torch.randn(batch_size, 3, input_height, input_width)

    torch.onnx.export(
        model,
        (dummy_input,),
        output_path,
        input_names=['input'],
        output_names=['output'],
        opset_version=11,
        do_constant_folding=True,
        dynamic_axes=None
    )

    print(f"ONNX model exported to: {output_path}")
    print(f"Input shape: [{batch_size}, 3, {input_height}, {input_width}]")

if __name__ == '__main__':
    for model in ['yolov5s', 'yolov5l', 'yolov8s', 'yolov8l', 'yolo11s', 'yolo11l']:
        for size in [640, 2048]:
            export_yolo_onnx(model, f"models/{model}-{size}.onnx", input_height=size, input_width=size)






