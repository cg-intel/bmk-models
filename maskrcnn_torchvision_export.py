import torch
import torchvision
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights

class MaskRCNNWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT).eval()

    def forward(self, x):
        result = self.model([x])[0]
        boxes = torch.as_tensor(result["boxes"])
        labels = torch.as_tensor(result["labels"])
        scores = torch.as_tensor(result["scores"])
        masks = torch.as_tensor(result["masks"])

        return boxes, labels, scores, masks

def export_maskrcnn_onnx(output_path, input_height=800, input_width=800):
    model = MaskRCNNWrapper()
    dummy_input = torch.randn(3, input_height, input_width)

    torch.onnx.export(
        model,
        (dummy_input,),
        output_path,
        input_names=["input"],
        output_names=["boxes", "labels", "scores", "masks"],
        opset_version=11,
        do_constant_folding=True,
        dynamic_axes=None
    )

    print(f"ONNX model exported to: {output_path}")
    print(f"Input shape: [3, {input_height}, {input_width}] (single image)")

if __name__ == "__main__":
    export_maskrcnn_onnx(output_path="models/maskrcnn-resnet50-fpn.onnx", input_height=800, input_width=800)
