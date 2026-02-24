import argparse
import os
from PIL import Image
import supervision as sv
import torch

from rfdetr import (
    RFDETRBase,
    RFDETRLarge,
    RFDETRNano,
    RFDETRSmall,
    RFDETRMedium,
    RFDETRSegPreview,
)
from rfdetr.util.coco_classes import COCO_CLASSES


def build_model(size: str):
    size = size.lower()
    if size == "rfdetr-base":
        return RFDETRBase()
    if size == "rfdetr-large":
        return RFDETRLarge()
    if size == "rfdetr-nano":
        return RFDETRNano()
    if size == "rfdetr-small":
        return RFDETRSmall()
    if size == "rfdetr-medium":
        return RFDETRMedium()
    if size == "rfdetr-seg-preview":
        return RFDETRSegPreview()
    raise ValueError(f"未知模型尺寸: {size}")


def annotate(image: Image.Image, detections: sv.Detections):
    labels = [
        f"{COCO_CLASSES[class_id]} {confidence:.2f}"
        for class_id, confidence in zip(detections.class_id, detections.confidence)
    ]
    if hasattr(detections, "mask") and detections.mask is not None:
        image = sv.MaskAnnotator(color=sv.ColorPalette.ROBOFLOW).annotate(image, detections)
    image = sv.BoxAnnotator(color=sv.ColorPalette.ROBOFLOW).annotate(image, detections)
    image = sv.LabelAnnotator(color=sv.ColorPalette.ROBOFLOW).annotate(image, detections, labels)
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="图片路径",
    )
    parser.add_argument(
        "--size",
        type=str,
        default="rfdetr-base",
        choices=[
            "rfdetr-base",
            "rfdetr-small",
            "rfdetr-medium",
            "rfdetr-large",
            "rfdetr-nano",
            "rfdetr-seg-preview",
        ],
        help="模型尺寸",
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="置信度阈值")
    parser.add_argument("--optimize", action="store_true", help="是否进行推理优化")
    parser.add_argument("--out", type=str, default="examples/out.jpg", help="输出图片路径")
    args = parser.parse_args()

    image_path = args.image
    if not image_path:
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            image_path = filedialog.askopenfilename(
                title="选择图片文件",
                filetypes=[("Images", "*.jpg;*.jpeg;*.png;*.bmp;*.webp"), ("All files", "*.*")]
            )
        except Exception:
            image_path = input("请输入图片路径: ").strip()
    if not image_path or not os.path.exists(image_path):
        raise FileNotFoundError("未选择有效图片文件或路径不存在")
    image = Image.open(image_path).convert("RGB")
    model = build_model(args.size)

    if args.optimize:
        model.optimize_for_inference(compile=True, batch_size=1, dtype=torch.float32)

    detections = model.predict(image, threshold=args.threshold)
    annotated = annotate(image.copy(), detections)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    annotated.save(args.out)
    print(f"保存标注结果到: {args.out}")


if __name__ == "__main__":
    main()
