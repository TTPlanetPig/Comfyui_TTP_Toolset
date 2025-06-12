import torch
import numpy as np
from PIL import Image
from typing import List, Tuple

from .TTP_toolsets import pil2tensor, tensor2pil

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - ultralytics optional
    YOLO = None


class TTP_Smart_Tile_Batch:
    """Split image into smart tiles using detection or segmentation."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model_path": ("STRING", {"default": "yolov8n-seg.pt"}),
            },
            "optional": {
                "task": (["detect", "segment"], {"default": "detect"}),
                "conf": ("FLOAT", {"default": 0.25, "min": 0.01, "max": 1.0}),
                "iou": ("FLOAT", {"default": 0.45, "min": 0.0, "max": 1.0}),
                "padding": ("INT", {"default": 0, "min": 0, "max": 256}),
            },
        }

    RETURN_TYPES = ("IMAGE", "LIST", "LIST")
    RETURN_NAMES = ("tiles", "infos", "ids")
    FUNCTION = "tile_image"
    CATEGORY = "TTP/Image"

    def tile_image(self, image, model_path="yolov8n-seg.pt", task="detect", conf=0.25, iou=0.45, padding=0):
        if YOLO is None:
            raise RuntimeError("ultralytics package is required for smart tiling")

        pil_img = tensor2pil(image.squeeze(0))
        model = YOLO(model_path)
        results = model.predict(pil_img, conf=conf, iou=iou, verbose=False)
        result = results[0]

        tiles: List[torch.Tensor] = []
        infos: List = []
        ids: List[int] = []

        boxes = result.boxes.xyxy.cpu().numpy().astype(int)
        classes = result.boxes.cls.cpu().numpy().astype(int)
        masks = None
        if task == "segment" and result.masks is not None:
            masks = result.masks.data.cpu().numpy()

        width, height = pil_img.size
        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = box.tolist()
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(width, x2 + padding)
            y2 = min(height, y2 + padding)
            tile = pil_img.crop((x1, y1, x2, y2))
            tiles.append(pil2tensor(tile))
            if masks is not None:
                mask = masks[idx][y1:y2, x1:x2]
                infos.append(torch.from_numpy(mask).unsqueeze(0))
            else:
                infos.append((x1, y1, x2, y2))
            ids.append(int(classes[idx]))

        tiles_tensor = torch.stack(tiles, dim=0).squeeze(1) if tiles else torch.empty((0, 3, 0, 0))
        return tiles_tensor, infos, ids


class TTP_Smart_Image_Assy:
    """Reassemble edited smart tiles back to the original image."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_image": ("IMAGE",),
                "tiles": ("IMAGE", {"forceInput": True}),
                "infos": ("LIST",),
                "task": (["detect", "segment"], {"default": "detect"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "assemble_image"
    CATEGORY = "TTP/Image"

    def assemble_image(self, base_image, tiles, infos, task="detect"):
        canvas = tensor2pil(base_image.squeeze(0)).copy()
        for idx, tile in enumerate(tiles):
            tile_img = tensor2pil(tile.unsqueeze(0))
            info = infos[idx]
            if task == "segment" and not isinstance(info, tuple):
                mask = tensor2pil(info)
                bbox = mask.getbbox()
                if bbox:
                    x1, y1, x2, y2 = bbox
                    mask_resized = mask.crop(bbox)
                    tile_crop = tile_img.crop((0, 0, x2 - x1, y2 - y1))
                    canvas.paste(tile_crop, (x1, y1, x2, y2), mask_resized)
            else:
                x1, y1, x2, y2 = info
                canvas.paste(tile_img, (x1, y1, x2, y2))
        return pil2tensor(canvas)
