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
        max_w = 0
        max_h = 0

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
            tile_tensor = pil2tensor(tile)
            _, h, w, _ = tile_tensor.shape
            max_h = max(max_h, h)
            max_w = max(max_w, w)
            tiles.append(tile_tensor)
            if masks is not None:
                mask = masks[idx][y1:y2, x1:x2]
                infos.append((x1, y1, x2, y2, torch.from_numpy(mask).unsqueeze(0)))
            else:
                infos.append((x1, y1, x2, y2))
            ids.append(int(classes[idx]))

        padded_tiles = []
        for tile in tiles:
            _, h, w, _ = tile.shape
            pad_w = max_w - w
            pad_h = max_h - h
            if pad_w or pad_h:
                tile_chw = tile.permute(0, 3, 1, 2)
                tile_chw = torch.nn.functional.pad(tile_chw, (0, pad_w, 0, pad_h))
                tile = tile_chw.permute(0, 2, 3, 1)
            padded_tiles.append(tile)
        tiles_tensor = torch.cat(padded_tiles, dim=0) if padded_tiles else torch.empty((0, max_h, max_w, 3))
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
            if task == "segment":
                x1, y1, x2, y2, mask_tensor = info
                width, height = x2 - x1, y2 - y1
                tile_crop = tile_img.crop((0, 0, width, height))
                mask = tensor2pil(mask_tensor)
                canvas.paste(tile_crop, (x1, y1, x2, y2), mask)
            else:
                x1, y1, x2, y2 = info
                width, height = x2 - x1, y2 - y1
                tile_crop = tile_img.crop((0, 0, width, height))
                canvas.paste(tile_crop, (x1, y1, x2, y2))
        return pil2tensor(canvas)
