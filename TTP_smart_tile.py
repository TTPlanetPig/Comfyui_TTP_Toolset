import torch
from PIL import Image
from typing import List, Tuple

from .TTP_toolsets import pil2tensor, tensor2pil


class TTP_Smart_Tile_Batch:
    """Manually crop image regions based on provided bounding boxes."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "boxes": ("LIST",),
            }
        }

    RETURN_TYPES = ("IMAGE", "LIST")
    RETURN_NAMES = ("tiles", "positions")
    FUNCTION = "tile_image"
    CATEGORY = "TTP/Image"

    def tile_image(self, image, boxes: List[Tuple[int, int, int, int]]):
        pil_img = tensor2pil(image.squeeze(0))

        tiles = []
        positions = []
        max_w = 0
        max_h = 0
        for box in boxes:
            if len(box) != 4:
                raise ValueError(f"Each box must contain 4 values, got {box}")
            x1, y1, x2, y2 = map(int, box)
            crop = pil_img.crop((x1, y1, x2, y2))
            tile_tensor = pil2tensor(crop)
            _, h, w, _ = tile_tensor.shape
            max_h = max(max_h, h)
            max_w = max(max_w, w)
            tiles.append(tile_tensor)
            positions.append((x1, y1, x2, y2))

        padded = []
        for tile in tiles:
            _, h, w, _ = tile.shape
            pad_w = max_w - w
            pad_h = max_h - h
            if pad_w or pad_h:
                tile_chw = tile.permute(0, 3, 1, 2)
                tile_chw = torch.nn.functional.pad(tile_chw, (0, pad_w, 0, pad_h))
                tile = tile_chw.permute(0, 2, 3, 1)
            padded.append(tile)

        tiles_tensor = torch.cat(padded, dim=0) if padded else torch.empty((0, max_h, max_w, 3))
        return tiles_tensor, positions


class TTP_Smart_Image_Assy:
    """Reassemble manually cropped tiles back onto the base image."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_image": ("IMAGE",),
                "tiles": ("IMAGE", {"forceInput": True}),
                "positions": ("LIST",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "assemble_image"
    CATEGORY = "TTP/Image"

    def assemble_image(self, base_image, tiles, positions):
        canvas = tensor2pil(base_image.squeeze(0)).copy()
        for idx, tile in enumerate(tiles):
            x1, y1, x2, y2 = positions[idx]
            width = x2 - x1
            height = y2 - y1
            tile_img = tensor2pil(tile.unsqueeze(0)).crop((0, 0, width, height))
            canvas.paste(tile_img, (x1, y1, x2, y2))
        return pil2tensor(canvas)
