import json
import os
import base64
import hashlib
import math
import re
import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageChops, ImageEnhance, ImageDraw, ImageOps
from io import BytesIO
from PIL.PngImagePlugin import PngInfo
import folder_paths
import node_helpers
import torch
import comfy.model_management
import comfy.sd
import comfy.samplers
import comfy.sample
import comfy.utils
import latent_preview
from typing import Any, List, Tuple, Optional, Union, Dict

try:
    from aiohttp import web
    from server import PromptServer
except Exception:
    web = None
    PromptServer = None

try:
    from comfy.cli_args import args
except Exception:
    class _TTPArgs:
        disable_metadata = False

    args = _TTPArgs()

_TTP_SMART_TILE_LOOP_SESSIONS = {}
_TTP_QWENVL3_MODEL_CACHE = {}
_TTP_QWENVL_PROMPT_CACHE = {}

_TTP_QWENVL_PRESETS = {
    "tile_img2img_prompt": {
        "system": "You are an image-to-generation-prompt tagger for tiled image refinement. Describe only visible visual facts useful for image-to-image refinement. Preserve identity, clothing, pose, material, lighting, camera perspective, and local details. Do not invent objects outside the tile. Return only strict JSON with exactly these string fields: label, caption, prompt, negative. Do not repeat the instruction. Do not wrap the JSON in markdown.",
        "instruction": "Analyze the visible tile image and write a concrete img2img refinement prompt for this tile. If a reference image or contact sheet is provided, keep this tile consistent with the global subject, style, lighting, and identity. Return only JSON like {\"label\":\"short label\",\"caption\":\"specific visible description\",\"prompt\":\"specific positive generation prompt\",\"negative\":\"tile-specific negative prompt\"}.",
    },
    "tile_caption_only": {
        "system": "You are a precise visual captioner for tiled image refinement. Describe visible facts only. Return strict JSON with fields: label, caption, prompt, negative.",
        "instruction": "Caption this tile concisely. Put the useful visual caption in both caption and prompt. Keep negative empty unless a visible artifact should be avoided. Return only JSON.",
    },
    "tile_json_strict": {
        "system": "You output strict compact JSON only. No markdown, no explanation, no extra text.",
        "instruction": "Analyze this tile and return exactly {\"label\":\"...\",\"caption\":\"...\",\"prompt\":\"...\",\"negative\":\"...\"}. Keep all fields concise and grounded in visible pixels.",
    },
    "bbox_detect": {
        "system": "You are a visual detector. Return strict JSON only.",
        "instruction": "Detect important objects, faces, eyes, hands, text, and small high-detail regions. Return JSON list only: [{\"label\":\"face\",\"bbox\":[x1,y1,x2,y2],\"score\":0.9,\"priority\":2000,\"prompt_hint\":\"short visual hint\"}]. Coordinates must be normalized 0..1000 integers.",
    },
    "style_material_lighting": {
        "system": "You are an image prompt engineer focused on style, material, lighting, and texture consistency. Return strict JSON with fields: label, caption, prompt, negative.",
        "instruction": "Describe this tile's material, texture, lighting direction, color palette, camera perspective, and local visual details for img2img refinement. Do not invent hidden content. Return only JSON.",
    },
}

def pil2tensor(image: Image) -> torch.Tensor:
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def tensor2pil(t_image: torch.Tensor) -> Image:
    return Image.fromarray(np.clip(255.0 * t_image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
    
def apply_gaussian_blur(image_np, ksize=5, sigmaX=1.0):
    if ksize % 2 == 0:
        ksize += 1  # ksize must be odd
    blurred_image = cv2.GaussianBlur(image_np, (ksize, ksize), sigmaX=sigmaX)
    return blurred_image

        
class TTPlanet_Tile_Preprocessor_Simple:
    def __init__(self, blur_strength=3.0):
        self.blur_strength = blur_strength

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "scale_factor": ("FLOAT", {"default": 2.00, "min": 1.00, "max": 8.00, "step": 0.05}),
                "blur_strength": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 20.0, "step": 0.1}),
            },
            "optional": {}
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image_output",)
    FUNCTION = 'process_image'
    CATEGORY = 'TTP/TILE'

    def process_image(self, image, scale_factor, blur_strength):
        ret_images = []
    
        for i in image:
            # Convert tensor to PIL for processing
            _canvas = tensor2pil(torch.unsqueeze(i, 0)).convert('RGB')
        
            # Convert PIL image to OpenCV format
            img_np = np.array(_canvas)[:, :, ::-1]  # RGB to BGR
        
            # Resize image first if you want blur to apply after resizing
            height, width = img_np.shape[:2]
            new_width = int(width / scale_factor)
            new_height = int(height / scale_factor)
            resized_down = cv2.resize(img_np, (new_width, new_height), interpolation=cv2.INTER_AREA)
            resized_img = cv2.resize(resized_down, (width, height), interpolation=cv2.INTER_LINEAR)
        
            # Apply Gaussian blur after resizing
            img_np = apply_gaussian_blur(resized_img, ksize=int(blur_strength), sigmaX=blur_strength / 2)
        
            # Convert OpenCV back to PIL and then to tensor
            _canvas = Image.fromarray(img_np[:, :, ::-1])  # BGR to RGB
            tensor_img = pil2tensor(_canvas)
            ret_images.append(tensor_img)
    
        return (torch.cat(ret_images, dim=0),)        


class TTP_Image_Tile_Batch:
    def __init__(self, *args, **kwargs):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "tile_width": ("INT", {"default": 1024, "min": 1}),
                "tile_height": ("INT", {"default": 1024, "min": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "LIST", "TUPLE", "TUPLE")
    RETURN_NAMES = ("IMAGES", "POSITIONS", "ORIGINAL_SIZE", "GRID_SIZE")
    FUNCTION = "tile_image"

    CATEGORY = "TTP/Image"

    def tile_image(self, image, tile_width=1024, tile_height=1024):
        image = tensor2pil(image.squeeze(0))
        img_width, img_height = image.size

        if img_width <= tile_width and img_height <= tile_height:
            return (pil2tensor(image), [(0, 0, img_width, img_height)], (img_width, img_height), (1, 1))

        def calculate_step(size, tile_size):
            if size <= tile_size:
                return 1, 0
            else:
                num_tiles = (size + tile_size - 1) // tile_size
                overlap = (num_tiles * tile_size - size) // (num_tiles - 1)
                step = tile_size - overlap
                return num_tiles, step

        num_cols, step_x = calculate_step(img_width, tile_width)
        num_rows, step_y = calculate_step(img_height, tile_height)

        tiles = []
        positions = []
        for y in range(num_rows):
            for x in range(num_cols):
                left = x * step_x
                upper = y * step_y
                right = min(left + tile_width, img_width)
                lower = min(upper + tile_height, img_height)

                if right - left < tile_width:
                    left = max(0, img_width - tile_width)
                if lower - upper < tile_height:
                    upper = max(0, img_height - tile_height)

                tile = image.crop((left, upper, right, lower))
                tile_tensor = pil2tensor(tile)
                tiles.append(tile_tensor)
                positions.append((left, upper, right, lower))

        tiles = torch.stack(tiles, dim=0).squeeze(1)
        return (tiles, positions, (img_width, img_height), (num_cols, num_rows))


class TTP_Image_Assy:
    def __init__(self, *args, **kwargs):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tiles": ("IMAGE",),
                "positions": ("LIST",),
                "original_size": ("TUPLE",),
                "grid_size": ("TUPLE",),
                "padding": ("INT", {"default": 64, "min": 0}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("RECONSTRUCTED_IMAGE",)
    FUNCTION = "assemble_image"

    CATEGORY = "TTP/Image"

    def create_gradient_mask(self, size, direction):
        """Create a gradient mask for blending."""
        mask = Image.new("L", size)
        for i in range(size[0] if direction == 'horizontal' else size[1]):
            value = int(255 * (1 - (i / size[0] if direction == 'horizontal' else i / size[1])))
            if direction == 'horizontal':
                mask.paste(value, (i, 0, i+1, size[1]))
            else:
                mask.paste(value, (0, i, size[0], i+1))
        return mask

    def blend_tiles(self, tile1, tile2, overlap_size, direction, padding):
        """Blend two tiles with a smooth transition."""
        blend_size = padding
        if blend_size > overlap_size:
            blend_size = overlap_size

        if blend_size == 0:
            # No blending, just concatenate the images at the correct overlap
            if direction == 'horizontal':
                result = Image.new("RGB", (tile1.width + tile2.width - overlap_size, tile1.height))
                # Paste the left part of tile1 excluding the overlap
                result.paste(tile1.crop((0, 0, tile1.width - overlap_size, tile1.height)), (0, 0))
                # Paste tile2 directly after tile1
                result.paste(tile2, (tile1.width - overlap_size, 0))
            else:
                # For vertical direction
                result = Image.new("RGB", (tile1.width, tile1.height + tile2.height - overlap_size))
                result.paste(tile1.crop((0, 0, tile1.width, tile1.height - overlap_size)), (0, 0))
                result.paste(tile2, (0, tile1.height - overlap_size))
            return result

        # 以下为原有的混合代码，当 blend_size > 0 时执行
        offset_total = overlap_size - blend_size
        offset_left = offset_total // 2
        offset_right = offset_total - offset_left

        size = (blend_size, tile1.height) if direction == 'horizontal' else (tile1.width, blend_size)
        mask = self.create_gradient_mask(size, direction)

        if direction == 'horizontal':
            crop_tile1 = tile1.crop((tile1.width - overlap_size + offset_left, 0, tile1.width - offset_right, tile1.height))
            crop_tile2 = tile2.crop((offset_left, 0, offset_left + blend_size, tile2.height))
            if crop_tile1.size != crop_tile2.size:
                raise ValueError(f"Crop sizes do not match: {crop_tile1.size} vs {crop_tile2.size}")

            blended = Image.composite(crop_tile1, crop_tile2, mask)
            result = Image.new("RGB", (tile1.width + tile2.width - overlap_size, tile1.height))
            result.paste(tile1.crop((0, 0, tile1.width - overlap_size + offset_left, tile1.height)), (0, 0))
            result.paste(blended, (tile1.width - overlap_size + offset_left, 0))
            result.paste(tile2.crop((offset_left + blend_size, 0, tile2.width, tile2.height)), (tile1.width - offset_right, 0))
        else:
            offset_total = overlap_size - blend_size
            offset_top = offset_total // 2
            offset_bottom = offset_total - offset_top

            size = (tile1.width, blend_size)
            mask = self.create_gradient_mask(size, direction)

            crop_tile1 = tile1.crop((0, tile1.height - overlap_size + offset_top, tile1.width, tile1.height - offset_bottom))
            crop_tile2 = tile2.crop((0, offset_top, tile2.width, offset_top + blend_size))
            if crop_tile1.size != crop_tile2.size:
                raise ValueError(f"Crop sizes do not match: {crop_tile1.size} vs {crop_tile2.size}")

            blended = Image.composite(crop_tile1, crop_tile2, mask)
            result = Image.new("RGB", (tile1.width, tile1.height + tile2.height - overlap_size))
            result.paste(tile1.crop((0, 0, tile1.width, tile1.height - overlap_size + offset_top)), (0, 0))
            result.paste(blended, (0, tile1.height - overlap_size + offset_top))
            result.paste(tile2.crop((0, offset_top + blend_size, tile2.width, tile2.height)), (0, tile1.height - offset_bottom))
        return result

    def assemble_image(self, tiles, positions, original_size, grid_size, padding):
        num_cols, num_rows = grid_size
        reconstructed_image = Image.new("RGB", original_size)

        # First, blend each row independently
        row_images = []
        for row in range(num_rows):
            row_image = tensor2pil(tiles[row * num_cols].unsqueeze(0))
            for col in range(1, num_cols):
                index = row * num_cols + col
                tile_image = tensor2pil(tiles[index].unsqueeze(0))
                prev_right = positions[index - 1][2]
                left = positions[index][0]
                overlap_width = prev_right - left
                if overlap_width > 0:
                    row_image = self.blend_tiles(row_image, tile_image, overlap_width, 'horizontal', padding)
                else:
                    # Adjust the size of row_image to accommodate the new tile
                    new_width = row_image.width + tile_image.width
                    new_height = max(row_image.height, tile_image.height)
                    new_row_image = Image.new("RGB", (new_width, new_height))
                    new_row_image.paste(row_image, (0, 0))
                    new_row_image.paste(tile_image, (row_image.width, 0))
                    row_image = new_row_image
            row_images.append(row_image)

        # Now, blend each row together vertically
        final_image = row_images[0]
        for row in range(1, num_rows):
            prev_lower = positions[(row - 1) * num_cols][3]
            upper = positions[row * num_cols][1]
            overlap_height = prev_lower - upper
            if overlap_height > 0:
                final_image = self.blend_tiles(final_image, row_images[row], overlap_height, 'vertical', padding)
            else:
                # Adjust the size of final_image to accommodate the new row image
                new_width = max(final_image.width, row_images[row].width)
                new_height = final_image.height + row_images[row].height
                new_final_image = Image.new("RGB", (new_width, new_height))
                new_final_image.paste(final_image, (0, 0))
                new_final_image.paste(row_images[row], (0, final_image.height))
                final_image = new_final_image

        return pil2tensor(final_image).unsqueeze(0)


def _ttp_clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


def _ttp_safe_int(value, default=0, minimum=None, maximum=None):
    try:
        if isinstance(value, str) and not value.strip().replace(".", "", 1).lstrip("-").isdigit():
            number = int(default)
        else:
            number = int(round(float(value)))
    except Exception:
        number = int(default)
    if minimum is not None:
        number = max(int(minimum), number)
    if maximum is not None:
        number = min(int(maximum), number)
    return number


def _ttp_safe_float(value, default=0.0, minimum=None, maximum=None):
    try:
        number = float(value)
    except Exception:
        number = float(default)
    if not np.isfinite(number):
        number = float(default)
    if minimum is not None:
        number = max(float(minimum), number)
    if maximum is not None:
        number = min(float(maximum), number)
    return number


def _ttp_safe_bool(value, default=False):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "1", "yes", "on"):
            return True
        if lowered in ("false", "0", "no", "off"):
            return False
    return bool(default)


def _ttp_round_to(value, multiple):
    if multiple <= 1:
        return int(round(value))
    return int(np.ceil(value / multiple) * multiple)


def _ttp_value_is_normalized(value):
    if not isinstance(value, (int, float)):
        return False
    numeric = float(value)
    return 0.0 <= numeric <= 1.0


def _ttp_parse_box_value(value, size, prefer_normalized=False):
    if prefer_normalized and _ttp_value_is_normalized(value):
        return int(round(float(value) * size))
    if isinstance(value, float) and 0.0 <= value <= 1.0:
        return int(round(value * size))
    return int(round(value))


def _ttp_parse_margin_value(value, size):
    if isinstance(value, float) and 0.0 <= value <= 1.0:
        return int(round(value * size))
    return int(round(value))


def _ttp_normalize_tile_box(tile, image_width, image_height, defaults):
    if all(key in tile for key in ("x0", "y0", "x1", "y1")):
        box_values = [tile.get("x0", 0), tile.get("y0", 0), tile.get("x1", image_width), tile.get("y1", image_height)]
        prefer_normalized = all(_ttp_value_is_normalized(value) for value in box_values)
        x0 = _ttp_parse_box_value(tile.get("x0", 0), image_width, prefer_normalized)
        y0 = _ttp_parse_box_value(tile.get("y0", 0), image_height, prefer_normalized)
        x1 = _ttp_parse_box_value(tile.get("x1", image_width), image_width, prefer_normalized)
        y1 = _ttp_parse_box_value(tile.get("y1", image_height), image_height, prefer_normalized)
        x, x_end = sorted((x0, x1))
        y, y_end = sorted((y0, y1))
        w = x_end - x
        h = y_end - y
    else:
        box_values = [
            tile.get("x", 0),
            tile.get("y", 0),
            tile.get("w", tile.get("width", image_width)),
            tile.get("h", tile.get("height", image_height)),
        ]
        prefer_normalized = all(_ttp_value_is_normalized(value) for value in box_values)
        x = _ttp_parse_box_value(tile.get("x", 0), image_width, prefer_normalized)
        y = _ttp_parse_box_value(tile.get("y", 0), image_height, prefer_normalized)
        w = _ttp_parse_box_value(tile.get("w", tile.get("width", image_width)), image_width, prefer_normalized)
        h = _ttp_parse_box_value(tile.get("h", tile.get("height", image_height)), image_height, prefer_normalized)

    x = _ttp_clamp(x, 0, max(0, image_width - 1))
    y = _ttp_clamp(y, 0, max(0, image_height - 1))
    w = _ttp_clamp(w, 1, max(1, image_width - x))
    h = _ttp_clamp(h, 1, max(1, image_height - y))

    pad_value = tile.get("pad", tile.get("padding", defaults.get("pad", 0)))
    blend_value = tile.get("blend", tile.get("feather", defaults.get("blend", 32)))
    pad = _ttp_parse_margin_value(pad_value, max(image_width, image_height))
    blend = _ttp_parse_margin_value(blend_value, max(w, h))

    normalized = {
        "name": str(tile.get("name", f"tile_{tile.get('id', 0)}")),
        "core_box": [x, y, w, h],
        "pad": max(0, pad),
        "blend": max(0, blend),
        "priority": float(tile.get("priority", defaults.get("priority", 50))),
        "importance": float(tile.get("importance", defaults.get("importance", 1.0))),
        "strength": float(tile.get("strength", defaults.get("strength", 1.0))),
        "prompt_tag": str(tile.get("prompt_tag", tile.get("name", ""))),
        "align": bool(tile.get("align", defaults.get("align", False))),
        "source": str(tile.get("source", defaults.get("source", "manual"))),
        "label": str(tile.get("label", tile.get("name", defaults.get("label", "")))),
        "score": float(tile.get("score", defaults.get("score", 1.0))),
        "layer": int(tile.get("layer", defaults.get("layer", 0))),
        "object_id": int(tile.get("object_id", tile.get("id", defaults.get("object_id", 0)))),
        "occlusion_priority": float(tile.get("occlusion_priority", defaults.get("occlusion_priority", 0.0))),
    }
    if isinstance(tile.get("object_mask"), dict):
        normalized["object_mask"] = dict(tile["object_mask"])
    return normalized


def _ttp_expand_grid_tiles(grid, defaults):
    x_lines = grid.get("x", [])
    y_lines = grid.get("y", [])
    if len(x_lines) < 2 or len(y_lines) < 2:
        raise ValueError("Smart Tile grid layout requires at least two x lines and two y lines")

    prefix = str(grid.get("prefix", defaults.get("prefix", "grid")))
    grid_defaults = defaults.copy()
    grid_defaults.update({k: v for k, v in grid.items() if k not in {"x", "y", "prefix"}})

    tiles = []
    for row in range(len(y_lines) - 1):
        for col in range(len(x_lines) - 1):
            x0, x1 = x_lines[col], x_lines[col + 1]
            y0, y1 = y_lines[row], y_lines[row + 1]
            tiles.append({
                **grid_defaults,
                "name": f"{prefix}_{row}_{col}",
                "x": x0,
                "y": y0,
                "w": x1 - x0,
                "h": y1 - y0,
            })
    return tiles


def _ttp_intervals_overlap(a0, a1, b0, b1):
    return min(int(a1), int(b1)) > max(int(a0), int(b0))


def _ttp_neighbor_overlap_edges(tiles_meta, tile_index):
    tile = tiles_meta[int(tile_index)]
    x, y, w, h = tile["core_box"]
    x0, y0, x1, y1 = x, y, x + w, y + h
    pad = max(0, int(tile.get("pad", 0)))
    edges = {"left": 0, "right": 0, "top": 0, "bottom": 0}
    if pad <= 0:
        return edges

    tolerance = 1
    for index, other in enumerate(tiles_meta):
        if index == int(tile_index):
            continue
        ox, oy, ow, oh = other["core_box"]
        ox0, oy0, ox1, oy1 = ox, oy, ox + ow, oy + oh
        vertical_overlap = _ttp_intervals_overlap(y0, y1, oy0, oy1)
        horizontal_overlap = _ttp_intervals_overlap(x0, x1, ox0, ox1)
        if vertical_overlap and (abs(ox1 - x0) <= tolerance or ox0 < x0 < ox1):
            edges["left"] = pad
        if vertical_overlap and (abs(ox0 - x1) <= tolerance or ox0 < x1 < ox1):
            edges["right"] = pad
        if horizontal_overlap and (abs(oy1 - y0) <= tolerance or oy0 < y0 < oy1):
            edges["top"] = pad
        if horizontal_overlap and (abs(oy0 - y1) <= tolerance or oy0 < y1 < oy1):
            edges["bottom"] = pad
    return edges


def _ttp_add_smart_tile_sample_boxes(tiles_meta, image_width, image_height):
    for index, tile in enumerate(tiles_meta):
        x, y, w, h = tile["core_box"]
        edges = _ttp_neighbor_overlap_edges(tiles_meta, index)
        sample_left = _ttp_clamp(x - edges["left"], 0, image_width)
        sample_top = _ttp_clamp(y - edges["top"], 0, image_height)
        sample_right = _ttp_clamp(x + w + edges["right"], 0, image_width)
        sample_bottom = _ttp_clamp(y + h + edges["bottom"], 0, image_height)
        tile["overlap_edges_px_source"] = edges
        tile["sample_box"] = [
            sample_left,
            sample_top,
            max(1, sample_right - sample_left),
            max(1, sample_bottom - sample_top),
        ]
        tile["paste_box"] = [x, y, w, h]
    return tiles_meta


def _ttp_parse_smart_tile_layout(layout_text, image_width, image_height):
    try:
        layout = json.loads(layout_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid Smart Tile layout JSON: {exc}") from exc

    defaults = {}
    raw_tiles = []

    if isinstance(layout, list):
        raw_tiles = layout
    elif isinstance(layout, dict):
        defaults = dict(layout.get("defaults", {}))
        if "grid" in layout:
            raw_tiles.extend(_ttp_expand_grid_tiles(layout["grid"], defaults))
        raw_tiles.extend(layout.get("tiles", []))
        raw_tiles.extend(layout.get("custom", []))
    else:
        raise ValueError("Smart Tile layout must be a JSON list or object")

    if not raw_tiles:
        raw_tiles = [{"name": "full_image", "x": 0, "y": 0, "w": image_width, "h": image_height}]

    normalized_tiles = []
    for index, tile in enumerate(raw_tiles):
        if not isinstance(tile, dict):
            raise ValueError(f"Smart Tile entry {index} must be an object")
        normalized = _ttp_normalize_tile_box({**tile, "id": index}, image_width, image_height, defaults)
        normalized["id"] = index
        normalized_tiles.append(normalized)

    return _ttp_add_smart_tile_sample_boxes(normalized_tiles, image_width, image_height)


def _ttp_create_feather_mask(width, height, feather):
    if feather <= 0:
        return np.ones((height, width, 1), dtype=np.float32)

    yy, xx = np.mgrid[0:height, 0:width]
    distance_left = xx
    distance_right = width - 1 - xx
    distance_top = yy
    distance_bottom = height - 1 - yy
    distance = np.minimum(np.minimum(distance_left, distance_right), np.minimum(distance_top, distance_bottom))
    mask = np.clip(distance / max(1, feather), 0.0, 1.0).astype(np.float32)
    return mask[:, :, None]


def _ttp_create_sample_blend_mask(width, height, overlap_edges, feather, scale_x=1.0, scale_y=1.0):
    mask = np.ones((height, width, 1), dtype=np.float32)
    if feather <= 0:
        return mask

    def scaled_edge(name, scale, limit):
        return _ttp_clamp(int(round(float(overlap_edges.get(name, 0)) * scale)), 0, limit)

    left = scaled_edge("left", scale_x, width)
    right = scaled_edge("right", scale_x, width)
    top = scaled_edge("top", scale_y, height)
    bottom = scaled_edge("bottom", scale_y, height)

    if left > 0:
        blend = min(int(feather), left, width)
        zero_end = max(0, left - blend)
        if zero_end > 0:
            mask[:, :zero_end, :] = 0.0
        if blend > 0:
            ramp = np.linspace(0.0, 1.0, blend, dtype=np.float32)[None, :, None]
            mask[:, zero_end:left, :] *= ramp

    if right > 0:
        blend = min(int(feather), right, width)
        start = max(0, width - right)
        blend_end = min(width, start + blend)
        if blend_end > start:
            ramp = np.linspace(1.0, 0.0, blend_end - start, dtype=np.float32)[None, :, None]
            mask[:, start:blend_end, :] *= ramp
        if blend_end < width:
            mask[:, blend_end:, :] = 0.0

    if top > 0:
        blend = min(int(feather), top, height)
        zero_end = max(0, top - blend)
        if zero_end > 0:
            mask[:zero_end, :, :] = 0.0
        if blend > 0:
            ramp = np.linspace(0.0, 1.0, blend, dtype=np.float32)[:, None, None]
            mask[zero_end:top, :, :] *= ramp

    if bottom > 0:
        blend = min(int(feather), bottom, height)
        start = max(0, height - bottom)
        blend_end = min(height, start + blend)
        if blend_end > start:
            ramp = np.linspace(1.0, 0.0, blend_end - start, dtype=np.float32)[:, None, None]
            mask[start:blend_end, :, :] *= ramp
        if blend_end < height:
            mask[blend_end:, :, :] = 0.0

    return mask


def _ttp_mask_tensor_to_pil_list(masks, image_width, image_height):
    if masks is None:
        return []
    if isinstance(masks, Image.Image):
        return [masks.convert("L").resize((image_width, image_height), Image.Resampling.BILINEAR)]
    if isinstance(masks, (list, tuple)):
        result = []
        for mask in masks:
            result.extend(_ttp_mask_tensor_to_pil_list(mask, image_width, image_height))
        return result

    array = None
    try:
        tensor = masks.detach() if hasattr(masks, "detach") else masks
        tensor = tensor.cpu() if hasattr(tensor, "cpu") else tensor
        array = tensor.numpy() if hasattr(tensor, "numpy") else np.array(tensor)
    except Exception:
        return []

    array = np.asarray(array)
    if array.ndim == 4:
        if array.shape[1] == 1:
            array = array[:, 0, :, :]
        elif array.shape[-1] == 1:
            array = array[:, :, :, 0]
    if array.ndim == 2:
        array = array[None, :, :]
    if array.ndim != 3:
        return []

    result = []
    for item in array:
        item = np.nan_to_num(item.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
        if item.max() > 1.0 or item.min() < 0.0:
            item = np.clip(item, 0.0, 255.0) / 255.0
        item = np.clip(item, 0.0, 1.0)
        pil = Image.fromarray((item * 255.0).astype(np.uint8), mode="L")
        if pil.size != (image_width, image_height):
            pil = pil.resize((image_width, image_height), Image.Resampling.BILINEAR)
        result.append(pil)
    return result


def _ttp_encode_object_mask_data(mask_images, mask_indices, box):
    if not mask_images or not mask_indices:
        return None
    x0, y0, x1, y1 = [int(round(value)) for value in box]
    width = max(1, x1 - x0)
    height = max(1, y1 - y0)
    combined = Image.new("L", (width, height), 0)
    for mask_index in mask_indices:
        if mask_index < 0 or mask_index >= len(mask_images):
            continue
        cropped = mask_images[mask_index].crop((x0, y0, x1, y1)).convert("L")
        if cropped.getbbox() is None:
            continue
        combined = ImageChops.lighter(combined, cropped)
    if combined.getbbox() is None:
        return None
    buffer = BytesIO()
    combined.save(buffer, format="PNG", optimize=True)
    return {
        "format": "png_base64",
        "box": [x0, y0, width, height],
        "width": width,
        "height": height,
        "data": base64.b64encode(buffer.getvalue()).decode("ascii"),
    }


def _ttp_decode_object_mask_data(mask_data):
    if not isinstance(mask_data, dict) or mask_data.get("format") != "png_base64":
        return None
    try:
        raw = base64.b64decode(str(mask_data.get("data", "")))
        return Image.open(BytesIO(raw)).convert("L")
    except Exception:
        return None


def _ttp_tile_object_mask_array(tile, out_width, out_height, mask_blend_mode="mask_feather", feather=0):
    if str(mask_blend_mode) == "off":
        return None
    mask_data = tile.get("object_mask")
    mask_pil = _ttp_decode_object_mask_data(mask_data)
    if mask_pil is None:
        return None

    sx, sy, sw, sh = tile["sample_box"]
    mx, my, mw, mh = [int(round(value)) for value in mask_data.get("box", [sx, sy, sw, sh])]
    source_mask = Image.new("L", (max(1, int(sw)), max(1, int(sh))), 0)
    if mask_pil.size != (max(1, int(mw)), max(1, int(mh))):
        mask_pil = mask_pil.resize((max(1, int(mw)), max(1, int(mh))), Image.Resampling.BILINEAR)

    paste_x = mx - int(sx)
    paste_y = my - int(sy)
    clip_left = max(0, -paste_x)
    clip_top = max(0, -paste_y)
    clip_right = min(mask_pil.width, source_mask.width - paste_x)
    clip_bottom = min(mask_pil.height, source_mask.height - paste_y)
    if clip_right <= clip_left or clip_bottom <= clip_top:
        return None
    source_mask.paste(
        mask_pil.crop((clip_left, clip_top, clip_right, clip_bottom)),
        (paste_x + clip_left, paste_y + clip_top),
    )
    source_mask = source_mask.resize((out_width, out_height), Image.Resampling.BILINEAR)
    if str(mask_blend_mode) in ("auto", "mask_feather"):
        radius = max(0.0, min(64.0, float(feather) * 0.35))
        if radius > 0:
            source_mask = source_mask.filter(ImageFilter.GaussianBlur(radius=radius))
    array = np.array(source_mask).astype(np.float32) / 255.0
    return array[:, :, None]


def _ttp_decode_interactive_paint_mask(mask_payload, image_width, image_height):
    try:
        payload = json.loads(str(mask_payload or "").strip() or "{}")
    except Exception:
        return None
    if not isinstance(payload, dict) or not payload.get("data"):
        return None
    try:
        data = str(payload.get("data", ""))
        if "," in data and data.lower().startswith("data:"):
            data = data.split(",", 1)[1]
        raw = base64.b64decode(data)
        mask = Image.open(BytesIO(raw)).convert("L")
    except Exception:
        return None
    if mask.getbbox() is None:
        return None
    if mask.size != (image_width, image_height):
        mask = mask.resize((image_width, image_height), Image.Resampling.BILINEAR)
    return mask


def _ttp_paint_mask_to_items(mask, image_width, image_height, min_area=24):
    if mask is None:
        return [], []
    array = np.array(mask.convert("L"), dtype=np.uint8)
    binary = np.where(array > 8, 255, 0).astype(np.uint8)
    if int(binary.max()) <= 0:
        return [], []
    if hasattr(cv2, "morphologyEx") and hasattr(cv2, "connectedComponentsWithStats"):
        kernel = np.ones((3, 3), dtype=np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        count, labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, 8)
    else:
        height, width = binary.shape
        labels = np.zeros((height, width), dtype=np.int32)
        stats_rows = [[0, 0, 0, 0, 0]]
        component = 0
        for start_y in range(height):
            for start_x in range(width):
                if binary[start_y, start_x] == 0 or labels[start_y, start_x] != 0:
                    continue
                component += 1
                stack = [(start_x, start_y)]
                labels[start_y, start_x] = component
                xs = []
                ys = []
                while stack:
                    x, y = stack.pop()
                    xs.append(x)
                    ys.append(y)
                    for next_y in range(max(0, y - 1), min(height, y + 2)):
                        for next_x in range(max(0, x - 1), min(width, x + 2)):
                            if binary[next_y, next_x] and labels[next_y, next_x] == 0:
                                labels[next_y, next_x] = component
                                stack.append((next_x, next_y))
                x0, x1 = min(xs), max(xs) + 1
                y0, y1 = min(ys), max(ys) + 1
                stats_rows.append([x0, y0, x1 - x0, y1 - y0, len(xs)])
        count = len(stats_rows)
        stats = np.asarray(stats_rows, dtype=np.int32)
    items = []
    masks = []
    min_area = max(1, int(min_area))
    for component in range(1, count):
        x, y, width, height, area = [int(value) for value in stats[component]]
        if area < min_area or width <= 0 or height <= 0:
            continue
        component_mask = np.where(labels == component, array, 0).astype(np.uint8)
        masks.append(Image.fromarray(component_mask, mode="L"))
        items.append({
            "x": x,
            "y": y,
            "width": width,
            "height": height,
            "label": f"paint mask {len(items) + 1}",
            "score": 1.0,
        })
    return items, masks


def _ttp_alignment_weight_mask(mask, mode="mask_edge_match"):
    mask_2d = np.clip(np.asarray(mask[:, :, 0], dtype=np.float32), 0.0, 1.0)
    if str(mode) == "edge_match":
        height, width = mask_2d.shape
        yy, xx = np.mgrid[0:height, 0:width]
        distance = np.minimum(np.minimum(xx, width - 1 - xx), np.minimum(yy, height - 1 - yy))
        edge = np.clip(1.0 - distance.astype(np.float32) / max(1.0, min(width, height) * 0.08), 0.0, 1.0)
        return (edge * mask_2d).astype(np.float32)
    band = 1.0 - np.abs(mask_2d * 2.0 - 1.0)
    band = np.where((mask_2d > 0.02) & (mask_2d < 0.98), band, 0.0)
    if float(band.sum()) < 16.0:
        band = np.where(mask_2d > 0.02, np.minimum(mask_2d, 1.0 - mask_2d * 0.5), 0.0)
    return band.astype(np.float32)


def _ttp_find_pixel_alignment_offset(region, reference, weights, out_x, out_y, mask, radius=0, mode="off"):
    radius = max(0, int(radius))
    if radius <= 0 or str(mode) == "off":
        return 0, 0
    if region.size == 0 or reference.size == 0:
        return 0, 0

    align_weights = _ttp_alignment_weight_mask(mask, mode)
    if float(align_weights.sum()) < 16.0:
        return 0, 0

    height, width = region.shape[:2]
    best_score = None
    best_offset = (0, 0)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            rx0 = max(0, out_x + dx)
            ry0 = max(0, out_y + dy)
            rx1 = min(reference.shape[1], out_x + dx + width)
            ry1 = min(reference.shape[0], out_y + dy + height)
            if rx1 <= rx0 or ry1 <= ry0:
                continue

            tx0 = rx0 - (out_x + dx)
            ty0 = ry0 - (out_y + dy)
            tx1 = tx0 + (rx1 - rx0)
            ty1 = ty0 + (ry1 - ry0)
            local_weight = align_weights[ty0:ty1, tx0:tx1]
            if weights is not None:
                local_weight = local_weight * (weights[ry0:ry1, rx0:rx1, 0] > 1e-6).astype(np.float32)
            weight_sum = float(local_weight.sum())
            if weight_sum < 16.0:
                continue

            diff = region[ty0:ty1, tx0:tx1] - reference[ry0:ry1, rx0:rx1]
            score = float((np.mean(diff * diff, axis=2) * local_weight).sum() / weight_sum)
            score += (abs(dx) + abs(dy)) * 1e-5
            if best_score is None or score < best_score:
                best_score = score
                best_offset = (dx, dy)
    return best_offset


def _ttp_alignment_torch_device(device_mode="auto"):
    device_mode = str(device_mode or "auto").lower()
    if device_mode == "cpu":
        return None
    if not all(hasattr(torch, name) for name in ("as_tensor", "zeros", "full")):
        return None
    try:
        device = comfy.model_management.get_torch_device()
    except Exception:
        try:
            if hasattr(torch, "cuda") and torch.cuda.is_available():
                device = torch.device("cuda")
            elif hasattr(torch, "backends") and getattr(getattr(torch.backends, "mps", None), "is_available", lambda: False)():
                device = torch.device("mps")
            else:
                return None
        except Exception:
            return None
    if getattr(device, "type", "cpu") == "cpu":
        return None
    return device


def _ttp_find_pixel_alignment_offset_gpu(
    region,
    reference,
    weights,
    out_x,
    out_y,
    mask,
    radius=0,
    mode="off",
    device_mode="auto",
    max_samples=32768,
):
    radius = max(0, int(radius))
    if radius <= 0 or str(mode) == "off":
        return (0, 0), {"device": "off", "samples": 0, "fallback": False}
    device = _ttp_alignment_torch_device(device_mode)
    if device is None:
        return None
    if region.size == 0 or reference.size == 0:
        return (0, 0), {"device": str(device), "samples": 0, "fallback": False}

    align_weights = _ttp_alignment_weight_mask(mask, mode)
    if float(align_weights.sum()) < 16.0:
        return (0, 0), {"device": str(device), "samples": 0, "fallback": False}

    height, width = region.shape[:2]
    ys, xs = np.nonzero(align_weights > 1e-6)
    if len(xs) < 16:
        return (0, 0), {"device": str(device), "samples": int(len(xs)), "fallback": False}
    max_samples = max(256, int(max_samples or 32768))
    original_samples = int(len(xs))
    if len(xs) > max_samples:
        indices = np.linspace(0, len(xs) - 1, max_samples).astype(np.int64)
        xs = xs[indices]
        ys = ys[indices]

    offsets = [(dx, dy) for dy in range(-radius, radius + 1) for dx in range(-radius, radius + 1)]
    if not offsets:
        return (0, 0), {"device": str(device), "samples": int(len(xs)), "fallback": False}

    try:
        with torch.inference_mode():
            xs_t = torch.as_tensor(xs.astype(np.int64), device=device)
            ys_t = torch.as_tensor(ys.astype(np.int64), device=device)
            sample_region = torch.as_tensor(region[ys, xs].astype(np.float32), device=device)
            sample_weight = torch.as_tensor(align_weights[ys, xs].astype(np.float32), device=device)
            reference_t = torch.as_tensor(reference.astype(np.float32), device=device)
            ref_flat = reference_t.reshape(-1, reference_t.shape[-1])
            if weights is not None:
                weight_t = torch.as_tensor(weights[:, :, 0].astype(np.float32), device=device)
                weight_flat = weight_t.reshape(-1)
            else:
                weight_flat = None

            offset_t = torch.as_tensor(offsets, dtype=torch.int64, device=device)
            ref_h, ref_w = int(reference.shape[0]), int(reference.shape[1])
            best_score = None
            best_offset = (0, 0)
            chunk_size = max(8, min(512, int(4000000 // max(1, len(xs)))))
            for start in range(0, int(offset_t.shape[0]), chunk_size):
                chunk = offset_t[start:start + chunk_size]
                dx = chunk[:, 0:1]
                dy = chunk[:, 1:2]
                rx = int(out_x) + dx + xs_t[None, :]
                ry = int(out_y) + dy + ys_t[None, :]
                valid = (rx >= 0) & (rx < ref_w) & (ry >= 0) & (ry < ref_h)
                flat_index = torch.clamp(ry, 0, ref_h - 1) * ref_w + torch.clamp(rx, 0, ref_w - 1)
                local_weight = sample_weight[None, :] * valid.to(torch.float32)
                if weight_flat is not None:
                    local_weight = local_weight * (weight_flat[flat_index] > 1e-6).to(torch.float32)
                weight_sum = local_weight.sum(dim=1)
                usable = weight_sum >= 16.0
                if not bool(torch.any(usable).item()):
                    continue
                ref_samples = ref_flat[flat_index]
                diff = sample_region[None, :, :] - ref_samples
                mse = torch.mean(diff * diff, dim=2)
                scores = (mse * local_weight).sum(dim=1) / torch.clamp(weight_sum, min=1e-6)
                penalty = (torch.abs(chunk[:, 0]).to(torch.float32) + torch.abs(chunk[:, 1]).to(torch.float32)) * 1e-5
                scores = scores + penalty
                scores = torch.where(usable, scores, torch.full_like(scores, float("inf")))
                score, index = torch.min(scores, dim=0)
                score_value = float(score.item())
                if np.isfinite(score_value) and (best_score is None or score_value < best_score):
                    best_score = score_value
                    best = chunk[int(index.item())]
                    best_offset = (int(best[0].item()), int(best[1].item()))
            return best_offset, {
                "device": str(device),
                "samples": int(len(xs)),
                "source_samples": int(original_samples),
                "candidates": int(len(offsets)),
                "fallback": False,
            }
    except Exception as error:
        if str(device_mode or "auto").lower() == "gpu":
            print(f"[TTP Smart Tile] pixel alignment GPU failed, falling back to CPU: {type(error).__name__}: {error}")
        return None


def _ttp_find_pixel_alignment_offset_torch_canvas(
    region,
    reference_t,
    weights_t,
    out_x,
    out_y,
    mask,
    radius=0,
    mode="off",
    max_samples=32768,
):
    radius = max(0, int(radius))
    if radius <= 0 or str(mode) == "off":
        return (0, 0), {"device": "off", "samples": 0, "fallback": False}
    if region.size == 0 or reference_t is None or int(reference_t.numel()) <= 0:
        device_name = str(getattr(reference_t, "device", "gpu"))
        return (0, 0), {"device": device_name, "samples": 0, "fallback": False}

    align_weights = _ttp_alignment_weight_mask(mask, mode)
    if float(align_weights.sum()) < 16.0:
        return (0, 0), {"device": str(reference_t.device), "samples": 0, "fallback": False}

    height, width = region.shape[:2]
    ys, xs = np.nonzero(align_weights > 1e-6)
    if len(xs) < 16:
        return (0, 0), {"device": str(reference_t.device), "samples": int(len(xs)), "fallback": False}
    max_samples = max(256, int(max_samples or 32768))
    original_samples = int(len(xs))
    if len(xs) > max_samples:
        indices = np.linspace(0, len(xs) - 1, max_samples).astype(np.int64)
        xs = xs[indices]
        ys = ys[indices]

    offsets = [(dx, dy) for dy in range(-radius, radius + 1) for dx in range(-radius, radius + 1)]
    if not offsets:
        return (0, 0), {"device": str(reference_t.device), "samples": int(len(xs)), "fallback": False}

    try:
        with torch.inference_mode():
            device = reference_t.device
            reference_t = reference_t.to(device=device, dtype=torch.float32)
            xs_t = torch.as_tensor(xs.astype(np.int64), device=device)
            ys_t = torch.as_tensor(ys.astype(np.int64), device=device)
            sample_region = torch.as_tensor(region[ys, xs].astype(np.float32), device=device)
            sample_weight = torch.as_tensor(align_weights[ys, xs].astype(np.float32), device=device)
            ref_flat = reference_t.reshape(-1, reference_t.shape[-1])
            if weights_t is not None:
                weight_flat = weights_t[:, :, 0].to(device=device, dtype=torch.float32).reshape(-1)
            else:
                weight_flat = None

            offset_t = torch.as_tensor(offsets, dtype=torch.int64, device=device)
            ref_h, ref_w = int(reference_t.shape[0]), int(reference_t.shape[1])
            best_score = None
            best_offset = (0, 0)
            chunk_size = max(8, min(512, int(4000000 // max(1, len(xs)))))
            for start in range(0, int(offset_t.shape[0]), chunk_size):
                chunk = offset_t[start:start + chunk_size]
                dx = chunk[:, 0:1]
                dy = chunk[:, 1:2]
                rx = int(out_x) + dx + xs_t[None, :]
                ry = int(out_y) + dy + ys_t[None, :]
                valid = (rx >= 0) & (rx < ref_w) & (ry >= 0) & (ry < ref_h)
                flat_index = torch.clamp(ry, 0, ref_h - 1) * ref_w + torch.clamp(rx, 0, ref_w - 1)
                local_weight = sample_weight[None, :] * valid.to(torch.float32)
                if weight_flat is not None:
                    local_weight = local_weight * (weight_flat[flat_index] > 1e-6).to(torch.float32)
                weight_sum = local_weight.sum(dim=1)
                usable = weight_sum >= 16.0
                if not bool(torch.any(usable).item()):
                    continue
                ref_samples = ref_flat[flat_index]
                diff = sample_region[None, :, :] - ref_samples
                mse = torch.mean(diff * diff, dim=2)
                scores = (mse * local_weight).sum(dim=1) / torch.clamp(weight_sum, min=1e-6)
                penalty = (torch.abs(chunk[:, 0]).to(torch.float32) + torch.abs(chunk[:, 1]).to(torch.float32)) * 1e-5
                scores = scores + penalty
                scores = torch.where(usable, scores, torch.full_like(scores, float("inf")))
                score, index = torch.min(scores, dim=0)
                score_value = float(score.item())
                if np.isfinite(score_value) and (best_score is None or score_value < best_score):
                    best_score = score_value
                    best = chunk[int(index.item())]
                    best_offset = (int(best[0].item()), int(best[1].item()))
            return best_offset, {
                "device": str(device),
                "samples": int(len(xs)),
                "source_samples": int(original_samples),
                "candidates": int(len(offsets)),
                "fallback": False,
            }
    except Exception as error:
        print(f"[TTP Smart Tile] pixel alignment torch canvas failed: {type(error).__name__}: {error}")
        return None


def _ttp_find_pixel_alignment_offset_auto(
    region,
    reference,
    weights,
    out_x,
    out_y,
    mask,
    radius=0,
    mode="off",
    device_mode="auto",
):
    if str(device_mode or "auto").lower() != "cpu":
        gpu_result = _ttp_find_pixel_alignment_offset_gpu(
            region,
            reference,
            weights,
            out_x,
            out_y,
            mask,
            radius,
            mode,
            device_mode,
        )
        if gpu_result is not None:
            offset, info = gpu_result
            return offset[0], offset[1], info
    offset = _ttp_find_pixel_alignment_offset(region, reference, weights, out_x, out_y, mask, radius, mode)
    return offset[0], offset[1], {
        "device": "cpu",
        "samples": int(np.count_nonzero(_ttp_alignment_weight_mask(mask, mode) > 1e-6)) if str(mode) != "off" else 0,
        "candidates": int((max(0, int(radius)) * 2 + 1) ** 2),
        "fallback": str(device_mode or "auto").lower() != "cpu",
    }


def _ttp_bbox_to_xyxy(box):
    if isinstance(box, dict):
        if all(key in box for key in ("x", "y", "width", "height")):
            return [
                float(box.get("x", 0)),
                float(box.get("y", 0)),
                float(box.get("x", 0)) + float(box.get("width", 0)),
                float(box.get("y", 0)) + float(box.get("height", 0)),
            ]
        raw = box.get("bbox_2d", box.get("bbox", box.get("box")))
        if raw is not None:
            return _ttp_bbox_to_xyxy(raw)
    if isinstance(box, (list, tuple)) and len(box) >= 4:
        return [float(box[0]), float(box[1]), float(box[2]), float(box[3])]
    return None


def _ttp_flatten_bboxes(bboxes):
    if bboxes is None:
        return []
    if isinstance(bboxes, dict):
        return [bboxes]
    if not isinstance(bboxes, (list, tuple)):
        return []
    result = []
    for item in bboxes:
        if isinstance(item, (list, tuple)) and item and not isinstance(item[0], (int, float, dict)):
            result.extend(_ttp_flatten_bboxes(item))
        elif isinstance(item, (list, tuple)) and item and isinstance(item[0], dict):
            result.extend(item)
        else:
            result.append(item)
    return result


def _ttp_clamp_box_xyxy(box, image_width, image_height, min_size=8):
    x0, y0, x1, y1 = box
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    x0 = _ttp_clamp(int(round(x0)), 0, max(0, image_width - 1))
    y0 = _ttp_clamp(int(round(y0)), 0, max(0, image_height - 1))
    x1 = _ttp_clamp(int(round(x1)), x0 + 1, image_width)
    y1 = _ttp_clamp(int(round(y1)), y0 + 1, image_height)
    if x1 - x0 < min_size:
        grow = int(np.ceil((min_size - (x1 - x0)) / 2))
        x0 = max(0, x0 - grow)
        x1 = min(image_width, x1 + grow)
    if y1 - y0 < min_size:
        grow = int(np.ceil((min_size - (y1 - y0)) / 2))
        y0 = max(0, y0 - grow)
        y1 = min(image_height, y1 + grow)
    return [x0, y0, x1, y1]


def _ttp_expand_box_xyxy(box, image_width, image_height, padding):
    x0, y0, x1, y1 = box
    return _ttp_clamp_box_xyxy(
        [x0 - padding, y0 - padding, x1 + padding, y1 + padding],
        image_width,
        image_height,
    )


def _ttp_box_iou(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    if inter <= 0:
        return 0.0
    area_a = max(1, (ax1 - ax0) * (ay1 - ay0))
    area_b = max(1, (bx1 - bx0) * (by1 - by0))
    return inter / max(1, area_a + area_b - inter)


def _ttp_merge_overlapping_boxes(boxes, threshold=0.18):
    merged = []
    for item in boxes:
        box = item["box"]
        target = None
        for existing in merged:
            if _ttp_box_iou(box, existing["box"]) >= threshold:
                target = existing
                break
        if target is None:
            merged.append({**item, "box": list(box)})
            continue
        ex = target["box"]
        target["box"] = [min(ex[0], box[0]), min(ex[1], box[1]), max(ex[2], box[2]), max(ex[3], box[3])]
        target["score"] = max(float(target.get("score", 1.0)), float(item.get("score", 1.0)))
        target["mask_indices"] = sorted(set(target.get("mask_indices", []) + item.get("mask_indices", [])))
        if not target.get("label") and item.get("label"):
            target["label"] = item["label"]
    return merged


def _ttp_auto_tile_rank(label, object_area, image_area, object_index):
    label_text = str(label or "").lower()
    detail_keywords = (
        "face", "head", "eye", "eyes", "eyelash", "eyebrow", "glasses",
        "hand", "hands", "finger", "fingers", "mouth", "lip", "lips",
        "teeth", "nose", "ear", "ears", "text", "letter", "logo",
        "脸", "头", "眼", "眼睛", "眉", "睫毛", "眼镜", "手", "手指",
        "嘴", "嘴唇", "牙", "鼻", "耳", "文字", "字体", "标志",
    )
    area_ratio = max(0.0, min(1.0, float(object_area) / max(1.0, float(image_area))))
    small_bonus = min(450, int(round((1.0 - area_ratio) * 450.0)))
    if any(keyword in label_text for keyword in detail_keywords):
        return 4, 2000 + small_bonus + int(object_index)
    return 2, 300 + small_bonus + int(object_index)


def _ttp_boxes_to_auto_layout(
    bboxes,
    image_width,
    image_height,
    default_pad=128,
    default_blend=64,
    object_padding=96,
    max_tiles=16,
    include_background=True,
    allow_object_overlap=True,
    masks=None,
):
    mask_images = _ttp_mask_tensor_to_pil_list(masks, image_width, image_height)
    raw_boxes = []
    for index, item in enumerate(_ttp_flatten_bboxes(bboxes)):
        box = _ttp_bbox_to_xyxy(item)
        if box is None:
            continue
        score = float(item.get("score", 1.0)) if isinstance(item, dict) else 1.0
        label = str(item.get("label", item.get("name", f"object_{index + 1}"))) if isinstance(item, dict) else f"object_{index + 1}"
        clamped = _ttp_clamp_box_xyxy(box, image_width, image_height)
        area = (clamped[2] - clamped[0]) * (clamped[3] - clamped[1])
        if area <= 0:
            continue
        raw_boxes.append({"box": clamped, "score": score, "label": label, "area": area, "mask_indices": [index]})

    raw_boxes.sort(key=lambda value: (float(value.get("score", 1.0)), value["area"]), reverse=True)
    merged_boxes = _ttp_merge_overlapping_boxes(raw_boxes[: max(1, int(max_tiles))])

    tiles = []
    if include_background:
        tiles.append({
            "name": "auto_background",
            "x0": 0.0,
            "y0": 0.0,
            "x1": 1.0,
            "y1": 1.0,
            "pad": 0,
            "blend": max(0, int(default_blend)),
            "priority": 5,
            "importance": 0.35,
            "source": "auto",
            "label": "background",
            "layer": 0,
            "object_id": 0,
            "occlusion_priority": 0,
        })

    for object_index, item in enumerate(merged_boxes[: max(1, int(max_tiles))], start=1):
        box = _ttp_expand_box_xyxy(item["box"], image_width, image_height, int(max(0, object_padding)))
        x0, y0, x1, y1 = box
        object_area = (x1 - x0) * (y1 - y0)
        image_area = max(1, image_width * image_height)
        layer, occlusion_priority = _ttp_auto_tile_rank(item.get("label", f"object_{object_index}"), object_area, image_area, object_index)
        priority = 70 + min(80, int(100 * object_area / image_area)) + int(float(item.get("score", 1.0)) * 20)
        tile = {
            "name": f"auto_{item.get('label', 'object')}_{object_index}",
            "x0": x0 / image_width,
            "y0": y0 / image_height,
            "x1": x1 / image_width,
            "y1": y1 / image_height,
            "pad": int(default_pad) if allow_object_overlap else max(0, int(default_pad) // 2),
            "blend": int(default_blend),
            "priority": priority,
            "importance": 1.0,
            "source": "auto",
            "label": item.get("label", f"object_{object_index}"),
            "score": float(item.get("score", 1.0)),
            "layer": layer,
            "object_id": object_index,
            "occlusion_priority": occlusion_priority,
        }
        object_mask = _ttp_encode_object_mask_data(mask_images, item.get("mask_indices", []), box)
        if object_mask is not None:
            tile["object_mask"] = object_mask
        tiles.append(tile)

    if len(tiles) == (1 if include_background else 0):
        tiles.extend([
            {"name": "auto_tile_1", "x0": 0.0, "y0": 0.0, "x1": 0.5, "y1": 0.5, "source": "auto", "layer": 1, "occlusion_priority": 10},
            {"name": "auto_tile_2", "x0": 0.5, "y0": 0.0, "x1": 1.0, "y1": 0.5, "source": "auto", "layer": 1, "occlusion_priority": 10},
            {"name": "auto_tile_3", "x0": 0.0, "y0": 0.5, "x1": 0.5, "y1": 1.0, "source": "auto", "layer": 1, "occlusion_priority": 10},
            {"name": "auto_tile_4", "x0": 0.5, "y0": 0.5, "x1": 1.0, "y1": 1.0, "source": "auto", "layer": 1, "occlusion_priority": 10},
        ])

    return json.dumps({
        "version": 1,
        "type": "ttp_smart_tile_interactive_layout",
        "source_size": [image_width, image_height],
        "defaults": {
            "pad": int(default_pad),
            "blend": int(default_blend),
            "priority": 50,
            "importance": 1.0,
            "source": "auto",
        },
        "tiles": tiles[: max(1, int(max_tiles)) + (1 if include_background else 0)],
    }, separators=(",", ":"))


def _ttp_draw_auto_layout_preview(pil_image, layout_json):
    preview = pil_image.copy().convert("RGB")
    draw = ImageDraw.Draw(preview)
    width, height = preview.size
    colors = ["#38bdf8", "#f97316", "#22c55e", "#e879f9", "#facc15", "#fb7185", "#a3e635", "#60a5fa"]
    try:
        layout = json.loads(layout_json)
    except Exception:
        return pil2tensor(preview)
    for index, tile in enumerate(layout.get("tiles", [])):
        box = _ttp_normalize_tile_box(tile, width, height, {})
        x, y, w, h = box["core_box"]
        color = colors[index % len(colors)]
        draw.rectangle((x, y, x + w, y + h), outline=color, width=3)
        draw.text((x + 5, y + 5), str(tile.get("label", tile.get("name", index + 1))), fill=color)
    return pil2tensor(preview)


def _ttp_tensor_from_pil_batch(pil_image):
    return pil2tensor(pil_image.convert("RGB"))


def _ttp_image_tensor_size(image_tensor):
    shape = image_tensor.shape
    if len(shape) == 4:
        return int(shape[2]), int(shape[1])
    if len(shape) == 3:
        return int(shape[1]), int(shape[0])
    raise ValueError("Expected an IMAGE tensor with shape [H,W,C] or [1,H,W,C].")


def _ttp_image_tensor_to_pil(image_tensor):
    if len(image_tensor.shape) == 3:
        return tensor2pil(image_tensor.unsqueeze(0)).convert("RGB")
    return tensor2pil(image_tensor).convert("RGB")


def _ttp_first_image_tensor(image_tensor):
    if len(image_tensor.shape) == 4:
        return image_tensor[0].unsqueeze(0)
    if len(image_tensor.shape) == 3:
        return image_tensor.unsqueeze(0)
    raise ValueError("Expected an IMAGE tensor with shape [B,H,W,C] or [H,W,C].")


def _ttp_resize_pil_to_round(image, scale=2.0, round_to=8, resampling="lanczos"):
    filters = {
        "nearest": Image.Resampling.NEAREST,
        "nearest-exact": Image.Resampling.NEAREST,
        "bilinear": Image.Resampling.BILINEAR,
        "bicubic": Image.Resampling.BICUBIC,
        "area": Image.Resampling.BOX,
        "lanczos": Image.Resampling.LANCZOS,
    }
    target_width = max(1, int(round(image.width * float(scale))))
    target_height = max(1, int(round(image.height * float(scale))))
    round_to = max(1, int(round_to))
    target_width = max(round_to, _ttp_round_to(target_width, round_to))
    target_height = max(round_to, _ttp_round_to(target_height, round_to))
    return image.resize((target_width, target_height), filters.get(str(resampling), Image.Resampling.LANCZOS))


def _ttp_resize_pil_to_size(image, width, height, resampling="lanczos"):
    filters = {
        "nearest": Image.Resampling.NEAREST,
        "nearest-exact": Image.Resampling.NEAREST,
        "bilinear": Image.Resampling.BILINEAR,
        "bicubic": Image.Resampling.BICUBIC,
        "area": Image.Resampling.BOX,
        "lanczos": Image.Resampling.LANCZOS,
    }
    return image.resize((max(1, int(width)), max(1, int(height))), filters.get(str(resampling), Image.Resampling.LANCZOS))


def _ttp_round_down_to(value, multiple):
    multiple = max(1, int(multiple))
    value = max(1, int(math.floor(float(value))))
    if multiple <= 1:
        return value
    rounded = int(math.floor(value / multiple) * multiple)
    return max(multiple, rounded)


def _ttp_smart_upscale_target_size(width, height, scale=2.0, round_to=8, max_megapixels=0.0):
    width = max(1, int(width))
    height = max(1, int(height))
    scale = max(0.01, float(scale))
    round_to = max(1, int(round_to))
    max_pixels = int(float(max_megapixels) * 1000000.0) if float(max_megapixels or 0.0) > 0 else 0
    desired_width = max(1, int(round(width * scale)))
    desired_height = max(1, int(round(height * scale)))
    capped = False
    if max_pixels > 0 and desired_width * desired_height > max_pixels:
        capped = True
        scale = math.sqrt(max_pixels / max(1.0, float(width) * float(height)))
        target_width = _ttp_round_down_to(width * scale, round_to)
        target_height = _ttp_round_down_to(height * scale, round_to)
        while target_width * target_height > max_pixels and (target_width > 1 or target_height > 1):
            if target_width / max(1, width) >= target_height / max(1, height) and target_width > 1:
                target_width = max(1, target_width - round_to)
            elif target_height > 1:
                target_height = max(1, target_height - round_to)
            else:
                target_width = max(1, target_width - round_to)
        return max(1, target_width), max(1, target_height), capped
    return max(round_to, _ttp_round_to(desired_width, round_to)), max(round_to, _ttp_round_to(desired_height, round_to)), capped


def _ttp_resize_image_tensor(image, width, height, method="lanczos"):
    if not hasattr(image, "movedim") or not hasattr(comfy.utils, "common_upscale"):
        pil = _ttp_image_tensor_to_pil(_ttp_first_image_tensor(image))
        return pil2tensor(_ttp_resize_pil_to_size(pil, int(width), int(height), method))
    method = str(method or "lanczos")
    method_map = {
        "nearest": "nearest-exact",
        "nearest-exact": "nearest-exact",
        "bilinear": "bilinear",
        "bicubic": "bicubic",
        "area": "area",
        "lanczos": "lanczos",
    }
    image = _ttp_first_image_tensor(image)
    samples = image.movedim(-1, 1)
    resized = comfy.utils.common_upscale(samples, int(width), int(height), method_map.get(method, "lanczos"), "disabled")
    return torch.clamp(resized.movedim(1, -1), min=0.0, max=1.0).to(dtype=image.dtype)


def _ttp_upscale_image_with_model(upscale_model, image):
    if upscale_model is None:
        return image
    device = comfy.model_management.get_torch_device()
    upscale_amount = max(float(getattr(upscale_model, "scale", 1.0) or 1.0), 1.0)
    memory_required = comfy.model_management.module_size(upscale_model.model)
    memory_required += (512 * 512 * 3) * image.element_size() * upscale_amount * 384.0
    memory_required += image.nelement() * image.element_size()
    comfy.model_management.free_memory(memory_required, device)

    upscale_model.to(device)
    in_img = image.movedim(-1, -3).to(device)
    tile = 512
    overlap = 32
    output_device = comfy.model_management.intermediate_device()
    try:
        while True:
            try:
                steps = in_img.shape[0] * comfy.utils.get_tiled_scale_steps(
                    in_img.shape[3],
                    in_img.shape[2],
                    tile_x=tile,
                    tile_y=tile,
                    overlap=overlap,
                )
                pbar = comfy.utils.ProgressBar(steps)
                scaled = comfy.utils.tiled_scale(
                    in_img,
                    lambda a: upscale_model(a.float()),
                    tile_x=tile,
                    tile_y=tile,
                    overlap=overlap,
                    upscale_amount=upscale_amount,
                    pbar=pbar,
                    output_device=output_device,
                )
                break
            except Exception as exc:
                comfy.model_management.raise_non_oom(exc)
                tile //= 2
                if tile < 128:
                    raise exc
    finally:
        upscale_model.to("cpu")

    scaled = torch.clamp(scaled.movedim(-3, -1), min=0.0, max=1.0)
    return scaled.to(device=comfy.model_management.intermediate_device(), dtype=comfy.model_management.intermediate_dtype())


def _ttp_align_pil_to_aspect(image, expected_width, expected_height, mode="center_crop"):
    expected_width = max(1, int(expected_width))
    expected_height = max(1, int(expected_height))
    if mode == "off" or image.width <= 0 or image.height <= 0:
        return image
    if mode == "resize":
        return image.resize((expected_width, expected_height), Image.Resampling.LANCZOS)

    target_aspect = expected_width / max(1, expected_height)
    source_aspect = image.width / max(1, image.height)
    if mode in ("center_crop", "crop_or_pad") and abs(source_aspect - target_aspect) > 1e-4:
        if source_aspect > target_aspect:
            crop_width = max(1, int(round(image.height * target_aspect)))
            left = max(0, (image.width - crop_width) // 2)
            image = image.crop((left, 0, left + crop_width, image.height))
        else:
            crop_height = max(1, int(round(image.width / target_aspect)))
            top = max(0, (image.height - crop_height) // 2)
            image = image.crop((0, top, image.width, top + crop_height))

    if mode == "crop_or_pad":
        pad_width = max(image.width, int(round(image.height * target_aspect)))
        pad_height = max(image.height, int(round(image.width / target_aspect)))
        if pad_width != image.width or pad_height != image.height:
            canvas = Image.new("RGB", (pad_width, pad_height))
            canvas.paste(image, ((pad_width - image.width) // 2, (pad_height - image.height) // 2))
            image = canvas

    return image.resize((expected_width, expected_height), Image.Resampling.LANCZOS)


def _ttp_clone_tile_set(tile_set):
    tile_meta = tile_set.get("tile_meta", {})
    return {
        "version": int(tile_set.get("version", 1)),
        "type": "ttp_smart_tile_set",
        "original_size": list(tile_set.get("original_size", tile_meta.get("original_size", [0, 0]))),
        "tile_meta": {
            "version": int(tile_meta.get("version", 3)),
            "type": "ttp_smart_tile",
            "storage": "tile_set",
            "original_size": list(tile_meta.get("original_size", tile_set.get("original_size", [0, 0]))),
            "tiles": [dict(tile) for tile in tile_meta.get("tiles", [])],
        },
        "tile_images": list(tile_set.get("tile_images", [])),
        "positions": list(tile_set.get("positions", [])),
    }


def _ttp_tile_set_fingerprint(tile_set):
    tile_meta = tile_set.get("tile_meta", {})
    tiles = tile_meta.get("tiles", [])
    return json.dumps({
        "size": tile_set.get("original_size", tile_meta.get("original_size")),
        "count": len(tile_set.get("tile_images", [])),
        "boxes": [tile.get("sample_box") for tile in tiles],
        "names": [tile.get("name") for tile in tiles],
        "masks": [bool(tile.get("object_mask")) for tile in tiles],
        "prompts": [
            {
                "prompt": tile.get("prompt", ""),
                "negative": tile.get("negative", ""),
                "caption": tile.get("caption", ""),
                "label": tile.get("label", ""),
                "prompt_tag": tile.get("prompt_tag", ""),
                "prompt_source": tile.get("prompt_source", ""),
                "tile_hash": tile.get("tile_hash", ""),
            }
            for tile in tiles
        ],
    }, sort_keys=True)


def _ttp_validate_tile_set(tile_set):
    if not isinstance(tile_set, dict) or tile_set.get("type") != "ttp_smart_tile_set":
        raise ValueError("tile_set must come from TTP Smart Tile Interactive Crop")
    tile_images = tile_set.get("tile_images", [])
    tile_meta = tile_set.get("tile_meta", {})
    tiles_info = tile_meta.get("tiles", [])
    if not tile_images:
        raise ValueError("tile_set contains no tile images.")
    if len(tile_images) != len(tiles_info):
        raise ValueError(f"tile_set images ({len(tile_images)}) do not match tile metadata ({len(tiles_info)}).")
    return tile_images, tile_meta, tiles_info


def _ttp_assemble_placeholder_output(tile_images=None, base_image=None, source_image=None):
    preview = None
    for candidate in (base_image, source_image):
        if candidate is not None:
            preview = _ttp_first_image_tensor(candidate)
            break
    if preview is None and tile_images:
        preview = _ttp_first_image_tensor(tile_images[0])
    if preview is None:
        preview = pil2tensor(Image.new("RGB", (1, 1), "black"))
    width, height = _ttp_image_tensor_size(preview)
    return preview, pil2tensor(Image.new("RGB", (width, height), "black"))


def _ttp_send_smart_tile_loop_event(task, done=False, message=""):
    if PromptServer is None or not isinstance(task, dict):
        return
    payload = {
        "source_node_id": str(task.get("source_node_id", "")),
        "session_id": str(task.get("session_id", "")),
        "index": int(task.get("index", 0)),
        "count": int(task.get("count", 0)),
        "done": bool(done),
        "message": str(message or ""),
    }
    try:
        PromptServer.instance.send_sync("ttp-smart-tile-loop", payload, PromptServer.instance.client_id)
    except Exception:
        pass


def _ttp_pil_to_data_url(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _ttp_model_base_dirs(model_type="text_encoders"):
    dirs = []
    try:
        names_and_paths = getattr(folder_paths, "folder_names_and_paths", {})
        for path in names_and_paths.get(model_type, [[], set()])[0]:
            dirs.append(path)
    except Exception:
        pass
    try:
        models_dir = getattr(folder_paths, "models_dir", None)
        if models_dir:
            dirs.append(os.path.join(models_dir, model_type))
    except Exception:
        pass
    return [path for index, path in enumerate(dirs) if path and path not in dirs[:index] and os.path.isdir(path)]


def _ttp_qwenvl_safetensor_files():
    files = []
    for base_dir in _ttp_model_base_dirs("text_encoders"):
        for root, _dirs, names in os.walk(base_dir):
            for name in names:
                if not name.lower().endswith((".safetensors", ".sft")):
                    continue
                rel = os.path.relpath(os.path.join(root, name), base_dir).replace("\\", "/")
                files.append(rel)
    return sorted(set(files)) or ["put_qwenvl3_model_in_models_text_encoders.safetensors"]


def _ttp_resolve_model_file(model_file, model_type="text_encoders"):
    model_file = str(model_file or "").replace("\\", "/")
    for base_dir in _ttp_model_base_dirs(model_type):
        candidate = os.path.abspath(os.path.join(base_dir, model_file))
        try:
            if os.path.commonpath([os.path.abspath(base_dir), candidate]) != os.path.abspath(base_dir):
                continue
        except ValueError:
            continue
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(f"Model file not found in ComfyUI models/{model_type}: {model_file}")


def _ttp_load_qwenvl3_local_model(model_file, model_family="auto", device_mode="default"):
    model_path = _ttp_resolve_model_file(model_file, "text_encoders")
    model_family = str(model_family or "auto")
    # For visual tagging we should not expose downstream image-model choices.
    # QWEN_IMAGE is ComfyUI's native single-file Qwen-VL text encoder entry;
    # Qwen3.5-VL variants are still auto-detected from the safetensors keys.
    clip_type_name = "QWEN_IMAGE"
    cache_key = json.dumps({
        "path": model_path,
        "device_mode": device_mode,
        "model_family": model_family,
    }, sort_keys=True)
    if cache_key in _TTP_QWENVL3_MODEL_CACHE:
        return _TTP_QWENVL3_MODEL_CACHE[cache_key]

    model_options = {}
    if str(device_mode) == "cpu":
        model_options["load_device"] = model_options["offload_device"] = torch.device("cpu")
    clip = comfy.sd.load_clip(
        ckpt_paths=[model_path],
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
        clip_type=comfy.sd.CLIPType.QWEN_IMAGE,
        model_options=model_options,
    )

    loaded = {
        "type": "ttp_qwenvl3_model",
        "model_file": model_file,
        "model_path": model_path,
        "clip_type": clip_type_name,
        "model_family": model_family,
        "clip": clip,
    }
    _TTP_QWENVL3_MODEL_CACHE[cache_key] = loaded
    return loaded


def _ttp_qwen_vl_local_chat(local_model, messages, max_new_tokens=256, temperature=0.2, seed=0):
    if not isinstance(local_model, dict) or local_model.get("type") != "ttp_qwenvl3_model":
        raise ValueError("qwen_vl_local mode requires TTP QwenVL3 Local Loader output.")
    clip = local_model["clip"]

    system_parts = []
    user_parts = []
    images = []
    for message in messages:
        role = str(message.get("role", "user"))
        content = message.get("content", [])
        if isinstance(content, str):
            if role == "system":
                system_parts.append(content)
            else:
                user_parts.append(content)
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    if role == "system":
                        system_parts.append(str(item.get("text", "")))
                    else:
                        user_parts.append(str(item.get("text", "")))
                if isinstance(item, dict) and item.get("type") == "image":
                    images.append(item.get("image"))
                elif isinstance(item, dict) and item.get("type") == "image_url":
                    url = item.get("image_url", {}).get("url", "")
                    if str(url).startswith("data:image"):
                        raw = base64.b64decode(str(url).split(",", 1)[1])
                        image = Image.open(BytesIO(raw)).convert("RGB")
                        images.append(image)
    system_prompt = "\n".join(part for part in system_parts if part).strip()
    user_prompt = "\n".join(part for part in user_parts if part).strip()
    if system_prompt:
        user_prompt = f"System instruction:\n{system_prompt}\n\nUser request:\n{user_prompt}".strip()
    image_tensors = [pil2tensor(image.convert("RGB")) if isinstance(image, Image.Image) else image for image in images]
    tokenize_kwargs = {}
    if image_tensors:
        tokenize_kwargs["images"] = image_tensors
    prompt = user_prompt.strip()
    tokens = clip.tokenize(prompt, **tokenize_kwargs)
    do_sample = float(temperature) > 0
    output_ids = clip.generate(
        tokens,
        do_sample=do_sample,
        max_length=int(max_new_tokens),
        temperature=float(temperature),
        seed=int(seed if seed is not None else 0),
    )
    text = clip.decode(output_ids, skip_special_tokens=True)
    return _ttp_clean_qwen_vl_response(str(text), prompt)


def _ttp_encode_text_conditioning(clip, text):
    if clip is None:
        return []
    tokens = clip.tokenize(str(text or ""))
    return clip.encode_from_tokens_scheduled(tokens)


def _ttp_tile_hash(image_tensor):
    pil = _ttp_image_tensor_to_pil(image_tensor)
    arr = np.array(pil.resize((min(128, pil.width), min(128, pil.height)), Image.Resampling.BILINEAR))
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def _ttp_resize_pil_for_qwen(image, max_side=0, max_pixels=0):
    if not isinstance(image, Image.Image):
        return image
    max_side = int(max_side or 0)
    max_pixels = int(max_pixels or 0)
    width, height = image.size
    scale = 1.0
    if max_side > 0:
        scale = min(scale, max_side / max(1, width, height))
    if max_pixels > 0 and width * height > max_pixels:
        scale = min(scale, (max_pixels / max(1, width * height)) ** 0.5)
    if scale >= 0.999:
        return image
    next_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return image.resize(next_size, Image.Resampling.LANCZOS)


def _ttp_make_contact_sheet(images, labels=None, thumb_size=192, columns=4):
    items = [image.convert("RGB") for image in images if isinstance(image, Image.Image)]
    if not items:
        return None
    labels = labels or [str(index) for index in range(len(items))]
    columns = max(1, int(columns or 1))
    rows = int(np.ceil(len(items) / columns))
    label_height = 24
    gap = 8
    sheet = Image.new("RGB", (
        columns * thumb_size + (columns + 1) * gap,
        rows * (thumb_size + label_height) + (rows + 1) * gap,
    ), (28, 31, 36))
    draw = ImageDraw.Draw(sheet)
    for index, image in enumerate(items):
        col = index % columns
        row = index // columns
        x = gap + col * (thumb_size + gap)
        y = gap + row * (thumb_size + label_height + gap)
        preview = image.copy()
        preview.thumbnail((thumb_size, thumb_size), Image.Resampling.LANCZOS)
        offset = (x + (thumb_size - preview.width) // 2, y + (thumb_size - preview.height) // 2)
        sheet.paste(preview, offset)
        draw.rectangle((x, y, x + thumb_size - 1, y + thumb_size - 1), outline=(148, 163, 184), width=1)
        draw.text((x + 5, y + thumb_size + 6), str(labels[index])[:40], fill=(226, 232, 240))
    return sheet


def _ttp_extract_json_object(text):
    text = str(text or "").strip()
    text = _ttp_clean_qwen_vl_response(text)
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except Exception:
            pass
    return {}


def _ttp_extract_json_array(text):
    text = _ttp_clean_qwen_vl_response(text)
    start = text.find("[")
    end = text.rfind("]")
    if start >= 0 and end > start:
        try:
            parsed = json.loads(text[start:end + 1])
            return parsed if isinstance(parsed, list) else []
        except Exception:
            pass
    parsed = _ttp_extract_json_object(text)
    if isinstance(parsed, dict):
        for key in ("objects", "boxes", "bboxes", "tiles", "detections"):
            value = parsed.get(key)
            if isinstance(value, list):
                return value
    return []


def _ttp_clean_qwen_vl_response(text, prompt=""):
    text = str(text or "").strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    for marker in ("<|im_start|>assistant", "assistant\n", "assistant:"):
        if marker in text:
            text = text.split(marker)[-1].strip()
    if prompt and text.startswith(prompt):
        text = text[len(prompt):].strip()
    if "<|im_end|>" in text:
        text = text.split("<|im_end|>", 1)[0].strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text.strip(), flags=re.IGNORECASE).strip()
        if text.endswith("```"):
            text = text[:-3].strip()
    for prefix in ("Here is the JSON:", "Here is the result:", "Result:", "JSON:"):
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].strip()
    return text.strip()


def _ttp_compact_text(text, max_chars=1200):
    text = str(text or "").strip()
    max_chars = max(0, int(max_chars or 0))
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def _ttp_qwen_response_looks_like_instruction(text):
    lowered = str(text or "").strip().lower()
    if not lowered:
        return True
    instruction_fragments = [
        "analyze this tile",
        "return json",
        "return strict json",
        "refine this tile",
        "preserve all visible visual facts",
        "do not invent",
        "write prompt fields",
    ]
    return any(fragment in lowered for fragment in instruction_fragments)


def _ttp_jsonish_field(raw, field):
    text = str(raw or "")
    pattern = r'"' + re.escape(str(field)) + r'"\s*:\s*"((?:\\.|[^"\\])*)'
    match = re.search(pattern, text, flags=re.DOTALL)
    if not match:
        return ""
    value = match.group(1)
    try:
        return json.loads(f'"{value}"').strip()
    except Exception:
        return value.replace('\\"', '"').replace("\\n", "\n").strip()


def _ttp_extract_jsonish_qwen_fields(raw):
    if "{" not in str(raw or ""):
        return {}
    fields = {
        "label": _ttp_jsonish_field(raw, "label"),
        "caption": _ttp_jsonish_field(raw, "caption"),
        "prompt": _ttp_jsonish_field(raw, "prompt"),
        "negative": _ttp_jsonish_field(raw, "negative"),
    }
    if any(fields.values()):
        return fields
    return {}


def _ttp_parse_qwen_tile_record(raw, index):
    parsed = _ttp_extract_json_object(raw)
    if not isinstance(parsed, dict) or not parsed:
        parsed = _ttp_extract_jsonish_qwen_fields(raw)
    if not isinstance(parsed, dict) or not parsed:
        snippet = str(raw or "").strip().replace("\n", " ")[:240]
        raise RuntimeError(f"QwenVL did not return JSON fields for tile {index}. Raw: {snippet}")

    label = str(parsed.get("label", "")).strip()
    caption = str(parsed.get("caption", "")).strip()
    prompt = str(parsed.get("prompt", "")).strip()
    negative = str(parsed.get("negative", "")).strip()
    if not caption and not prompt:
        snippet = str(raw or "").strip().replace("\n", " ")[:240]
        raise RuntimeError(f"QwenVL JSON for tile {index} is missing caption/prompt. Raw: {snippet}")
    if _ttp_qwen_response_looks_like_instruction(caption) or _ttp_qwen_response_looks_like_instruction(prompt):
        snippet = str(raw or "").strip().replace("\n", " ")[:240]
        raise RuntimeError(f"QwenVL appears to have echoed instructions for tile {index}. Raw: {snippet}")
    return {
        "label": label,
        "caption": caption,
        "prompt": prompt,
        "negative": negative,
    }


def _ttp_qwen_tile_retry_messages(
    system_prompt,
    tile_pil,
    index,
    tile,
    output_language,
    global_prompt="",
    global_negative="",
    reference_context="",
):
    language_hint = {
        "english": "Use English.",
        "chinese": "Use Chinese.",
        "bilingual": "Use concise English and Chinese.",
    }.get(str(output_language), "Use English.")
    user_text = "\n".join([
        "Return compact JSON only. No markdown. No explanation. Stop immediately after the closing brace.",
        'Schema: {"label":"short label","caption":"visible facts","prompt":"img2img positive prompt","negative":"negative prompt"}',
        "Keep caption under 35 words, prompt under 55 words, negative under 25 words.",
        language_hint,
        f"Tile index: {index}. Existing label: {tile.get('label', tile.get('name', 'tile'))}.",
        f"Global prompt keywords: {_ttp_compact_text(global_prompt, 360)}",
        f"Global negative keywords: {_ttp_compact_text(global_negative, 240)}",
        f"Reference context: {_ttp_compact_text(reference_context, 360)}",
    ])
    return [
        {"role": "system", "content": _ttp_compact_text(system_prompt, 900) or "You output only compact JSON for tile image tagging."},
        {"role": "user", "content": [
            {"type": "text", "text": user_text},
            {"type": "image_url", "image_url": {"url": _ttp_pil_to_data_url(tile_pil)}},
        ]},
    ]


def _ttp_compose_tile_prompt(tile, caption, global_prompt, global_negative, merge_mode):
    label = str(tile.get("label", tile.get("name", "tile")) or "tile")
    caption = str(caption or "").strip()
    pieces = []
    if merge_mode in ("global_plus_caption", "global_plus_label_plus_caption") and str(global_prompt or "").strip():
        pieces.append(str(global_prompt).strip())
    if merge_mode == "global_plus_label_plus_caption" and label:
        pieces.append(label)
    if caption:
        pieces.append(caption)
    if not pieces:
        pieces.append(label)
    negative = str(global_negative or "").strip()
    return ", ".join(piece for piece in pieces if piece), negative


def _ttp_template_prompt_for_tile(tile, global_prompt, global_negative, merge_mode):
    label = str(tile.get("label", tile.get("name", "tile")) or "tile").lower()
    source = str(tile.get("source", "manual") or "manual")
    x, y, w, h = tile.get("sample_box", tile.get("core_box", [0, 0, 1, 1]))
    area_hint = "small detail tile" if int(w) * int(h) < 160000 else "large context tile"
    if "face" in label or "head" in label:
        caption = "detailed face region, sharp eyes, clean skin texture, consistent identity"
        prompt_label = "face"
    elif "hand" in label:
        caption = "detailed hands, natural fingers, clean anatomy, consistent lighting"
        prompt_label = "hands"
    elif "text" in label:
        caption = "sharp readable text, clean edges, preserved typography"
        prompt_label = "text"
    elif "background" in label:
        caption = "clean background, consistent lighting, coherent texture"
        prompt_label = "background"
    else:
        caption = f"{area_hint}, local visual detail, consistent style, coherent texture"
        prompt_label = label or source
    prompt, negative = _ttp_compose_tile_prompt({**tile, "label": prompt_label}, caption, global_prompt, global_negative, merge_mode)
    return {
        "label": prompt_label,
        "caption": caption,
        "prompt": prompt,
        "negative": negative,
    }


def _ttp_apply_color_transfer(image_tensor, ref_tensor, method, strength):
    if method == "off" or strength <= 0:
        return image_tensor
    try:
        from comfy_extras.nodes_post_processing import ColorTransfer
    except Exception as exc:
        raise RuntimeError("ComfyUI Transfer Color is unavailable; set color_correction to off.") from exc

    result = ColorTransfer.execute(
        image_tensor,
        ref_tensor,
        method,
        {"source_stats": "per_frame", "target_index": 0},
        float(strength),
    )
    if isinstance(result, (tuple, list)):
        return result[0]
    if hasattr(result, "result"):
        return result.result[0]
    try:
        return result[0]
    except Exception as exc:
        raise RuntimeError("Unexpected Transfer Color result from ComfyUI.") from exc


def _ttp_apply_local_mean_std_color(region, reference, mask=None, strength=1.0):
    strength = float(np.clip(float(strength), 0.0, 1.0))
    if strength <= 0:
        return region

    source = np.asarray(region, dtype=np.float32)
    target = np.asarray(reference, dtype=np.float32)
    if source.shape != target.shape or source.size == 0:
        return source

    if mask is None:
        weights = np.ones(source.shape[:2], dtype=np.float32)
    else:
        weights = np.asarray(mask[:, :, 0] if getattr(mask, "ndim", 0) == 3 else mask, dtype=np.float32)
        weights = np.clip(weights, 0.0, 1.0)
    if float(weights.sum()) < 16.0:
        weights = np.ones(source.shape[:2], dtype=np.float32)

    weights_3 = weights[:, :, None]
    weight_sum = max(1e-6, float(weights.sum()))
    source_mean = (source * weights_3).sum(axis=(0, 1)) / weight_sum
    target_mean = (target * weights_3).sum(axis=(0, 1)) / weight_sum
    source_var = (((source - source_mean) ** 2) * weights_3).sum(axis=(0, 1)) / weight_sum
    target_var = (((target - target_mean) ** 2) * weights_3).sum(axis=(0, 1)) / weight_sum
    source_std = np.sqrt(np.maximum(source_var, 1e-6))
    target_std = np.sqrt(np.maximum(target_var, 1e-6))
    corrected = (source - source_mean) * (target_std / source_std) + target_mean
    corrected = np.clip(corrected, 0.0, 1.0)
    return np.clip(source * (1.0 - strength) + corrected * strength, 0.0, 1.0)


def _ttp_apply_local_mean_std_color_torch(region, reference, mask=None, strength=1.0):
    strength = float(np.clip(float(strength), 0.0, 1.0))
    if strength <= 0:
        return region
    if region.shape != reference.shape or region.numel() == 0:
        return region

    if mask is None:
        weights = torch.ones(region.shape[:2], dtype=region.dtype, device=region.device)
    else:
        weights = mask[:, :, 0] if getattr(mask, "ndim", 0) == 3 else mask
        weights = torch.clamp(weights.to(dtype=region.dtype, device=region.device), 0.0, 1.0)
    if float(weights.sum().detach().cpu().item()) < 16.0:
        weights = torch.ones(region.shape[:2], dtype=region.dtype, device=region.device)

    weights_3 = weights[:, :, None]
    weight_sum = torch.clamp(weights.sum(), min=1e-6)
    source_mean = (region * weights_3).sum(dim=(0, 1)) / weight_sum
    target_mean = (reference * weights_3).sum(dim=(0, 1)) / weight_sum
    source_var = (((region - source_mean) ** 2) * weights_3).sum(dim=(0, 1)) / weight_sum
    target_var = (((reference - target_mean) ** 2) * weights_3).sum(dim=(0, 1)) / weight_sum
    source_std = torch.sqrt(torch.clamp(source_var, min=1e-6))
    target_std = torch.sqrt(torch.clamp(target_var, min=1e-6))
    corrected = torch.clamp((region - source_mean) * (target_std / source_std) + target_mean, 0.0, 1.0)
    return torch.clamp(region * (1.0 - strength) + corrected * strength, 0.0, 1.0)


def _ttp_pad_image_to_size(image, target_width, target_height):
    if image.width == target_width and image.height == target_height:
        return image

    padded = Image.new("RGB", (target_width, target_height))
    padded.paste(image, (0, 0))

    if image.width < target_width:
        right_edge = image.crop((image.width - 1, 0, image.width, image.height))
        right_fill = right_edge.resize((target_width - image.width, image.height), Image.Resampling.NEAREST)
        padded.paste(right_fill, (image.width, 0))

    if image.height < target_height:
        bottom_edge = padded.crop((0, image.height - 1, target_width, image.height))
        bottom_fill = bottom_edge.resize((target_width, target_height - image.height), Image.Resampling.NEAREST)
        padded.paste(bottom_fill, (0, image.height))

    return padded


def _ttp_expand_axis_to_size(start, length, target_length, limit, allow_before, allow_after):
    start = int(start)
    length = int(length)
    target_length = min(int(target_length), int(limit))
    if target_length <= length:
        return start, length

    deficit = target_length - length
    before = 0
    after = 0
    if allow_before:
        before = min(deficit, start)
        deficit -= before
    if allow_after:
        after_space = max(0, int(limit) - (start + length))
        after = min(deficit, after_space)
        deficit -= after
    if allow_before and deficit > 0:
        before_space = max(0, start - before)
        extra_before = min(deficit, before_space)
        before += extra_before
        deficit -= extra_before

    return start - before, length + before + after


def _ttp_actual_overlap_edges_from_sample(tile):
    sx, sy, sw, sh = tile["sample_box"]
    x, y, w, h = tile["core_box"]
    return {
        "left": max(0, x - sx),
        "right": max(0, sx + sw - (x + w)),
        "top": max(0, y - sy),
        "bottom": max(0, sy + sh - (y + h)),
    }


def _ttp_expand_smart_tile_samples_to_batch(tiles_meta, image_width, image_height, target_width, target_height):
    for tile in tiles_meta:
        sx, sy, sw, sh = tile["sample_box"]
        edges = tile.get("overlap_edges_px_source", {})
        sx, sw = _ttp_expand_axis_to_size(
            sx,
            sw,
            target_width,
            image_width,
            int(edges.get("left", 0)) > 0,
            int(edges.get("right", 0)) > 0,
        )
        sy, sh = _ttp_expand_axis_to_size(
            sy,
            sh,
            target_height,
            image_height,
            int(edges.get("top", 0)) > 0,
            int(edges.get("bottom", 0)) > 0,
        )
        tile["sample_box"] = [sx, sy, sw, sh]
        tile["overlap_edges_px_source"] = _ttp_actual_overlap_edges_from_sample(tile)
    return tiles_meta


def _ttp_crop_smart_tiles_from_meta(pil_image, tiles_meta, round_to):
    image_width, image_height = pil_image.size
    positions = []
    for tile in tiles_meta:
        sx, sy, sw, sh = tile["sample_box"]
        sw = max(1, _ttp_round_to(sw, round_to))
        sh = max(1, _ttp_round_to(sh, round_to))
        if sx + sw > image_width:
            sx = max(0, image_width - sw)
        if sy + sh > image_height:
            sy = max(0, image_height - sh)
        sw = min(sw, image_width - sx)
        sh = min(sh, image_height - sy)
        tile["sample_box"] = [sx, sy, sw, sh]
        x, y, w, h = tile["paste_box"]
        positions.append((x, y, x + w, y + h))

    max_tile_width = max(tile["sample_box"][2] for tile in tiles_meta)
    max_tile_height = max(tile["sample_box"][3] for tile in tiles_meta)
    max_tile_width = max(1, _ttp_round_to(max_tile_width, round_to))
    max_tile_height = max(1, _ttp_round_to(max_tile_height, round_to))
    _ttp_expand_smart_tile_samples_to_batch(tiles_meta, image_width, image_height, max_tile_width, max_tile_height)

    tile_images = []
    for tile in tiles_meta:
        sx, sy, sw, sh = tile["sample_box"]
        crop = pil_image.crop((sx, sy, sx + sw, sy + sh))
        tile_images.append(crop)
        tile["tile_canvas_box"] = [0, 0, sw, sh]

    tile_tensors = []
    for tile, crop in zip(tiles_meta, tile_images):
        tile["tile_canvas_size"] = [max_tile_width, max_tile_height]
        padded_crop = _ttp_pad_image_to_size(crop, max_tile_width, max_tile_height)
        tile_tensors.append(pil2tensor(padded_crop))

    tiles = torch.cat(tile_tensors, dim=0)
    preview = pil_image.copy()
    draw = ImageDraw.Draw(preview)
    colors = ["red", "lime", "cyan", "yellow", "magenta", "orange", "blue", "white"]
    for tile in tiles_meta:
        color = colors[tile["id"] % len(colors)]
        sx, sy, sw, sh = tile["sample_box"]
        x, y, w, h = tile["paste_box"]
        draw.rectangle((sx, sy, sx + sw, sy + sh), outline=color, width=2)
        draw.rectangle((x, y, x + w, y + h), outline="white", width=2)
        draw.text((x + 4, y + 4), f'{tile["id"]}:{tile["name"]}', fill=color)

    tile_meta = {
        "version": 2,
        "type": "ttp_smart_tile",
        "original_size": [image_width, image_height],
        "tiles": tiles_meta,
    }
    return tiles, tile_meta, positions, pil2tensor(preview)


def _ttp_crop_smart_tile_set_from_meta(pil_image, tiles_meta, round_to):
    image_width, image_height = pil_image.size
    positions = []
    tile_tensors = []
    set_tiles_meta = []
    for tile in tiles_meta:
        item = dict(tile)
        sx, sy, sw, sh = item["sample_box"]
        sw = max(1, _ttp_round_to(sw, round_to))
        sh = max(1, _ttp_round_to(sh, round_to))
        if sx + sw > image_width:
            sx = max(0, image_width - sw)
        if sy + sh > image_height:
            sy = max(0, image_height - sh)
        sw = min(sw, image_width - sx)
        sh = min(sh, image_height - sy)
        item["sample_box"] = [sx, sy, sw, sh]
        item["overlap_edges_px_source"] = _ttp_actual_overlap_edges_from_sample(item)
        item["tile_canvas_box"] = [0, 0, sw, sh]
        item["tile_canvas_size"] = [sw, sh]
        x, y, w, h = item["paste_box"]
        positions.append((x, y, x + w, y + h))
        crop = pil_image.crop((sx, sy, sx + sw, sy + sh))
        tile_tensors.append(pil2tensor(crop))
        set_tiles_meta.append(item)

    tile_meta = {
        "version": 3,
        "type": "ttp_smart_tile",
        "storage": "tile_set",
        "original_size": [image_width, image_height],
        "tiles": set_tiles_meta,
    }
    return {
        "version": 1,
        "type": "ttp_smart_tile_set",
        "original_size": [image_width, image_height],
        "tile_meta": tile_meta,
        "tile_images": tile_tensors,
        "positions": positions,
    }


def _ttp_input_image_files():
    input_dir = folder_paths.get_input_directory()
    files = [file for file in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, file))]
    try:
        files = folder_paths.filter_files_content_types(files, ["image"])
    except Exception:
        pass
    return sorted(files)


def _ttp_load_input_image(image):
    image_name = str(image or "").strip()
    if not image_name:
        raise ValueError("No image was selected. Use the official image upload/dropdown on this node, or connect source_image.")
    image_path = folder_paths.get_annotated_filepath(image_name)
    pil_image = Image.open(image_path)
    pil_image = ImageOps.exif_transpose(pil_image)
    return pil_image.convert("RGB")


def _ttp_get_interactive_source_image(source_image=None, image=""):
    if source_image is not None:
        if not isinstance(source_image, torch.Tensor) or source_image.ndim != 4 or int(source_image.shape[0]) <= 0:
            raise ValueError("source_image input must be a non-empty ComfyUI IMAGE tensor.")
        return tensor2pil(source_image[0].unsqueeze(0)).convert("RGB")
    return _ttp_load_input_image(image)


def _ttp_send_smart_tile_layout_event(node_id, layout_json=None, message="", ok=True):
    if PromptServer is None or not node_id:
        return
    payload = {
        "node_id": str(node_id),
        "ok": bool(ok),
        "message": str(message or ""),
    }
    if layout_json:
        payload["layout_json"] = layout_json
    try:
        PromptServer.instance.send_sync("ttp-smart-tile-layout", payload, PromptServer.instance.client_id)
    except Exception:
        pass


def _ttp_unpack_node_output(result, index):
    if isinstance(result, (tuple, list)):
        return result[index]
    if hasattr(result, "result"):
        return result.result[index]
    return result[index]


def _ttp_split_prompt_terms(prompt):
    terms = []
    for part in str(prompt or "").replace("\n", ",").split(","):
        term = part.strip()
        if term:
            terms.append(term)
    return terms


def _ttp_encode_sam3_prompt_conditioning(clip, prompt, max_detections=1):
    if clip is None:
        return None
    terms = _ttp_split_prompt_terms(prompt)
    if not terms:
        terms = [str(prompt or "").strip() or "foreground object"]

    entries = []
    base_meta = {}
    for term in terms:
        tokens = clip.tokenize(term)
        cond = clip.encode_from_tokens_scheduled(tokens)
        if not cond:
            continue
        meta = dict(cond[0][1]) if len(cond[0]) > 1 and isinstance(cond[0][1], dict) else {}
        if not base_meta:
            base_meta = dict(meta)
        entries.append({
            "cond": cond[0][0],
            "attention_mask": meta.get("attention_mask"),
            "max_detections": max(1, int(max_detections)),
            "prompt": term,
        })

    if not entries:
        return None
    if len(entries) == 1:
        meta = dict(base_meta)
        meta["sam3_prompt"] = entries[0]["prompt"]
        return [[entries[0]["cond"], meta]]

    meta = dict(base_meta)
    meta["sam3_multi_cond"] = entries
    meta["sam3_prompts"] = [entry["prompt"] for entry in entries]
    return [[entries[0]["cond"], meta]]


def _ttp_run_sam3_auto_layout(
    pil_image,
    vision_model,
    vision_conditioning,
    clip,
    auto_prompt,
    default_pad,
    default_blend,
    object_padding,
    max_tiles,
    allow_object_overlap,
    paint_mask_payload=None,
):
    if vision_model is None:
        raise ValueError("Connect an official SAM3/SAM3.1 model to vision_model before inference.")
    if vision_conditioning is None:
        per_prompt_max = max(1, int(max_tiles) // max(1, len(_ttp_split_prompt_terms(auto_prompt))))
        vision_conditioning = _ttp_encode_sam3_prompt_conditioning(clip, auto_prompt, per_prompt_max)
    if vision_conditioning is None:
        raise ValueError("Connect CLIP to this node, or connect SAM3 text conditioning to vision_conditioning, before SAM3 text-prompt inference.")

    try:
        from comfy_extras.nodes_sam3 import SAM3_Detect
    except Exception as exc:
        raise RuntimeError("Official ComfyUI SAM3 Detect is unavailable in this ComfyUI build.") from exc

    image_tensor = pil2tensor(pil_image)
    result = SAM3_Detect.execute(
        vision_model,
        image_tensor,
        conditioning=vision_conditioning,
        threshold=0.5,
        refine_iterations=2,
        individual_masks=True,
    )
    masks = _ttp_unpack_node_output(result, 0)
    bboxes = _ttp_unpack_node_output(result, 1)
    image_width, image_height = pil_image.size
    paint_mask = _ttp_decode_interactive_paint_mask(paint_mask_payload, image_width, image_height)
    paint_items, paint_masks = _ttp_paint_mask_to_items(paint_mask, image_width, image_height)
    detected_items = _ttp_flatten_bboxes(bboxes)
    detected_masks = _ttp_mask_tensor_to_pil_list(masks, image_width, image_height)
    while len(detected_masks) < len(detected_items):
        detected_masks.append(Image.new("L", (image_width, image_height), 0))
    layout_json = _ttp_boxes_to_auto_layout(
        detected_items + paint_items,
        image_width,
        image_height,
        default_pad=int(default_pad),
        default_blend=int(default_blend),
        object_padding=int(object_padding),
        max_tiles=int(max_tiles),
        include_background=False,
        allow_object_overlap=bool(allow_object_overlap),
        masks=detected_masks + paint_masks,
    )
    count = len(detected_items)
    prompt_note = "using internal auto_prompt CLIP encode" if clip is not None else "using external conditioning"
    paint_note = f", plus {len(paint_items)} paint mask region(s)" if paint_items else ""
    return layout_json, f"SAM3 inference created an auto tile layout from {count} detected object(s){paint_note}, {prompt_note}."


def _ttp_qwen_bbox_items(raw, image_width, image_height):
    items = []
    for index, item in enumerate(_ttp_extract_json_array(raw)):
        if not isinstance(item, dict):
            continue
        bbox = item.get("bbox", item.get("box", item.get("bbox_2d")))
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            continue
        values = [float(value) for value in bbox[:4]]
        if max(values) <= 1.5:
            x0, y0, x1, y1 = values[0] * image_width, values[1] * image_height, values[2] * image_width, values[3] * image_height
        elif max(values) <= 1000.0:
            x0, y0, x1, y1 = values[0] * image_width / 1000.0, values[1] * image_height / 1000.0, values[2] * image_width / 1000.0, values[3] * image_height / 1000.0
        else:
            x0, y0, x1, y1 = values
        x0, y0, x1, y1 = _ttp_clamp_box_xyxy([x0, y0, x1, y1], image_width, image_height)
        items.append({
            "x": x0,
            "y": y0,
            "width": x1 - x0,
            "height": y1 - y0,
            "label": str(item.get("label", item.get("name", f"object_{index + 1}"))),
            "score": float(item.get("score", 1.0) or 1.0),
            "prompt_hint": str(item.get("prompt_hint", "")),
        })
    return items


def _ttp_run_qwenvl3_auto_layout(
    pil_image,
    qwen_vl_model,
    auto_prompt,
    default_pad,
    default_blend,
    object_padding,
    max_tiles,
    allow_object_overlap,
    paint_mask_payload=None,
):
    if not isinstance(qwen_vl_model, dict) or qwen_vl_model.get("type") != "ttp_qwenvl3_model":
        raise ValueError("Connect TTP QwenVL3 Local Loader to qwen_vl_model before QwenVL3 Auto Tile.")
    image_width, image_height = pil_image.size
    prompt = "\n".join([
        _TTP_QWENVL_PRESETS["bbox_detect"]["instruction"],
        f"User focus prompt: {auto_prompt}",
        "Prioritize face, eyes, hands, text, glasses, foreground subject, and important small details.",
        "Return only the JSON list. No markdown. No explanation.",
    ])
    messages = [
        {"role": "system", "content": _TTP_QWENVL_PRESETS["bbox_detect"]["system"]},
        {"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": _ttp_pil_to_data_url(_ttp_resize_pil_for_qwen(pil_image, 1024, 1048576))}},
        ]},
    ]
    raw = _ttp_qwen_vl_local_chat(qwen_vl_model, messages, max_new_tokens=1024, temperature=0.0, seed=0)
    bboxes = _ttp_qwen_bbox_items(raw, image_width, image_height)
    paint_mask = _ttp_decode_interactive_paint_mask(paint_mask_payload, image_width, image_height)
    paint_items, paint_masks = _ttp_paint_mask_to_items(paint_mask, image_width, image_height)
    if not bboxes and not paint_items:
        snippet = _ttp_compact_text(str(raw or "").replace("\n", " "), 300)
        raise RuntimeError(f"QwenVL3 Auto Tile did not return bbox JSON. Raw: {snippet}")
    layout_json = _ttp_boxes_to_auto_layout(
        bboxes + paint_items,
        image_width,
        image_height,
        default_pad=int(default_pad),
        default_blend=int(default_blend),
        object_padding=int(object_padding),
        max_tiles=int(max_tiles),
        include_background=False,
        allow_object_overlap=bool(allow_object_overlap),
        masks=[Image.new("L", (image_width, image_height), 0) for _item in bboxes] + paint_masks,
    )
    paint_note = f", plus {len(paint_items)} paint mask region(s)" if paint_items else ""
    return layout_json, f"QwenVL3 inference created an auto tile layout from {len(bboxes)} detected object(s){paint_note}."


def _ttp_run_paint_mask_auto_layout(
    pil_image,
    paint_mask_payload,
    default_pad,
    default_blend,
    object_padding,
    max_tiles,
    allow_object_overlap,
):
    image_width, image_height = pil_image.size
    paint_mask = _ttp_decode_interactive_paint_mask(paint_mask_payload, image_width, image_height)
    paint_items, paint_masks = _ttp_paint_mask_to_items(paint_mask, image_width, image_height)
    if not paint_items:
        raise ValueError("Paint mask is empty. Brush on the image first, then click Auto Tile.")
    layout_json = _ttp_boxes_to_auto_layout(
        paint_items,
        image_width,
        image_height,
        default_pad=int(default_pad),
        default_blend=int(default_blend),
        object_padding=int(object_padding),
        max_tiles=int(max_tiles),
        include_background=False,
        allow_object_overlap=bool(allow_object_overlap),
        masks=paint_masks,
    )
    return layout_json, f"Paint mask created an auto tile layout from {len(paint_items)} painted region(s)."


def _ttp_interactive_layout_with_defaults(layout_json, default_pad, default_blend, include_full_image):
    try:
        layout = json.loads(str(layout_json or "").strip() or "{}")
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid interactive Smart Tile layout JSON: {exc}") from exc

    if not isinstance(layout, dict):
        raise ValueError("interactive Smart Tile layout must be a JSON object.")

    defaults = {
        "pad": int(max(0, default_pad)),
        "blend": int(max(0, default_blend)),
        "priority": 50,
        "importance": 1.0,
    }
    defaults.update(layout.get("defaults", {}) if isinstance(layout.get("defaults"), dict) else {})

    raw_tiles = layout.get("tiles", [])
    if not isinstance(raw_tiles, list):
        raw_tiles = []

    tiles = []
    if include_full_image:
        tiles.append({
            "name": "full_image",
            "x0": 0.0,
            "y0": 0.0,
            "x1": 1.0,
            "y1": 1.0,
            "pad": 0,
            "blend": max(int(default_blend), int(defaults.get("blend", 0))),
            "priority": 10,
            "importance": 0.5,
        })

    for index, tile in enumerate(raw_tiles):
        if isinstance(tile, dict):
            tiles.append({
                "name": tile.get("name", f"tile_{index + 1}"),
                **tile,
            })

    if not tiles:
        tiles = [
            {"name": "tile_1", "x0": 0.0, "y0": 0.0, "x1": 0.5, "y1": 0.5},
            {"name": "tile_2", "x0": 0.5, "y0": 0.0, "x1": 1.0, "y1": 0.5},
            {"name": "tile_3", "x0": 0.0, "y0": 0.5, "x1": 0.5, "y1": 1.0},
            {"name": "tile_4", "x0": 0.5, "y0": 0.5, "x1": 1.0, "y1": 1.0},
        ]

    return json.dumps({
        "type": "ttp_smart_tile_interactive_layout",
        "defaults": defaults,
        "tiles": tiles,
    })


class TTP_Smart_Tile_Layout_Experimental:
    @classmethod
    def INPUT_TYPES(cls):
        default_layout = json.dumps({
            "defaults": {"pad": 128, "blend": 48, "priority": 50, "importance": 1.0},
            "tiles": [
                {"name": "full_image", "x": 0, "y": 0, "w": 1.0, "h": 1.0, "pad": 0, "blend": 96, "priority": 10, "importance": 0.5}
            ]
        }, indent=2)
        return {
            "required": {
                "layout_json": ("STRING", {"default": default_layout, "multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("layout_json",)
    FUNCTION = "pass_layout"
    CATEGORY = "TTP/Smart Tile"

    def pass_layout(self, layout_json):
        return (layout_json,)


class TTP_Smart_Tile_Crop_Experimental:
    @classmethod
    def INPUT_TYPES(cls):
        default_layout = json.dumps({
            "defaults": {"pad": 128, "blend": 48, "priority": 50, "importance": 1.0},
            "tiles": [
                {"name": "full_image", "x": 0, "y": 0, "w": 1.0, "h": 1.0, "pad": 0, "blend": 96, "priority": 10, "importance": 0.5},
                {"name": "center_detail", "x": 0.25, "y": 0.25, "w": 0.5, "h": 0.5, "pad": 128, "blend": 48, "priority": 100, "importance": 1.0}
            ]
        }, indent=2)
        return {
            "required": {
                "image": ("IMAGE",),
                "layout_json": ("STRING", {"default": default_layout, "multiline": True}),
                "round_to": ("INT", {"default": 8, "min": 1, "max": 128, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "TTP_SMART_TILE_META", "LIST", "IMAGE")
    RETURN_NAMES = ("tiles", "tile_meta", "positions", "preview")
    FUNCTION = "crop_tiles"
    CATEGORY = "TTP/Smart Tile"

    def crop_tiles(self, image, layout_json, round_to=8):
        pil_image = tensor2pil(image[0].unsqueeze(0)).convert("RGB")
        image_width, image_height = pil_image.size
        tiles_meta = _ttp_parse_smart_tile_layout(layout_json, image_width, image_height)
        return _ttp_crop_smart_tiles_from_meta(pil_image, tiles_meta, round_to)


class TTP_Smart_Tile_Visual_Crop_Experimental:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "grid_columns": ("INT", {"default": 2, "min": 1, "max": 8, "step": 1}),
                "grid_rows": ("INT", {"default": 2, "min": 1, "max": 8, "step": 1}),
                "include_full_image": ("BOOLEAN", {"default": True}),
                "grid_pad": ("INT", {"default": 128, "min": 0, "max": 1024, "step": 8}),
                "grid_blend": ("INT", {"default": 64, "min": 0, "max": 512, "step": 8}),
                "grid_priority": ("FLOAT", {"default": 40.0, "min": 0.0, "max": 200.0, "step": 1.0}),
                "grid_importance": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 4.0, "step": 0.05}),
                "round_to": ("INT", {"default": 8, "min": 1, "max": 128, "step": 1}),
                "focus_1_enabled": ("BOOLEAN", {"default": False}),
                "focus_1_x": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}),
                "focus_1_y": ("FLOAT", {"default": 0.08, "min": 0.0, "max": 1.0, "step": 0.01}),
                "focus_1_w": ("FLOAT", {"default": 0.30, "min": 0.01, "max": 1.0, "step": 0.01}),
                "focus_1_h": ("FLOAT", {"default": 0.28, "min": 0.01, "max": 1.0, "step": 0.01}),
                "focus_2_enabled": ("BOOLEAN", {"default": False}),
                "focus_2_x": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
                "focus_2_y": ("FLOAT", {"default": 0.55, "min": 0.0, "max": 1.0, "step": 0.01}),
                "focus_2_w": ("FLOAT", {"default": 0.50, "min": 0.01, "max": 1.0, "step": 0.01}),
                "focus_2_h": ("FLOAT", {"default": 0.20, "min": 0.01, "max": 1.0, "step": 0.01}),
                "focus_pad": ("INT", {"default": 192, "min": 0, "max": 1024, "step": 8}),
                "focus_blend": ("INT", {"default": 64, "min": 0, "max": 512, "step": 8}),
                "focus_priority": ("FLOAT", {"default": 110.0, "min": 0.0, "max": 300.0, "step": 1.0}),
                "focus_importance": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("IMAGE", "TTP_SMART_TILE_META", "LIST", "IMAGE")
    RETURN_NAMES = ("tiles", "tile_meta", "positions", "preview")
    FUNCTION = "visual_crop_tiles"
    CATEGORY = "TTP/Smart Tile"

    def _add_tile_meta(self, raw_tiles, image_width, image_height):
        tiles_meta = []
        for index, tile in enumerate(raw_tiles):
            normalized = _ttp_normalize_tile_box({**tile, "id": index}, image_width, image_height, {})
            normalized["id"] = index
            tiles_meta.append(normalized)
        return _ttp_add_smart_tile_sample_boxes(tiles_meta, image_width, image_height)

    def visual_crop_tiles(
        self,
        image,
        grid_columns,
        grid_rows,
        include_full_image,
        grid_pad,
        grid_blend,
        grid_priority,
        grid_importance,
        round_to,
        focus_1_enabled,
        focus_1_x,
        focus_1_y,
        focus_1_w,
        focus_1_h,
        focus_2_enabled,
        focus_2_x,
        focus_2_y,
        focus_2_w,
        focus_2_h,
        focus_pad,
        focus_blend,
        focus_priority,
        focus_importance,
    ):
        pil_image = tensor2pil(image[0].unsqueeze(0)).convert("RGB")
        image_width, image_height = pil_image.size
        raw_tiles = []

        if include_full_image:
            raw_tiles.append({
                "name": "full_image",
                "x": 0.0,
                "y": 0.0,
                "w": 1.0,
                "h": 1.0,
                "pad": 0,
                "blend": max(grid_blend, focus_blend),
                "priority": max(0.0, grid_priority * 0.25),
                "importance": max(0.0, grid_importance * 0.5),
                "layer": 0,
                "occlusion_priority": 0,
            })

        for row in range(grid_rows):
            for col in range(grid_columns):
                raw_tiles.append({
                    "name": f"grid_{row}_{col}",
                    "x": col / grid_columns,
                    "y": row / grid_rows,
                    "w": 1.0 / grid_columns,
                    "h": 1.0 / grid_rows,
                    "pad": grid_pad,
                    "blend": grid_blend,
                    "priority": grid_priority,
                    "importance": grid_importance,
                    "layer": 1,
                    "occlusion_priority": 100,
                })

        focus_tiles = [
            (focus_1_enabled, "focus_1", focus_1_x, focus_1_y, focus_1_w, focus_1_h),
            (focus_2_enabled, "focus_2", focus_2_x, focus_2_y, focus_2_w, focus_2_h),
        ]
        for enabled, name, x, y, w, h in focus_tiles:
            if enabled:
                raw_tiles.append({
                    "name": name,
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                    "pad": focus_pad,
                    "blend": focus_blend,
                    "priority": focus_priority,
                    "importance": focus_importance,
                    "layer": 5,
                    "occlusion_priority": 3000,
                })

        tiles_meta = self._add_tile_meta(raw_tiles, image_width, image_height)
        return _ttp_crop_smart_tiles_from_meta(pil_image, tiles_meta, round_to)


class TTP_Smart_Tile_Interactive_Crop_Experimental:
    @classmethod
    def INPUT_TYPES(cls):
        default_layout = json.dumps({
            "tiles": [
                {"name": "tile_1", "x0": 0.0, "y0": 0.0, "x1": 0.5, "y1": 0.5},
                {"name": "tile_2", "x0": 0.5, "y0": 0.0, "x1": 1.0, "y1": 0.5},
                {"name": "tile_3", "x0": 0.0, "y0": 0.5, "x1": 0.5, "y1": 1.0},
                {"name": "tile_4", "x0": 0.5, "y0": 0.5, "x1": 1.0, "y1": 1.0},
            ]
        }, separators=(",", ":"))
        return {
            "required": {
                "image": (_ttp_input_image_files(), {"image_upload": True}),
                "layout_json": ("STRING", {"default": default_layout, "multiline": False}),
                "default_pad": ("INT", {"default": 128, "min": 0, "max": 2048, "step": 8}),
                "default_blend": ("INT", {"default": 64, "min": 0, "max": 1024, "step": 8}),
                "include_full_image": ("BOOLEAN", {"default": False}),
                "round_to": ("INT", {"default": 8, "min": 1, "max": 128, "step": 1}),
                "auto_detect_mode": (["none", "sam3.1", "qwenvl3"], {"default": "none"}),
                "auto_detect_request": ("INT", {"default": 0, "min": 0, "max": 2147483647, "step": 1}),
                "auto_prompt": ("STRING", {
                    "default": "person, face, hands, eyes, text, foreground object, important object",
                    "multiline": False,
                }),
                "allow_object_overlap": ("BOOLEAN", {"default": True}),
                "auto_object_padding": ("INT", {"default": 96, "min": 0, "max": 2048, "step": 8}),
                "auto_max_tiles": ("INT", {"default": 16, "min": 1, "max": 64, "step": 1}),
                "auto_paint_mask": ("STRING", {"default": "", "multiline": False}),
            },
            "optional": {
                "source_image": ("IMAGE",),
                "vision_model": ("MODEL",),
                "vision_conditioning": ("CONDITIONING",),
                "clip": ("CLIP",),
                "qwen_vl_model": ("TTP_QWENVL3_MODEL",),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "TTP_SMART_TILE_SET", "TTP_SMART_TILE_META", "LIST", "IMAGE", "STRING")
    RETURN_NAMES = ("source_image", "tiles", "tile_set", "tile_meta", "positions", "preview", "layout_json")
    FUNCTION = "interactive_crop_tiles"
    CATEGORY = "TTP/Smart Tile"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        image = str(kwargs.get("image", ""))
        fingerprint = {
            "image": image,
            "layout_json": kwargs.get("layout_json", ""),
            "default_pad": kwargs.get("default_pad"),
            "default_blend": kwargs.get("default_blend"),
            "include_full_image": kwargs.get("include_full_image"),
            "round_to": kwargs.get("round_to"),
            "auto_detect_mode": kwargs.get("auto_detect_mode", "none"),
            "auto_detect_request": kwargs.get("auto_detect_request", 0),
            "auto_prompt": kwargs.get("auto_prompt", ""),
            "allow_object_overlap": kwargs.get("allow_object_overlap", True),
            "auto_object_padding": kwargs.get("auto_object_padding", 96),
            "auto_max_tiles": kwargs.get("auto_max_tiles", 16),
            "auto_paint_mask": hashlib.sha256(str(kwargs.get("auto_paint_mask", "") or "").encode("utf-8")).hexdigest()[:16],
            "source_image": type(kwargs.get("source_image")).__name__ if kwargs.get("source_image") is not None else None,
            "vision_model": type(kwargs.get("vision_model")).__name__ if kwargs.get("vision_model") is not None else None,
            "vision_conditioning": type(kwargs.get("vision_conditioning")).__name__ if kwargs.get("vision_conditioning") is not None else None,
            "clip": type(kwargs.get("clip")).__name__ if kwargs.get("clip") is not None else None,
            "qwen_vl_model": type(kwargs.get("qwen_vl_model")).__name__ if kwargs.get("qwen_vl_model") is not None else None,
        }
        if image and folder_paths.exists_annotated_filepath(image):
            try:
                fingerprint["image_mtime"] = os.path.getmtime(folder_paths.get_annotated_filepath(image))
            except OSError:
                pass
        return json.dumps(fingerprint, sort_keys=True)

    @classmethod
    def VALIDATE_INPUTS(cls, image="", **kwargs):
        if image and not folder_paths.exists_annotated_filepath(image):
            return f"Invalid image file: {image}"
        return True

    def interactive_crop_tiles(
        self,
        image,
        layout_json,
        default_pad=128,
        default_blend=64,
        include_full_image=False,
        round_to=8,
        auto_detect_mode="none",
        auto_detect_request=0,
        auto_prompt="person, face, hands, eyes, text, foreground object, important object",
        allow_object_overlap=True,
        auto_object_padding=96,
        auto_max_tiles=16,
        auto_paint_mask="",
        source_image=None,
        vision_model=None,
        vision_conditioning=None,
        clip=None,
        qwen_vl_model=None,
        unique_id=None,
    ):
        pil_image = _ttp_get_interactive_source_image(source_image=source_image, image=image)
        image_width, image_height = pil_image.size
        mode = str(auto_detect_mode or "none").strip().lower()
        has_paint_mask = bool(str(auto_paint_mask or "").strip())
        if int(auto_detect_request or 0) > 0 and (mode != "none" or has_paint_mask):
            try:
                if mode == "sam3.1":
                    layout_json, message = _ttp_run_sam3_auto_layout(
                        pil_image,
                        vision_model,
                        vision_conditioning,
                        clip,
                        str(auto_prompt or ""),
                        int(default_pad),
                        int(default_blend),
                        int(auto_object_padding),
                        int(auto_max_tiles),
                        bool(allow_object_overlap),
                        auto_paint_mask,
                    )
                    _ttp_send_smart_tile_layout_event(unique_id, layout_json, message, ok=True)
                elif mode == "qwenvl3":
                    layout_json, message = _ttp_run_qwenvl3_auto_layout(
                        pil_image,
                        qwen_vl_model,
                        str(auto_prompt or ""),
                        int(default_pad),
                        int(default_blend),
                        int(auto_object_padding),
                        int(auto_max_tiles),
                        bool(allow_object_overlap),
                        auto_paint_mask,
                    )
                    _ttp_send_smart_tile_layout_event(unique_id, layout_json, message, ok=True)
                elif has_paint_mask:
                    layout_json, message = _ttp_run_paint_mask_auto_layout(
                        pil_image,
                        auto_paint_mask,
                        int(default_pad),
                        int(default_blend),
                        int(auto_object_padding),
                        int(auto_max_tiles),
                        bool(allow_object_overlap),
                    )
                    _ttp_send_smart_tile_layout_event(unique_id, layout_json, message, ok=True)
                else:
                    _ttp_send_smart_tile_layout_event(unique_id, None, f"Unsupported auto detect mode: {mode}", ok=False)
            except Exception as exc:
                _ttp_send_smart_tile_layout_event(unique_id, None, str(exc), ok=False)

        normalized_layout_json = _ttp_interactive_layout_with_defaults(
            layout_json,
            int(default_pad),
            int(default_blend),
            bool(include_full_image),
        )
        tiles_meta = _ttp_parse_smart_tile_layout(normalized_layout_json, image_width, image_height)
        tile_set = _ttp_crop_smart_tile_set_from_meta(pil_image, [dict(tile) for tile in tiles_meta], int(round_to))
        tiles, tile_meta, positions, preview = _ttp_crop_smart_tiles_from_meta(pil_image, tiles_meta, int(round_to))
        return (
            pil2tensor(pil_image),
            tiles,
            tile_set,
            tile_meta,
            positions,
            preview,
            normalized_layout_json,
        )


class TTP_Smart_Tile_Set_Preview_Experimental:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tile_set": ("TTP_SMART_TILE_SET", {"forceInput": True}),
                "mode": (["contact_sheet", "selected_tile"], {"default": "contact_sheet"}),
                "selected_index": ("INT", {"default": 0, "min": 0, "max": 63, "step": 1}),
                "thumbnail_size": ("INT", {"default": 256, "min": 64, "max": 1024, "step": 16}),
                "columns": ("INT", {"default": 4, "min": 1, "max": 8, "step": 1}),
                "show_labels": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "info")
    FUNCTION = "preview_tile_set"
    CATEGORY = "TTP/Smart Tile"

    def preview_tile_set(
        self,
        tile_set,
        mode="contact_sheet",
        selected_index=0,
        thumbnail_size=256,
        columns=4,
        show_labels=True,
    ):
        if not isinstance(tile_set, dict) or tile_set.get("type") != "ttp_smart_tile_set":
            raise ValueError("tile_set must come from TTP Smart Tile Interactive Crop")
        tile_images = tile_set.get("tile_images", [])
        tile_meta = tile_set.get("tile_meta", {})
        tiles_info = tile_meta.get("tiles", [])
        if not tile_images:
            raise ValueError("tile_set contains no tile images.")
        if len(tile_images) != len(tiles_info):
            raise ValueError(f"tile_set images ({len(tile_images)}) do not match tile metadata ({len(tiles_info)}).")

        index = _ttp_clamp(int(selected_index), 0, len(tile_images) - 1)
        thumbs = []
        info_lines = []
        for tile_index, image_tensor in enumerate(tile_images):
            tile = tiles_info[tile_index]
            pil = _ttp_image_tensor_to_pil(image_tensor)
            label = str(tile.get("label", tile.get("name", f"tile_{tile_index + 1}")))
            layer = int(tile.get("layer", 0))
            priority = float(tile.get("occlusion_priority", tile.get("priority", 0)))
            info_lines.append(f"{tile_index}: {label} {pil.width}x{pil.height} layer={layer} priority={priority:g}")
            thumbs.append((pil, label, layer, priority))

        if mode == "selected_tile":
            pil, label, layer, priority = thumbs[index]
            preview = pil.copy().convert("RGB")
            if show_labels:
                draw = ImageDraw.Draw(preview)
                text = f"{index}: {label}  {preview.width}x{preview.height}  L{layer}  P{priority:g}"
                draw.rectangle((0, 0, min(preview.width, 12 + len(text) * 7), 24), fill=(0, 0, 0))
                draw.text((6, 5), text, fill=(255, 255, 255))
            return (pil2tensor(preview), "\n".join(info_lines))

        thumb_size = int(thumbnail_size)
        columns = max(1, int(columns))
        rows = int(np.ceil(len(thumbs) / columns))
        label_height = 30 if show_labels else 0
        gap = 8
        sheet_width = columns * thumb_size + (columns + 1) * gap
        sheet_height = rows * (thumb_size + label_height) + (rows + 1) * gap
        sheet = Image.new("RGB", (sheet_width, sheet_height), (24, 24, 27))
        draw = ImageDraw.Draw(sheet)

        for tile_index, (pil, label, layer, priority) in enumerate(thumbs):
            col = tile_index % columns
            row = tile_index // columns
            x = gap + col * (thumb_size + gap)
            y = gap + row * (thumb_size + label_height + gap)
            tile_preview = pil.copy().convert("RGB")
            tile_preview.thumbnail((thumb_size, thumb_size), Image.Resampling.LANCZOS)
            box = Image.new("RGB", (thumb_size, thumb_size), (39, 39, 42))
            offset = ((thumb_size - tile_preview.width) // 2, (thumb_size - tile_preview.height) // 2)
            box.paste(tile_preview, offset)
            sheet.paste(box, (x, y))
            draw.rectangle((x, y, x + thumb_size - 1, y + thumb_size - 1), outline=(148, 163, 184), width=1)
            if show_labels:
                text = f"{tile_index}: {pil.width}x{pil.height} L{layer} P{priority:g}"
                draw.text((x + 4, y + thumb_size + 7), text, fill=(226, 232, 240))

        return (pil2tensor(sheet), "\n".join(info_lines))


class TTP_QwenVL3_Local_Loader_Experimental:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_file": (_ttp_qwenvl_safetensor_files(),),
                "model_family": (["auto", "qwen_vl"], {"default": "auto"}),
                "device": (["default", "cpu"], {"default": "default", "advanced": True}),
            }
        }

    RETURN_TYPES = ("TTP_QWENVL3_MODEL", "STRING")
    RETURN_NAMES = ("qwen_vl_model", "info")
    FUNCTION = "load_model"
    CATEGORY = "TTP/Smart Tile"

    def load_model(self, model_file, model_family="auto", device="default"):
        loaded = _ttp_load_qwenvl3_local_model(
            model_file,
            model_family=str(model_family),
            device_mode=str(device),
        )
        info = f"Loaded QwenVL tagging model from {loaded['model_file']} using ComfyUI native Qwen-VL loader"
        return (loaded, info)


class TTP_Smart_Tile_QwenVL_Prompt_Set_Builder_Experimental:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference_image_mode": (["none", "first_message", "every_tile", "contact_sheet"], {"default": "contact_sheet"}),
                "system_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "dynamicPrompts": False,
                }),
                "tile_instruction": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "dynamicPrompts": False,
                }),
                "global_prompt": ("STRING", {"default": "", "multiline": True, "dynamicPrompts": False}),
                "prompt_merge_mode": (["caption_only", "global_plus_caption", "global_plus_label_plus_caption"], {"default": "global_plus_label_plus_caption"}),
                "output_language": (["english", "chinese", "bilingual"], {"default": "english"}),
                "max_new_tokens": ("INT", {"default": 512, "min": 32, "max": 4096, "step": 16}),
                "temperature": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 2.0, "step": 0.05}),
                "prompt_preset": (list(_TTP_QWENVL_PRESETS.keys()), {"default": "tile_img2img_prompt"}),
                "qwen_max_side": ("INT", {"default": 768, "min": 0, "max": 4096, "step": 32}),
                "qwen_max_pixels": ("INT", {"default": 786432, "min": 0, "max": 16777216, "step": 65536}),
                "use_tile_cache": ("BOOLEAN", {"default": True}),
                "global_negative": ("STRING", {"default": "", "multiline": True, "dynamicPrompts": False}),
                "qwen_seed": ("INT", {"default": 123, "min": 0, "max": 2147483647, "step": 1}),
            },
            "optional": {
                "tile_set": ("TTP_SMART_TILE_SET", {"forceInput": True}),
                "reference_image": ("IMAGE",),
                "qwen_vl_model": ("TTP_QWENVL3_MODEL",),
            }
        }

    RETURN_TYPES = ("TTP_SMART_TILE_SET", "STRING", "STRING")
    RETURN_NAMES = ("tile_set", "prompt_set_json", "summary")
    FUNCTION = "build_prompt_set"
    CATEGORY = "TTP/Smart Tile"

    def build_prompt_set(
        self,
        tile_set=None,
        reference_image_mode="first_message",
        system_prompt="",
        tile_instruction="",
        global_prompt="",
        prompt_merge_mode="global_plus_label_plus_caption",
        output_language="english",
        max_new_tokens=512,
        temperature=0.2,
        prompt_preset="tile_img2img_prompt",
        qwen_max_side=768,
        qwen_max_pixels=786432,
        use_tile_cache=True,
        global_negative="",
        qwen_seed=123,
        reference_image=None,
        qwen_vl_model=None,
        **kwargs,
    ):
        requested_mode = str(kwargs.get("mode", "") or "").strip() if kwargs else ""
        if kwargs:
            reference_image_mode = kwargs.get("reference_image_mode", reference_image_mode)
            prompt_preset = kwargs.get("prompt_preset", prompt_preset)
            system_prompt = kwargs.get("system_prompt", system_prompt)
            tile_instruction = kwargs.get("tile_instruction", tile_instruction)
            global_prompt = kwargs.get("global_prompt", global_prompt)
            global_negative = kwargs.get("global_negative", global_negative)
            prompt_merge_mode = kwargs.get("prompt_merge_mode", prompt_merge_mode)
            output_language = kwargs.get("output_language", output_language)
            max_new_tokens = kwargs.get("max_new_tokens", max_new_tokens)
            temperature = kwargs.get("temperature", temperature)
            qwen_max_side = kwargs.get("qwen_max_side", qwen_max_side)
            qwen_max_pixels = kwargs.get("qwen_max_pixels", qwen_max_pixels)
            use_tile_cache = kwargs.get("use_tile_cache", use_tile_cache)
            reference_image = kwargs.get("reference_image", reference_image)
            qwen_vl_model = kwargs.get("qwen_vl_model", qwen_vl_model)
            qwen_seed = kwargs.get("qwen_seed", kwargs.get("seed", qwen_seed))

        preset_names = set(_TTP_QWENVL_PRESETS.keys())
        merge_modes = {"caption_only", "global_plus_caption", "global_plus_label_plus_caption"}
        languages = {"english", "chinese", "bilingual"}
        if str(system_prompt or "") in preset_names and str(prompt_preset or "") not in preset_names:
            shifted_prompt_preset = system_prompt
            shifted_system_prompt = tile_instruction
            shifted_tile_instruction = global_prompt
            shifted_global_prompt = global_negative
            shifted_global_negative = prompt_merge_mode
            shifted_prompt_merge_mode = output_language
            shifted_output_language = max_new_tokens
            shifted_max_new_tokens = temperature
            shifted_temperature = prompt_preset
            prompt_preset = shifted_prompt_preset
            system_prompt = shifted_system_prompt
            tile_instruction = shifted_tile_instruction
            global_prompt = shifted_global_prompt
            global_negative = shifted_global_negative
            prompt_merge_mode = shifted_prompt_merge_mode
            output_language = shifted_output_language
            max_new_tokens = shifted_max_new_tokens
            temperature = shifted_temperature

        if str(reference_image_mode or "") not in ("none", "first_message", "every_tile", "contact_sheet"):
            reference_image_mode = "contact_sheet"
        if str(prompt_preset or "") not in preset_names:
            prompt_preset = "tile_img2img_prompt"
        if str(prompt_merge_mode or "") not in merge_modes:
            prompt_merge_mode = "global_plus_label_plus_caption"
        if str(output_language or "") not in languages:
            output_language = "english"
        max_new_tokens = _ttp_safe_int(max_new_tokens, 512, 32, 4096)
        temperature = _ttp_safe_float(temperature, 0.2, 0.0, 2.0)
        qwen_max_side = _ttp_safe_int(qwen_max_side, 768, 0, 4096)
        qwen_max_pixels = _ttp_safe_int(qwen_max_pixels, 786432, 0, 16777216)
        use_tile_cache = _ttp_safe_bool(use_tile_cache, True)
        qwen_seed = _ttp_safe_int(qwen_seed, 123, 0, 2147483647)

        if isinstance(reference_image, dict) and reference_image.get("type") == "ttp_qwenvl3_model" and qwen_vl_model is None:
            qwen_vl_model = reference_image
            reference_image = None
        if isinstance(qwen_seed, dict) and qwen_seed.get("type") == "ttp_qwenvl3_model" and qwen_vl_model is None:
            qwen_vl_model = qwen_seed
            qwen_seed = 0
        tile_images, _tile_meta, tiles_info = _ttp_validate_tile_set(tile_set)
        next_tile_set = _ttp_clone_tile_set(tile_set)
        next_tiles = next_tile_set["tile_meta"]["tiles"]
        preset = _TTP_QWENVL_PRESETS.get(str(prompt_preset or ""), _TTP_QWENVL_PRESETS["tile_img2img_prompt"])
        effective_system_prompt = str(system_prompt or "").strip() or preset["system"]
        effective_tile_instruction = str(tile_instruction or "").strip() or preset["instruction"]
        reference_context = ""
        reference_pil = None
        if reference_image is not None:
            reference_pil = _ttp_image_tensor_to_pil(_ttp_first_image_tensor(reference_image))
            reference_pil = _ttp_resize_pil_for_qwen(reference_pil, qwen_max_side, qwen_max_pixels)

        effective_mode = "qwen_vl_local" if qwen_vl_model is not None else "template"

        if effective_mode == "qwen_vl_local" and qwen_vl_model is None:
            raise ValueError("qwen_vl_local mode requires TTP QwenVL3 Local Loader output.")

        def chat_fn(messages, retry=False):
            chat_temperature = 0.0 if retry else float(temperature)
            if effective_mode == "qwen_vl_local":
                return _ttp_qwen_vl_local_chat(
                    qwen_vl_model,
                    messages,
                    max_new_tokens=max_new_tokens,
                    temperature=chat_temperature,
                    seed=qwen_seed,
                )
            return ""

        contact_sheet_pil = None
        if effective_mode == "qwen_vl_local" and str(reference_image_mode) == "contact_sheet":
            contact_images = []
            contact_labels = []
            for index, (image_tensor, tile) in enumerate(zip(tile_images, next_tiles)):
                contact_images.append(_ttp_resize_pil_for_qwen(_ttp_image_tensor_to_pil(image_tensor), 256, 0))
                contact_labels.append(f'{index}: {tile.get("label", tile.get("name", "tile"))}')
            contact_sheet_pil = _ttp_resize_pil_for_qwen(_ttp_make_contact_sheet(contact_images, contact_labels), qwen_max_side, qwen_max_pixels)

        if effective_mode == "qwen_vl_local" and reference_pil is not None and reference_image_mode == "first_message":
            messages = [
                {"role": "system", "content": _ttp_compact_text(effective_system_prompt, 900)},
                {"role": "user", "content": [
                    {"type": "text", "text": "Analyze this reference image for identity, style, lighting, material, camera perspective, and global visual consistency. Return one concise paragraph under 80 words. Do not output JSON."},
                    {"type": "image_url", "image_url": {"url": _ttp_pil_to_data_url(reference_pil)}},
                ]},
            ]
            reference_context = _ttp_compact_text(chat_fn(messages), 900)
        elif effective_mode == "qwen_vl_local" and contact_sheet_pil is not None:
            messages = [
                {"role": "system", "content": _ttp_compact_text(effective_system_prompt, 900)},
                {"role": "user", "content": [
                    {"type": "text", "text": "This is a contact sheet of all Smart Tile crops. Each crop is labeled by tile index. Summarize the global subject, style, lighting, identity, and how tile indices relate to the whole image. Return one concise paragraph under 100 words. Do not output JSON."},
                    {"type": "image_url", "image_url": {"url": _ttp_pil_to_data_url(contact_sheet_pil)}},
                ]},
            ]
            reference_context = _ttp_compact_text(chat_fn(messages), 1000)

        prompt_records = []
        for index, (image_tensor, tile) in enumerate(zip(tile_images, next_tiles)):
            raw = ""
            retry_raw = ""
            if effective_mode == "template":
                record = _ttp_template_prompt_for_tile(tile, global_prompt, global_negative, prompt_merge_mode)
            elif effective_mode == "qwen_vl_local":
                source_tile_hash = _ttp_tile_hash(image_tensor)
                tile_pil = _ttp_resize_pil_for_qwen(_ttp_image_tensor_to_pil(image_tensor), qwen_max_side, qwen_max_pixels)
                language_hint = {
                    "english": "Write prompt fields in English.",
                    "chinese": "Write prompt fields in Chinese.",
                    "bilingual": "Write prompt fields in English and Chinese.",
                }.get(str(output_language), "Write prompt fields in English.")
                user_text = "\n".join([
                    str(effective_tile_instruction or ""),
                    language_hint,
                    "Return one compact JSON object only. No markdown, no explanation, no repeated instruction. Stop immediately after the closing brace.",
                    "Keep caption under 45 words, prompt under 75 words, negative under 30 words.",
                    f"Tile index: {index}. Existing label: {tile.get('label', tile.get('name', 'tile'))}.",
                    f"Global prompt: {_ttp_compact_text(global_prompt, 900)}",
                    f"Global negative: {_ttp_compact_text(global_negative, 600)}",
                    f"Reference context: {_ttp_compact_text(reference_context, 700)}",
                ])
                content = [{"type": "text", "text": user_text}]
                if reference_pil is not None and reference_image_mode == "every_tile":
                    content.append({"type": "image_url", "image_url": {"url": _ttp_pil_to_data_url(reference_pil)}})
                elif contact_sheet_pil is not None:
                    content.append({"type": "image_url", "image_url": {"url": _ttp_pil_to_data_url(contact_sheet_pil)}})
                content.append({"type": "image_url", "image_url": {"url": _ttp_pil_to_data_url(tile_pil)}})
                messages = [
                    {"role": "system", "content": _ttp_compact_text(effective_system_prompt, 1200)},
                    {"role": "user", "content": content},
                ]
                cache_key = json.dumps({
                    "model_file": qwen_vl_model.get("model_file", "") if isinstance(qwen_vl_model, dict) else "",
                    "tile_index": int(index),
                    "tile_name": tile.get("name", ""),
                    "sample_box": tile.get("sample_box", []),
                    "core_box": tile.get("core_box", []),
                    "tile_hash": source_tile_hash,
                    "system_prompt": effective_system_prompt,
                    "tile_instruction": effective_tile_instruction,
                    "global_prompt": global_prompt,
                    "global_negative": global_negative,
                    "reference_context": reference_context,
                    "reference_mode": reference_image_mode,
                    "prompt_merge_mode": prompt_merge_mode,
                    "output_language": output_language,
                    "qwen_seed": int(qwen_seed if qwen_seed is not None else 0),
                    "max_side": int(qwen_max_side or 0),
                    "max_pixels": int(qwen_max_pixels or 0),
                }, sort_keys=True, ensure_ascii=False)
                cached = _TTP_QWENVL_PROMPT_CACHE.get(cache_key) if bool(use_tile_cache) else None
                if cached:
                    raw = cached.get("raw", "")
                    retry_raw = cached.get("retry_raw", "")
                    parsed = dict(cached.get("parsed", {}))
                else:
                    raw = chat_fn(messages)
                    parsed = None
                if not str(raw or "").strip():
                    raise RuntimeError(f"QwenVL returned an empty response for tile {index}.")
                if parsed is None:
                    try:
                        parsed = _ttp_parse_qwen_tile_record(raw, index)
                    except RuntimeError as first_exc:
                        retry_messages = _ttp_qwen_tile_retry_messages(
                            effective_system_prompt,
                            tile_pil,
                            index,
                            tile,
                            output_language,
                            global_prompt=global_prompt,
                            global_negative=global_negative,
                            reference_context=reference_context,
                        )
                        retry_raw = chat_fn(retry_messages, retry=True)
                        if not str(retry_raw or "").strip():
                            raise RuntimeError(f"QwenVL retry returned an empty response for tile {index}. First error: {first_exc}") from first_exc
                        try:
                            parsed = _ttp_parse_qwen_tile_record(retry_raw, index)
                        except RuntimeError as retry_exc:
                            first_snippet = _ttp_compact_text(str(raw or "").replace("\n", " "), 240)
                            retry_snippet = _ttp_compact_text(str(retry_raw or "").replace("\n", " "), 240)
                            raise RuntimeError(
                                f"QwenVL failed to produce usable JSON for tile {index} after retry. "
                                f"First error: {first_exc} First raw: {first_snippet} Retry raw: {retry_snippet}"
                            ) from retry_exc
                    if bool(use_tile_cache):
                        _TTP_QWENVL_PROMPT_CACHE[cache_key] = {"raw": raw, "retry_raw": retry_raw, "parsed": dict(parsed)}
                caption = parsed.get("caption", "")
                prompt = parsed.get("prompt", "")
                negative = parsed.get("negative", global_negative) or global_negative
                label = parsed.get("label", tile.get("label", tile.get("name", "tile"))) or tile.get("label", tile.get("name", "tile"))
                if not prompt:
                    prompt, negative = _ttp_compose_tile_prompt({**tile, "label": label}, caption, global_prompt, negative, prompt_merge_mode)
                record = {
                    "label": str(label),
                    "caption": str(caption),
                    "prompt": str(prompt),
                    "negative": str(negative),
                }
            else:
                raise ValueError(f"Unsupported prompt set builder mode: {effective_mode}")

            tile.update({
                "label": record["label"],
                "caption": record["caption"],
                "prompt": record["prompt"],
                "negative": record["negative"],
                "prompt_tag": tile.get("prompt_tag", f"tile_{index}_{record['label']}"),
                "prompt_source": effective_mode,
                "prompt_preset": str(prompt_preset or ""),
                "qwen_model": str(qwen_vl_model.get("model_file", "")) if effective_mode == "qwen_vl_local" and isinstance(qwen_vl_model, dict) else "",
                "tile_hash": _ttp_tile_hash(image_tensor),
                "qwen_input_size": [tile_pil.width, tile_pil.height] if effective_mode == "qwen_vl_local" else [],
                "qwen_cache": "hit" if effective_mode == "qwen_vl_local" and cached else "miss" if effective_mode == "qwen_vl_local" else "",
            })
            if raw:
                tile["qwen_raw"] = raw
            if retry_raw:
                tile["qwen_raw_retry"] = retry_raw
            prompt_records.append({
                "index": index,
                "name": tile.get("name", f"tile_{index}"),
                "label": tile["label"],
                "caption": tile["caption"],
                "prompt": tile["prompt"],
                "negative": tile["negative"],
                "prompt_tag": tile["prompt_tag"],
            })

        prompt_set_json = json.dumps({
            "type": "ttp_smart_tile_prompt_set",
            "mode": effective_mode,
            "requested_mode": requested_mode or effective_mode,
            "reference_image_mode": reference_image_mode,
            "prompt_preset": str(prompt_preset or ""),
            "qwen_seed": int(qwen_seed if qwen_seed is not None else 0),
            "model_file": qwen_vl_model.get("model_file", "") if effective_mode == "qwen_vl_local" and isinstance(qwen_vl_model, dict) else "",
            "qwen_max_side": int(qwen_max_side or 0),
            "qwen_max_pixels": int(qwen_max_pixels or 0),
            "tiles": prompt_records,
        }, ensure_ascii=False, indent=2)
        source_name = qwen_vl_model.get("model_file", "") if effective_mode == "qwen_vl_local" and isinstance(qwen_vl_model, dict) else ""
        summary_lines = [f"mode={effective_mode} model={source_name or 'none'} tiles={len(prompt_records)}"]
        summary_lines.extend(f'{item["index"]}: {item["label"]} -> {item["prompt"]}' for item in prompt_records)
        summary = "\n".join(summary_lines)
        return (next_tile_set, prompt_set_json, summary)


class TTP_Smart_Tile_Loop_Source_Experimental:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tile_set": ("TTP_SMART_TILE_SET", {"forceInput": True}),
                "session_id": ("STRING", {"default": "default", "multiline": False}),
                "restart_request": ("INT", {"default": 0, "min": 0, "max": 2147483647, "step": 1}),
                "loop_request": ("INT", {"default": 0, "min": 0, "max": 2147483647, "step": 1}),
            },
            "optional": {
                "clip": ("CLIP",),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            }
        }

    RETURN_TYPES = ("IMAGE", "TTP_SMART_TILE_TASK", "INT", "INT", "BOOLEAN", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("image", "tile_task", "index", "count", "done", "status", "prompt", "negative", "caption", "label", "prompt_tag", "positive_conditioning", "negative_conditioning")
    FUNCTION = "loop_source"
    CATEGORY = "TTP/Smart Tile"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        session_id = str(kwargs.get("session_id", "default") or "default")
        session = _TTP_SMART_TILE_LOOP_SESSIONS.get(session_id, {})
        tile_set = kwargs.get("tile_set")
        fingerprint = _ttp_tile_set_fingerprint(tile_set) if isinstance(tile_set, dict) else None
        return json.dumps({
            "session_id": session_id,
            "restart_request": kwargs.get("restart_request", 0),
            "loop_request": kwargs.get("loop_request", 0),
            "fingerprint": fingerprint,
            "index": session.get("index", 0),
            "done": session.get("done", False),
            "clip": type(kwargs.get("clip")).__name__ if kwargs.get("clip") is not None else None,
        }, sort_keys=True)

    def loop_source(self, tile_set, session_id="default", restart_request=0, loop_request=0, clip=None, unique_id=None):
        tile_images, tile_meta, tiles_info = _ttp_validate_tile_set(tile_set)
        session_key = str(session_id or "default")
        fingerprint = _ttp_tile_set_fingerprint(tile_set)
        session = _TTP_SMART_TILE_LOOP_SESSIONS.get(session_key)
        should_restart = (
            session is None
            or session.get("fingerprint") != fingerprint
            or int(session.get("restart_request", -1)) != int(restart_request)
        )
        if should_restart:
            session = {
                "fingerprint": fingerprint,
                "restart_request": int(restart_request),
                "source_node_id": str(unique_id or ""),
                "source_tile_set": _ttp_clone_tile_set(tile_set),
                "processed_tile_set": _ttp_clone_tile_set(tile_set),
                "index": 0,
                "done": False,
            }
            _TTP_SMART_TILE_LOOP_SESSIONS[session_key] = session

        count = len(tile_images)
        index = _ttp_clamp(int(session.get("index", 0)), 0, max(0, count - 1))
        done = bool(session.get("done", False)) or count <= 0
        if done:
            index = max(0, count - 1)

        task = {
            "type": "ttp_smart_tile_task",
            "session_id": session_key,
            "source_node_id": str(unique_id or session.get("source_node_id", "")),
            "index": int(index),
            "count": int(count),
            "done": bool(done),
            "tile_meta": dict(tiles_info[index]) if count else {},
        }
        image = tile_images[index] if count else pil2tensor(Image.new("RGB", (1, 1), "black"))
        status = "done" if done else f"tile {index + 1}/{count}"
        current_tile = tiles_info[index] if count else {}
        prompt = str(current_tile.get("prompt", ""))
        negative = str(current_tile.get("negative", ""))
        return (
            image,
            task,
            int(index),
            int(count),
            bool(done),
            status,
            prompt,
            negative,
            str(current_tile.get("caption", "")),
            str(current_tile.get("label", current_tile.get("name", ""))),
            str(current_tile.get("prompt_tag", "")),
            _ttp_encode_text_conditioning(clip, prompt),
            _ttp_encode_text_conditioning(clip, negative),
        )


class TTP_Smart_Tile_Loop_Collect_Experimental:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tile_task": ("TTP_SMART_TILE_TASK", {"forceInput": True}),
                "processed_image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("TTP_SMART_TILE_SET", "BOOLEAN", "INT", "STRING")
    RETURN_NAMES = ("tile_set", "done", "next_index", "status")
    FUNCTION = "loop_collect"
    CATEGORY = "TTP/Smart Tile"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        task = kwargs.get("tile_task", {})
        session_id = task.get("session_id", "") if isinstance(task, dict) else ""
        session = _TTP_SMART_TILE_LOOP_SESSIONS.get(session_id, {})
        return json.dumps({
            "session_id": session_id,
            "task_index": task.get("index") if isinstance(task, dict) else None,
            "session_index": session.get("index", 0),
            "done": session.get("done", False),
        }, sort_keys=True)

    def loop_collect(self, tile_task, processed_image):
        if not isinstance(tile_task, dict) or tile_task.get("type") != "ttp_smart_tile_task":
            raise ValueError("tile_task must come from TTP Smart Tile Loop Source")
        session_id = str(tile_task.get("session_id", "default"))
        session = _TTP_SMART_TILE_LOOP_SESSIONS.get(session_id)
        if session is None:
            raise ValueError(f"Smart Tile loop session not found: {session_id}")

        processed_tile_set = session["processed_tile_set"]
        tile_images, _tile_meta, _tiles_info = _ttp_validate_tile_set(processed_tile_set)
        count = len(tile_images)
        index = _ttp_clamp(int(tile_task.get("index", 0)), 0, max(0, count - 1))
        if not bool(tile_task.get("done", False)) and count > 0:
            tile_images[index] = _ttp_first_image_tensor(processed_image)

        next_index = index + 1
        done = next_index >= count
        session["index"] = min(next_index, max(0, count - 1))
        session["done"] = bool(done)
        status = "done" if done else f"next tile {next_index + 1}/{count}"
        event_task = dict(tile_task)
        event_task["index"] = min(next_index, max(0, count - 1))
        _ttp_send_smart_tile_loop_event(event_task, done=done, message=status)
        return (_ttp_clone_tile_set(processed_tile_set), bool(done), int(next_index), status)


class TTP_Smart_Tile_Image_Upscale_Prep_Experimental:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "scale": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 16.0, "step": 0.05}),
                "round_to": ("INT", {"default": 8, "min": 1, "max": 256, "step": 1}),
                "resampling": (["lanczos", "bicubic", "bilinear", "area", "nearest-exact", "nearest"], {"default": "lanczos"}),
                "max_megapixels": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 64.0, "step": 0.05}),
                "use_upscale_model": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "upscale_model": ("UPSCALE_MODEL",),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "info")
    FUNCTION = "upscale_tile"
    CATEGORY = "TTP/Smart Tile"

    def upscale_tile(
        self,
        image,
        scale=2.0,
        round_to=8,
        resampling="lanczos",
        max_megapixels=0.0,
        use_upscale_model=True,
        upscale_model=None,
    ):
        tile = _ttp_first_image_tensor(image)
        width, height = _ttp_image_tensor_size(tile[0])
        target_width, target_height, capped = _ttp_smart_upscale_target_size(
            width,
            height,
            float(scale),
            int(round_to),
            float(max_megapixels),
        )
        model_used = bool(use_upscale_model and upscale_model is not None)
        if model_used:
            upscaled = _ttp_upscale_image_with_model(upscale_model, tile)
            if _ttp_image_tensor_size(upscaled[0]) != (target_width, target_height):
                upscaled = _ttp_resize_image_tensor(upscaled, target_width, target_height, str(resampling))
        else:
            upscaled = _ttp_resize_image_tensor(tile, target_width, target_height, str(resampling))
        effective_scale_x = target_width / max(1, width)
        effective_scale_y = target_height / max(1, height)
        cap_text = " capped" if capped else ""
        model_text = "model" if model_used else str(resampling)
        info = (
            f"{width}x{height} -> {target_width}x{target_height} "
            f"requested_scale={float(scale):g} effective_scale={effective_scale_x:.4g}x{effective_scale_y:.4g} "
            f"round_to={int(round_to)} max_megapixels={float(max_megapixels):g}{cap_text} method={model_text}"
        )
        return (upscaled, info)


class TTP_Smart_Tile_Assemble_Experimental:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "blend_multiplier": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05}),
                "output_scale": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 16.0, "step": 0.05}),
                "use_priority": ("BOOLEAN", {"default": True}),
                "tile_alignment": (["center_crop", "crop_or_pad", "resize", "off"], {"default": "center_crop"}),
                "edge_crop_px": ("INT", {"default": 0, "min": 0, "max": 512, "step": 1}),
                "color_correction": (["off", "local_mean_std", "reinhard_lab", "mkl_lab", "histogram"], {"default": "off"}),
                "color_strength": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}),
                "mask_blend_mode": (["off", "auto", "mask_only", "mask_feather"], {"default": "mask_feather"}),
                "pixel_alignment": (["off", "mask_edge_match", "edge_match"], {"default": "off"}),
                "pixel_alignment_radius": ("INT", {"default": 8, "min": 0, "max": 64, "step": 1}),
                "pixel_alignment_device": (["auto", "cpu", "gpu"], {"default": "auto"}),
                "large_tile_policy": (["use_if_higher_resolution", "context_only", "always_use"], {"default": "use_if_higher_resolution"}),
                "large_tile_area_threshold": ("FLOAT", {"default": 0.55, "min": 0.05, "max": 1.0, "step": 0.01}),
                "min_tile_scale_ratio": ("FLOAT", {"default": 0.95, "min": 0.25, "max": 2.0, "step": 0.05}),
                "context_tile_weight": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.05}),
                "assemble_device": (["auto", "cpu", "gpu"], {"default": "auto"}),
                "assemble_mode": (["final_only", "always"], {"default": "final_only"}),
                "base_canvas_mode": (["auto", "black", "base_image", "source_image"], {"default": "auto"}),
                "small_tile_on_top": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "sampled_tiles": ("IMAGE",),
                "tile_meta": ("TTP_SMART_TILE_META",),
                "tile_set": ("TTP_SMART_TILE_SET",),
                "base_image": ("IMAGE",),
                "source_image": ("IMAGE",),
                "color_reference_image": ("IMAGE",),
                "done": ("BOOLEAN", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("image", "weight_preview")
    FUNCTION = "assemble_tiles"
    CATEGORY = "TTP/Smart Tile"

    def assemble_tiles(
        self,
        blend_multiplier=1.0,
        output_scale=0.0,
        use_priority=True,
        tile_alignment="center_crop",
        edge_crop_px=0,
        color_correction="off",
        color_strength=0.35,
        mask_blend_mode="mask_feather",
        pixel_alignment="off",
        pixel_alignment_radius=8,
        pixel_alignment_device="auto",
        large_tile_policy="use_if_higher_resolution",
        large_tile_area_threshold=0.55,
        min_tile_scale_ratio=0.95,
        context_tile_weight=0.25,
        assemble_device="auto",
        assemble_mode="final_only",
        base_canvas_mode="auto",
        small_tile_on_top=False,
        sampled_tiles=None,
        tile_meta=None,
        tile_set=None,
        base_image=None,
        source_image=None,
        color_reference_image=None,
        done=None,
    ):
        tile_images = None
        if tile_set is not None:
            if not isinstance(tile_set, dict) or tile_set.get("type") != "ttp_smart_tile_set":
                raise ValueError("tile_set must come from TTP Smart Tile Interactive Crop")
            tile_meta = tile_set.get("tile_meta")
            tile_images = tile_set.get("tile_images", [])

        if not isinstance(tile_meta, dict) or tile_meta.get("type") != "ttp_smart_tile":
            raise ValueError("tile_meta must come from TTP Smart Tile Crop")

        tiles_info = tile_meta.get("tiles", [])
        if tile_images is None:
            if sampled_tiles is None:
                raise ValueError("Connect either tile_set, or sampled_tiles plus tile_meta.")
            if len(tiles_info) != sampled_tiles.shape[0]:
                raise ValueError(f"sampled_tiles batch ({sampled_tiles.shape[0]}) does not match tile_meta tiles ({len(tiles_info)})")
            tile_images = [sampled_tiles[index] for index in range(int(sampled_tiles.shape[0]))]
        elif len(tile_images) != len(tiles_info):
            raise ValueError(f"tile_set images ({len(tile_images)}) do not match tile_meta tiles ({len(tiles_info)})")

        requested_assemble_mode = str(assemble_mode or "final_only")
        effective_assemble_mode = requested_assemble_mode
        if requested_assemble_mode == "always" and str(pixel_alignment) != "off" and done is not None:
            effective_assemble_mode = "final_only"

        canvas_mode = str(base_canvas_mode or "auto")
        if canvas_mode == "base_image":
            base_source_image = base_image
        elif canvas_mode == "source_image":
            base_source_image = source_image
        elif canvas_mode == "black":
            base_source_image = None
        else:
            canvas_mode = "auto"
            base_source_image = base_image if base_image is not None else source_image

        if effective_assemble_mode == "final_only" and done is not None and not bool(done):
            return _ttp_assemble_placeholder_output(tile_images, base_image=base_source_image)

        original_width, original_height = tile_meta["original_size"]
        if output_scale <= 0:
            inferred_scales = []
            scale_source_image = None
            if canvas_mode == "base_image":
                scale_source_image = base_image
            elif canvas_mode == "source_image":
                scale_source_image = source_image
            elif canvas_mode == "auto" and base_image is not None:
                scale_source_image = base_image
            if scale_source_image is not None:
                base_width, base_height = _ttp_image_tensor_size(scale_source_image[0])
                inferred_scales.append(float(base_width) / max(1, original_width))
                inferred_scales.append(float(base_height) / max(1, original_height))
            else:
                for index, tile in enumerate(tiles_info):
                    _, _, sw, sh = tile["sample_box"]
                    canvas_width, canvas_height = tile.get("tile_canvas_size", [sw, sh])
                    canvas_box = tile.get("tile_canvas_box", [0, 0, sw, sh])
                    canvas_content_w = max(1, float(canvas_box[2]) - float(canvas_box[0]))
                    canvas_content_h = max(1, float(canvas_box[3]) - float(canvas_box[1]))
                    tile_width, tile_height = _ttp_image_tensor_size(tile_images[index])
                    content_width = float(tile_width) * (canvas_content_w / max(1, canvas_width))
                    content_height = float(tile_height) * (canvas_content_h / max(1, canvas_height))
                    inferred_scales.append(content_width / max(1, sw))
                    inferred_scales.append(content_height / max(1, sh))
            output_scale = float(np.median(inferred_scales)) if inferred_scales else 1.0
            output_scale = max(0.01, output_scale)

        output_width = max(1, int(round(original_width * output_scale)))
        output_height = max(1, int(round(original_height * output_scale)))

        assemble_device_mode = str(assemble_device or "auto").lower()
        assemble_torch_device = _ttp_alignment_torch_device(assemble_device_mode)
        assemble_fallback_reason = ""
        if assemble_torch_device is None and assemble_device_mode == "gpu":
            assemble_fallback_reason = "gpu_unavailable"
        use_gpu_assemble = assemble_torch_device is not None

        if base_source_image is not None:
            base_pil = tensor2pil(base_source_image[0].unsqueeze(0)).convert("RGB")
            if base_pil.size != (output_width, output_height):
                base_pil = base_pil.resize((output_width, output_height), Image.Resampling.LANCZOS)
            base_array = np.array(base_pil).astype(np.float32) / 255.0
            if use_gpu_assemble:
                canvas = torch.as_tensor(base_array, dtype=torch.float32, device=assemble_torch_device)
                weights = torch.ones((output_height, output_width, 1), dtype=torch.float32, device=assemble_torch_device)
            else:
                canvas = base_array
                weights = np.ones((output_height, output_width, 1), dtype=np.float32)
        else:
            if use_gpu_assemble:
                canvas = torch.zeros((output_height, output_width, 3), dtype=torch.float32, device=assemble_torch_device)
                weights = torch.zeros((output_height, output_width, 1), dtype=torch.float32, device=assemble_torch_device)
            else:
                canvas = np.zeros((output_height, output_width, 3), dtype=np.float32)
                weights = np.zeros((output_height, output_width, 1), dtype=np.float32)

        small_tile_on_top = bool(small_tile_on_top)
        use_occlusion = small_tile_on_top or any(
            float(tile.get("occlusion_priority", 0.0)) != 0.0 or int(tile.get("layer", 0)) != 0
            for tile in tiles_info
        )
        if use_gpu_assemble:
            priority_ranks = torch.full((output_height, output_width, 1), -float("inf"), dtype=torch.float32, device=assemble_torch_device)
            if base_source_image is not None and use_occlusion:
                priority_ranks.fill_(0.0)
        else:
            priority_ranks = np.full((output_height, output_width, 1), -np.inf, dtype=np.float32)
            if base_source_image is not None and use_occlusion:
                priority_ranks[:, :, :] = 0.0

        color_reference_pil = None
        if color_reference_image is not None:
            color_reference_pil = tensor2pil(color_reference_image[0].unsqueeze(0)).convert("RGB")
        elif source_image is not None:
            color_reference_pil = tensor2pil(source_image[0].unsqueeze(0)).convert("RGB")
        elif base_image is not None:
            color_reference_pil = tensor2pil(base_image[0].unsqueeze(0)).convert("RGB")
        if color_correction != "off" and color_reference_pil is None:
            raise ValueError("color_correction requires source_image, color_reference_image, or base_image as a reference.")

        target_scale = max(0.01, float(output_scale))
        large_tile_policy = str(large_tile_policy or "use_if_higher_resolution")
        large_tile_area_threshold = float(large_tile_area_threshold)
        min_tile_scale_ratio = max(0.01, float(min_tile_scale_ratio))
        context_tile_weight = _ttp_clamp(float(context_tile_weight), 0.0, 1.0)
        original_area = max(1.0, float(original_width) * float(original_height))
        has_base_canvas = base_source_image is not None
        focus_label_keywords = (
            "face", "head", "eye", "eyes", "eyelash", "eyebrow", "glasses",
            "hand", "hands", "finger", "fingers", "mouth", "lip", "lips",
            "teeth", "nose", "ear", "ears", "text", "letter", "logo",
            "detail", "focus",
        )
        alignment_stats = {
            "gpu": 0,
            "cpu": 0,
            "fallback": 0,
            "aligned": 0,
            "samples": 0,
            "candidates": 0,
            "devices": set(),
        }

        for index, tile in enumerate(tiles_info):
            tile_pil = _ttp_image_tensor_to_pil(tile_images[index])
            sx, sy, sw, sh = tile["sample_box"]

            canvas_width, canvas_height = tile.get("tile_canvas_size", [sw, sh])
            canvas_box = tile.get("tile_canvas_box", [0, 0, sw, sh])
            content_left = int(round(tile_pil.width * canvas_box[0] / max(1, canvas_width)))
            content_top = int(round(tile_pil.height * canvas_box[1] / max(1, canvas_height)))
            canvas_content_w = max(1, float(canvas_box[2]) - float(canvas_box[0]))
            canvas_content_h = max(1, float(canvas_box[3]) - float(canvas_box[1]))
            content_width = max(1, int(round(tile_pil.width * canvas_content_w / max(1, canvas_width))))
            content_height = max(1, int(round(tile_pil.height * canvas_content_h / max(1, canvas_height))))

            tile_scale_x = content_width / max(1, sw)
            tile_scale_y = content_height / max(1, sh)
            tile_effective_scale = min(tile_scale_x, tile_scale_y)
            tile_area_ratio = (float(sw) * float(sh)) / original_area
            tile_label = str(tile.get("label", tile.get("source", ""))).lower()
            is_context_label = any(word in tile_label for word in ("background", "context", "full image", "full_image", "large context"))
            is_large_tile = tile_area_ratio >= large_tile_area_threshold or is_context_label
            low_resolution_large_tile = (
                has_base_canvas
                and is_large_tile
                and tile_effective_scale < target_scale * min_tile_scale_ratio
            )
            local_left = _ttp_clamp(content_left, 0, tile_pil.width - 1)
            local_top = _ttp_clamp(content_top, 0, tile_pil.height - 1)
            local_right = _ttp_clamp(content_left + content_width, local_left + 1, tile_pil.width)
            local_bottom = _ttp_clamp(content_top + content_height, local_top + 1, tile_pil.height)

            paste_region = tile_pil.crop((local_left, local_top, local_right, local_bottom))
            out_x = int(round(sx * output_scale))
            out_y = int(round(sy * output_scale))
            out_w = max(1, int(round(sw * output_scale)))
            out_h = max(1, int(round(sh * output_scale)))
            crop_px = max(0, int(edge_crop_px))
            if crop_px > 0 and paste_region.width > crop_px * 2 and paste_region.height > crop_px * 2:
                paste_region = paste_region.crop((
                    crop_px,
                    crop_px,
                    paste_region.width - crop_px,
                    paste_region.height - crop_px,
                ))
            paste_region = _ttp_align_pil_to_aspect(paste_region, out_w, out_h, str(tile_alignment))

            if color_correction not in ("off", "local_mean_std"):
                ref = color_reference_pil
                if ref.size != (original_width, original_height):
                    ref = ref.resize((original_width, original_height), Image.Resampling.LANCZOS)
                ref_region = ref.crop((sx, sy, sx + sw, sy + sh))
                if ref_region.size != (out_w, out_h):
                    ref_region = ref_region.resize((out_w, out_h), Image.Resampling.LANCZOS)
                corrected = _ttp_apply_color_transfer(
                    pil2tensor(paste_region),
                    pil2tensor(ref_region),
                    color_correction,
                    float(color_strength),
                )
                paste_region = tensor2pil(corrected[0].unsqueeze(0)).convert("RGB")

            blend = int(round(tile.get("blend", 0) * output_scale * blend_multiplier))
            mask = _ttp_create_sample_blend_mask(
                out_w,
                out_h,
                tile.get("overlap_edges_px_source", {}),
                blend,
                output_scale,
                output_scale,
            )
            object_mask = _ttp_tile_object_mask_array(tile, out_w, out_h, str(mask_blend_mode), blend)
            if object_mask is not None:
                if str(mask_blend_mode) == "mask_only":
                    mask = object_mask
                else:
                    mask = mask * object_mask
            coverage_mask = np.clip(mask.copy(), 0.0, 1.0)
            importance = max(0.0, float(tile.get("importance", 1.0)))
            priority = max(0.0, float(tile.get("priority", 50.0)))
            priority_scale = (1.0 + priority / 100.0) if use_priority else 1.0
            mask = mask * importance * priority_scale
            rank_occlusion = float(tile.get("occlusion_priority", 0.0))
            rank_layer = float(tile.get("layer", 0.0))
            rank_priority = priority
            if small_tile_on_top and not is_large_tile:
                small_tile_rank = 10000.0 + (1.0 - _ttp_clamp(tile_area_ratio, 0.0, 1.0)) * 9000.0
                if any(keyword in tile_label for keyword in focus_label_keywords):
                    small_tile_rank += 1000.0
                rank_occlusion = max(rank_occlusion, small_tile_rank)
                rank_layer = max(rank_layer, 1.0)
            if is_large_tile and large_tile_policy == "context_only" and has_base_canvas:
                mask = mask * context_tile_weight
                rank_occlusion = min(rank_occlusion, 0.0)
                rank_layer = min(rank_layer, 0.0)
                rank_priority = 0.0
            elif low_resolution_large_tile and large_tile_policy == "use_if_higher_resolution":
                mask = mask * context_tile_weight
                rank_occlusion = min(rank_occlusion, 0.0)
                rank_layer = min(rank_layer, 0.0)
                rank_priority = 0.0

            region = np.array(paste_region).astype(np.float32) / 255.0
            if str(pixel_alignment) != "off":
                align_info = None
                if use_gpu_assemble and str(pixel_alignment_device or "auto").lower() != "cpu":
                    reference_t = canvas / torch.clamp(weights, min=1e-6)
                    gpu_align = _ttp_find_pixel_alignment_offset_torch_canvas(
                        region,
                        reference_t,
                        weights,
                        out_x,
                        out_y,
                        coverage_mask,
                        int(pixel_alignment_radius),
                        str(pixel_alignment),
                    )
                    if gpu_align is not None:
                        offset, align_info = gpu_align
                        dx, dy = int(offset[0]), int(offset[1])
                    else:
                        reference = reference_t.detach().cpu().numpy()
                        weights_np = weights.detach().cpu().numpy()
                        dx, dy, align_info = _ttp_find_pixel_alignment_offset_auto(
                            region,
                            reference,
                            weights_np,
                            out_x,
                            out_y,
                            coverage_mask,
                            int(pixel_alignment_radius),
                            str(pixel_alignment),
                            "cpu",
                        )
                        align_info = {**align_info, "fallback": True, "fallback_reason": "torch_canvas_failed"}
                elif use_gpu_assemble:
                    reference = (canvas / torch.clamp(weights, min=1e-6)).detach().cpu().numpy()
                    weights_np = weights.detach().cpu().numpy()
                    dx, dy, align_info = _ttp_find_pixel_alignment_offset_auto(
                        region,
                        reference,
                        weights_np,
                        out_x,
                        out_y,
                        coverage_mask,
                        int(pixel_alignment_radius),
                        str(pixel_alignment),
                        "cpu",
                    )
                else:
                    reference = canvas / np.maximum(weights, 1e-6)
                    dx, dy, align_info = _ttp_find_pixel_alignment_offset_auto(
                        region,
                        reference,
                        weights,
                        out_x,
                        out_y,
                        coverage_mask,
                        int(pixel_alignment_radius),
                        str(pixel_alignment),
                        str(pixel_alignment_device),
                    )
                align_device = str(align_info.get("device", "cpu"))
                if align_device.startswith("cuda") or align_device.startswith("mps") or align_device not in ("cpu", "off"):
                    alignment_stats["gpu"] += 1
                    alignment_stats["devices"].add(align_device)
                elif align_device == "cpu":
                    alignment_stats["cpu"] += 1
                if bool(align_info.get("fallback", False)):
                    alignment_stats["fallback"] += 1
                alignment_stats["samples"] = max(alignment_stats["samples"], int(align_info.get("samples", 0) or 0))
                alignment_stats["candidates"] = max(alignment_stats["candidates"], int(align_info.get("candidates", 0) or 0))
                if dx or dy:
                    alignment_stats["aligned"] += 1
                out_x += dx
                out_y += dy
            paste_x = max(0, out_x)
            paste_y = max(0, out_y)
            x_end = min(output_width, out_x + out_w)
            y_end = min(output_height, out_y + out_h)
            region_left = paste_x - out_x
            region_top = paste_y - out_y
            region_w = x_end - paste_x
            region_h = y_end - paste_y
            if region_w <= 0 or region_h <= 0:
                continue
            region = region[region_top:region_top + region_h, region_left:region_left + region_w]
            mask = mask[region_top:region_top + region_h, region_left:region_left + region_w]
            coverage_mask = coverage_mask[region_top:region_top + region_h, region_left:region_left + region_w]
            if str(color_correction) == "local_mean_std":
                ref = color_reference_pil
                if ref.size != (output_width, output_height):
                    ref = ref.resize((output_width, output_height), Image.Resampling.LANCZOS)
                ref_region = np.array(ref.crop((paste_x, paste_y, x_end, y_end))).astype(np.float32) / 255.0
                if use_gpu_assemble:
                    region_t = torch.as_tensor(region, dtype=torch.float32, device=assemble_torch_device)
                    ref_region_t = torch.as_tensor(ref_region, dtype=torch.float32, device=assemble_torch_device)
                    coverage_t = torch.as_tensor(coverage_mask, dtype=torch.float32, device=assemble_torch_device)
                    region_t = _ttp_apply_local_mean_std_color_torch(region_t, ref_region_t, coverage_t, float(color_strength))
                    region = None
                else:
                    region = _ttp_apply_local_mean_std_color(region, ref_region, coverage_mask, float(color_strength))
            if use_gpu_assemble:
                if str(color_correction) != "local_mean_std":
                    region_t = torch.as_tensor(region, dtype=torch.float32, device=assemble_torch_device)
                    coverage_t = torch.as_tensor(coverage_mask, dtype=torch.float32, device=assemble_torch_device)
                mask_t = torch.as_tensor(mask, dtype=torch.float32, device=assemble_torch_device)
                if use_occlusion:
                    rank = rank_occlusion * 1000000.0 + rank_layer * 10000.0 + (rank_priority if use_priority else 0.0)
                    rank_view = priority_ranks[paste_y:y_end, paste_x:x_end]
                    mask_active = coverage_t > 1e-6
                    higher = (rank > rank_view + 1e-6) & mask_active
                    eligible = (rank >= rank_view - 1e-6) & mask_active
                    canvas_view = canvas[paste_y:y_end, paste_x:x_end]
                    weights_view = weights[paste_y:y_end, paste_x:x_end]
                    higher_alpha = coverage_t * higher.to(dtype=torch.float32)
                    keep = 1.0 - higher_alpha
                    canvas_view.mul_(keep)
                    weights_view.mul_(keep)
                    rank_view[higher] = rank
                    eligible_f = eligible.to(dtype=torch.float32)
                    canvas_view.add_(region_t * mask_t * eligible_f)
                    weights_view.add_(mask_t * eligible_f)
                else:
                    canvas[paste_y:y_end, paste_x:x_end].add_(region_t * mask_t)
                    weights[paste_y:y_end, paste_x:x_end].add_(mask_t)
            elif use_occlusion:
                rank = rank_occlusion * 1000000.0 + rank_layer * 10000.0 + (rank_priority if use_priority else 0.0)
                rank_view = priority_ranks[paste_y:y_end, paste_x:x_end]
                mask_active = coverage_mask > 1e-6
                higher = (rank > rank_view + 1e-6) & mask_active
                eligible = (rank >= rank_view - 1e-6) & mask_active
                canvas_view = canvas[paste_y:y_end, paste_x:x_end]
                weights_view = weights[paste_y:y_end, paste_x:x_end]
                if np.any(higher):
                    higher_alpha = coverage_mask * higher.astype(np.float32)
                    keep = 1.0 - higher_alpha
                    canvas_view *= keep
                    weights_view *= keep
                    rank_view[higher[:, :, 0], :] = rank
                canvas_view += region * mask * eligible
                weights_view += mask * eligible
            else:
                canvas[paste_y:y_end, paste_x:x_end] += region * mask
                weights[paste_y:y_end, paste_x:x_end] += mask

        if str(pixel_alignment) != "off":
            devices = ",".join(sorted(alignment_stats["devices"])) or ("cpu" if alignment_stats["cpu"] else "none")
            print(
                "[TTP Smart Tile] pixel alignment "
                f"mode={pixel_alignment} requested_device={pixel_alignment_device} "
                f"actual_gpu_tiles={alignment_stats['gpu']} actual_cpu_tiles={alignment_stats['cpu']} "
                f"fallback_tiles={alignment_stats['fallback']} moved_tiles={alignment_stats['aligned']} "
                f"radius={int(pixel_alignment_radius)} max_samples={alignment_stats['samples']} "
                f"candidates={alignment_stats['candidates']} devices={devices}"
            )

        actual_assemble_device = str(assemble_torch_device) if use_gpu_assemble else "cpu"
        if assemble_device_mode != "cpu" or assemble_fallback_reason:
            print(
                "[TTP Smart Tile] assemble paste "
                f"requested_device={assemble_device_mode} actual_device={actual_assemble_device} "
                f"mode={effective_assemble_mode} requested_mode={requested_assemble_mode} tiles={len(tiles_info)} "
                f"fallback={assemble_fallback_reason or 'none'}"
            )

        if use_gpu_assemble:
            safe_weights = torch.clamp(weights, min=1e-6)
            output = torch.clamp(canvas / safe_weights, 0.0, 1.0).detach().cpu()
            max_weight = torch.clamp(weights.max(), min=1e-6)
            weight_preview = torch.clamp(weights / max_weight, 0.0, 1.0).repeat(1, 1, 3).detach().cpu()
            return (output.unsqueeze(0), weight_preview.unsqueeze(0))

        safe_weights = np.maximum(weights, 1e-6)
        output = canvas / safe_weights
        output = np.clip(output, 0.0, 1.0)
        weight_preview = np.clip(weights / max(1e-6, weights.max()), 0.0, 1.0)
        weight_preview = np.repeat(weight_preview, 3, axis=2)

        return (
            torch.from_numpy(output.astype(np.float32)).unsqueeze(0),
            torch.from_numpy(weight_preview.astype(np.float32)).unsqueeze(0),
        )


class TTP_Smart_Tile_Save_Final_Image_Experimental:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "done": ("BOOLEAN", {"default": True}),
                "filename_prefix": ("STRING", {"default": "TTP_Smart_Tile"}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_final_image"
    OUTPUT_NODE = True
    CATEGORY = "TTP/Smart Tile"

    def save_final_image(self, images, done=True, filename_prefix="TTP_Smart_Tile", prompt=None, extra_pnginfo=None):
        if not bool(done):
            return {"ui": {"images": []}}

        if images is None or int(images.shape[0]) <= 0:
            raise ValueError("images must contain at least one image.")

        image = images[-1]
        full_output_folder, filename, counter, subfolder, _filename_prefix = folder_paths.get_save_image_path(
            str(filename_prefix or "TTP_Smart_Tile"),
            self.output_dir,
            image.shape[1],
            image.shape[0],
        )
        i = 255.0 * image.cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

        metadata = None
        if not args.disable_metadata:
            metadata = PngInfo()
            if prompt is not None:
                metadata.add_text("prompt", json.dumps(prompt))
            if extra_pnginfo is not None:
                for key in extra_pnginfo:
                    metadata.add_text(key, json.dumps(extra_pnginfo[key]))

        file = f"{filename}_{counter:05}_.png"
        img.save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=self.compress_level)
        return {
            "ui": {
                "images": [{
                    "filename": file,
                    "subfolder": subfolder,
                    "type": self.type,
                }]
            }
        }


class TTP_CoordinateSplitter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Positions": ("LIST", {"forceInput": True}),
            }
        }
        
    RETURN_TYPES = ("LIST",)
    RETURN_NAMES = ("Coordinates",)
    FUNCTION = "split_coordinates"

    CATEGORY = "TTP/Conditioning"
    
    def split_coordinates(self, Positions):
        coordinates = []
        for i, coords in enumerate(Positions):
            if len(coords) != 4:
                raise ValueError(f"Coordinate group {i+1} must contain exactly 4 values, but got {len(coords)}")
            
            x, y, x2, y2 = coords
            width = x2 - x
            height = y2 - y
            coordinates.append((x, y, width, height))  # Create a tuple for each coordinate group
        
        return (coordinates,)  # Return as a tuple containing a list of tuples



class TTP_condtobatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditionings": ("CONDITIONING", {"forceInput": True}),
            }
        }

    INPUT_IS_LIST = True

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "combine_to_batch"

    CATEGORY = "TTP/Conditioning"

    def combine_to_batch(self, conditionings):
        # 直接将所有conditioning组合在一起并返回
        combined_conditioning = sum(conditionings, [])
        return (combined_conditioning,)
        
        
class TTP_condsetarea_merge:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning_batch": ("CONDITIONING", {"forceInput": True}),
                "coordinates": ("LIST", {"forceInput": True}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "apply_coordinates_to_batch"

    CATEGORY = "TTP/Conditioning"

    def apply_coordinates_to_batch(self, conditioning_batch, coordinates, strength):
        # 确保coordinates和conditioning_batch的数量一致
        if len(coordinates) != len(conditioning_batch):
            raise ValueError(f"The number of coordinates ({len(coordinates)}) does not match the number of conditionings ({len(conditioning_batch)})")

        updated_conditionings = []

        # 遍历每个conditioning和相应的coordinate
        for conditioning, coord in zip(conditioning_batch, coordinates):
            if len(coord) != 4:
                raise ValueError(f"Each coordinate should have exactly 4 values, but got {len(coord)}")

            x, y, width, height = coord

            # Print x, y, width, height for debugging
            print(f"Processing coordinate - x: {x}, y: {y}, width: {width}, height: {height}")

            # 将每个 conditioning 处理为列表格式
            single_conditioning = [conditioning]

            # 使用标准的 node_helpers.conditioning_set_values 方法进行区域设置
            updated_conditioning = node_helpers.conditioning_set_values(
                single_conditioning,
                {
                    "area": (height // 8, width // 8, y // 8, x // 8),
                    "strength": strength,
                    "set_area_to_bounds": False,
                }
            )

            updated_conditionings.append(updated_conditioning)

        # 将所有更新后的conditioning重新组合为一个batch
        combined_conditioning = sum(updated_conditionings, [])
        return (combined_conditioning,)

class TTP_condsetarea_merge_test:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning_batch": ("CONDITIONING", {"forceInput": True}),
                "coordinates": ("LIST", {"forceInput": True}),
                "group_size": ("INT", {"default": 1, "min": 1, "step": 1}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "apply_coordinates_to_batch"

    CATEGORY = "TTP/Conditioning"

    def apply_coordinates_to_batch(self, conditioning_batch, coordinates, group_size, strength):
        import math

        # 计算 conditioning 中的组数
        num_conditionings = len(conditioning_batch)
        num_groups = math.ceil(num_conditionings / group_size)

        # 如果坐标数量大于组数，需要复制 conditioning
        if len(coordinates) > num_groups:
            # 计算需要的倍数
            multiplier = math.ceil(len(coordinates) * group_size / num_conditionings)
            # 复制 conditioning_batch
            conditioning_batch = conditioning_batch * multiplier
            num_conditionings = len(conditioning_batch)
            num_groups = math.ceil(num_conditionings / group_size)

        # 重新计算需要的坐标数量
        required_coords = num_groups

        # 检查坐标数量是否足够
        if len(coordinates) != required_coords:
            raise ValueError(f"The number of coordinates ({len(coordinates)}) does not match the required number ({required_coords}) based on group size ({group_size}) and conditioning length ({num_conditionings})")

        updated_conditionings = []
        conditioning_index = 0

        # 遍历坐标和分组
        for coord in coordinates:
            if len(coord) != 4:
                raise ValueError(f"Each coordinate should have exactly 4 values, but got {len(coord)}")

            x, y, width, height = coord

            # 打印调试信息
            print(f"Processing coordinate - x: {x}, y: {y}, width: {width}, height: {height}")

            # 获取当前组的 conditioning
            group_conditionings = conditioning_batch[conditioning_index:conditioning_index + group_size]

            for conditioning in group_conditionings:
                # 使用标准的 node_helpers.conditioning_set_values 方法进行区域设置
                updated_conditioning = node_helpers.conditioning_set_values(
                    [conditioning],
                    {
                        "area": (height // 8, width // 8, y // 8, x // 8),
                        "strength": strength,
                        "set_area_to_bounds": False,
                    }
                )

                updated_conditionings.append(updated_conditioning)

            conditioning_index += group_size

        # 将所有更新后的 conditioning 重新组合为一个批次
        combined_conditioning = sum(updated_conditionings, [])
        return (combined_conditioning,)

        
class Tile_imageSize:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "width_factor": ("INT", {"default": 3, "min": 1, "max": 10, "step": 1}),
                "height_factor": ("INT", {"default": 3, "min": 1, "max": 10, "step": 1}),
                "overlap_rate": ("FLOAT", {"default": 0.1, "min": 0.00, "max": 0.95, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("tile_width", "tile_height")
    CATEGORY = "TTP/Image"
    FUNCTION = "image_width_height"

    def image_width_height(self, image, width_factor, height_factor, overlap_rate):
        _, raw_H, raw_W, _ = image.shape
        if overlap_rate == 0:
            # 水平方向
            if width_factor == 1:
                tile_width = raw_W
            else:
                tile_width = int(raw_W / width_factor)
                if tile_width % 8 != 0:
                    tile_width = ((tile_width + 7) // 8) * 8
            # 垂直方向
            if height_factor == 1:
                tile_height = raw_H
            else:
                tile_height = int(raw_H / height_factor)
                if tile_height % 8 != 0:
                    tile_height = ((tile_height + 7) // 8) * 8

        else:
            # 水平方向
            if width_factor == 1:
                tile_width = raw_W
            else:
                tile_width = int(raw_W / (1 + (width_factor - 1) * (1 - overlap_rate)))
                if tile_width % 8 != 0:
                    tile_width = (tile_width // 8) * 8
            # 垂直方向
            if height_factor == 1:
                tile_height = raw_H
            else:
                tile_height = int(raw_H / (1 + (height_factor - 1) * (1 - overlap_rate)))
                if tile_height % 8 != 0:
                    tile_height = (tile_height // 8) * 8

        return (tile_width, tile_height)
        
class TTP_Expand_And_Mask:
    """
    这是一个节点类，用于将输入图片在指定方向扩展一定数量的块并创建相应蒙版。

    功能：
    1. 支持同时在多个方向上扩展图像。
    2. 分别控制每个方向的扩展块数量。
    3. 将输入图像的透明通道（Alpha 通道）信息转换为蒙版，并与新创建的蒙版合并。
    4. 添加一个布尔参数 fill_alpha_decision 来决定是否将输出图片中的透明区域填充为指定颜色，并输出 RGB 图像。
    """
    def __init__(self, *args, **kwargs):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        directions = ["left", "right", "top", "bottom"]
        return {
            "required": {
                "image": ("IMAGE",),  # 输入一张图片
                "fill_mode": (["duplicate", "white"], {"default": "duplicate", "label": "Fill Mode"}),
                "fill_alpha_decision": ("BOOLEAN", {"default": False, "label": "Fill Alpha with Color"}),
                "fill_color": ("STRING", {"default": "#7F7F7F", "label": "Fill Color"}),
            },
            "optional": {
                **{f"expand_{dir}": ("BOOLEAN", {"default": False, "label": f"Expand {dir.capitalize()}"}) for dir in directions},
                **{f"num_blocks_{dir}": ("INT", {"default": 1, "min": 0, "max": 3, "step": 1, "label": f"Blocks {dir.capitalize()}"}) for dir in directions},
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("EXPANDED_IMAGE", "MASK")
    FUNCTION = "expand_and_mask"
    CATEGORY = "TTP/Image"

    def hex_to_rgba(self, hex_color):
        # 去除可能存在的 '#' 字符
        hex_color = hex_color.lstrip('#')
        # 如果为6位十六进制字符串，默认为不透明
        if len(hex_color) == 6:
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            return (r, g, b, 255)
        # 如果为8位，则最后两位为透明度
        elif len(hex_color) == 8:
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            a = int(hex_color[6:8], 16)
            return (r, g, b, a)
        else:
            raise ValueError("Invalid hex color format")

    def expand_and_mask(self, image, fill_mode="duplicate", fill_alpha_decision=False, fill_color="#7F7F7F", **kwargs):
        pil_image = tensor2pil(image)
        orig_width, orig_height = pil_image.size
        has_alpha = (pil_image.mode == 'RGBA')

        # 解析方向和块数
        directions = ["left", "right", "top", "bottom"]
        expand_directions = {dir: kwargs.get(f"expand_{dir}", False) for dir in directions}
        num_blocks = {dir: kwargs.get(f"num_blocks_{dir}", 0) if expand_directions[dir] else 0 for dir in directions}

        # 计算扩展后的尺寸
        total_width = orig_width + orig_width * (num_blocks["left"] + num_blocks["right"])
        total_height = orig_height + orig_height * (num_blocks["top"] + num_blocks["bottom"])

        # 创建扩展后的图像
        expanded_image_mode = pil_image.mode
        expanded_image = Image.new(expanded_image_mode, (total_width, total_height))

        # 根据 fill_mode 创建填充图像
        def create_fill_image():
            if pil_image.mode == 'RGBA':
                return Image.new("RGBA", (orig_width, orig_height), color=(255, 255, 255, 255))
            elif pil_image.mode == 'RGB':
                return Image.new("RGB", (orig_width, orig_height), color=(255, 255, 255))
            elif pil_image.mode == 'L':
                return Image.new("L", (orig_width, orig_height), color=255)
            else:
                raise ValueError(f"Unsupported image mode for fill: {pil_image.mode}")

        if fill_mode == "duplicate":
            fill_image = pil_image.copy()
        elif fill_mode == "white":
            fill_image = create_fill_image()
        else:
            fill_image = pil_image.copy()

        # 计算原图在扩展图像中的位置
        left_offset = orig_width * num_blocks["left"]
        top_offset = orig_height * num_blocks["top"]

        # 粘贴原始图像
        expanded_image.paste(pil_image, (left_offset, top_offset))

        # 粘贴填充区域
        for dir in directions:
            blocks = num_blocks[dir]
            for i in range(blocks):
                if dir == "left":
                    x = left_offset - orig_width * (i + 1)
                    y = top_offset
                elif dir == "right":
                    x = left_offset + orig_width * (i + 1)
                    y = top_offset
                elif dir == "top":
                    x = left_offset
                    y = top_offset - orig_height * (i + 1)
                elif dir == "bottom":
                    x = left_offset
                    y = top_offset + orig_height * (i + 1)
                else:
                    continue
                expanded_image.paste(fill_image, (x, y))

        # 粘贴角落填充区域（处理同时选择多个方向的情况）
        corner_positions = []
        if expand_directions["left"] and expand_directions["top"]:
            for i in range(num_blocks["left"]):
                for j in range(num_blocks["top"]):
                    x = left_offset - orig_width * (i + 1)
                    y = top_offset - orig_height * (j + 1)
                    corner_positions.append((x, y))
        if expand_directions["left"] and expand_directions["bottom"]:
            for i in range(num_blocks["left"]):
                for j in range(num_blocks["bottom"]):
                    x = left_offset - orig_width * (i + 1)
                    y = top_offset + orig_height * (j + 1)
                    corner_positions.append((x, y))
        if expand_directions["right"] and expand_directions["top"]:
            for i in range(num_blocks["right"]):
                for j in range(num_blocks["top"]):
                    x = left_offset + orig_width * (i + 1)
                    y = top_offset - orig_height * (j + 1)
                    corner_positions.append((x, y))
        if expand_directions["right"] and expand_directions["bottom"]:
            for i in range(num_blocks["right"]):
                for j in range(num_blocks["bottom"]):
                    x = left_offset + orig_width * (i + 1)
                    y = top_offset + orig_height * (j + 1)
                    corner_positions.append((x, y))

        for pos in corner_positions:
            expanded_image.paste(fill_image, pos)

        # 创建蒙版
        mask_array = np.zeros((total_height, total_width), dtype=np.float32)

        # 原始图像区域蒙版处理
        if has_alpha:
            alpha_array = np.array(pil_image.getchannel("A"), dtype=np.float32) / 255.0
            alpha_mask_array = 1.0 - alpha_array
            mask_array[top_offset:top_offset + orig_height, left_offset:left_offset + orig_width] = alpha_mask_array

        # 填充区域蒙版设置为1.0
        # 左右扩展区域
        for dir in ["left", "right"]:
            blocks = num_blocks[dir]
            for i in range(blocks):
                if dir == "left":
                    x_start = left_offset - orig_width * (i + 1)
                    x_end = left_offset - orig_width * i
                elif dir == "right":
                    x_start = left_offset + orig_width * (i + 1)
                    x_end = left_offset + orig_width * (i + 2)
                else:
                    continue
                mask_array[top_offset:top_offset + orig_height, x_start:x_end] = 1.0

        # 上下扩展区域
        for dir in ["top", "bottom"]:
            blocks = num_blocks[dir]
            for i in range(blocks):
                if dir == "top":
                    y_start = top_offset - orig_height * (i + 1)
                    y_end = top_offset - orig_height * i
                elif dir == "bottom":
                    y_start = top_offset + orig_height * (i + 1)
                    y_end = top_offset + orig_height * (i + 2)
                else:
                    continue
                mask_array[y_start:y_end, left_offset:left_offset + orig_width] = 1.0

        # 角落区域蒙版设置为1.0
        for pos in corner_positions:
            x, y = pos
            mask_array[y:y + orig_height, x:x + orig_width] = 1.0

        # 创建蒙版张量 (1, 1, height, width)
        mask_tensor = torch.from_numpy(mask_array).unsqueeze(0).unsqueeze(0)

        # 根据 fill_alpha_decision 参数决定是否将输出图像中的透明区域填充为指定颜色
        if fill_alpha_decision and has_alpha:
            expanded_image = expanded_image.convert('RGBA')  # 确保图像是RGBA模式
            # 使用自定义填充颜色代替纯白色
            fill_rgba = self.hex_to_rgba(fill_color)
            background = Image.new('RGBA', expanded_image.size, fill_rgba)
            expanded_image = Image.alpha_composite(background, expanded_image)
            expanded_image = expanded_image.convert('RGB')  # 转换为RGB模式
            expanded_image_mode = 'RGB'

        expanded_image_tensor = pil2tensor(expanded_image)

        return (expanded_image_tensor, mask_tensor)
        
class TTP_text_mix:
    def __init__(self, *args, **kwargs):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text1": ("STRING", {"default": "", "multiline": True, "label": "Text Box 1"}),
                "text2": ("STRING", {"default": "", "multiline": True, "label": "Text Box 2"}),
                "text3": ("STRING", {"default": "", "multiline": True, "label": "Text Box 3"}),
                "template": ("STRING", {"default": "", "multiline": True, "label": "Template Text Box"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("text1", "text2", "text3", "final_text")
    FUNCTION = "mix_texts"
    CATEGORY = "TTP/text"

    def mix_texts(self, text1, text2, text3, template):
        # 使用replace方法替换模板中的占位符{text1}和{text2}
        final_text = template.replace("{text1}", text1).replace("{text2}", text2).replace("{text3}", text3)

        return (text1, text2, text3, final_text)

def horner_poly(x: torch.Tensor, coefficients: torch.Tensor) -> torch.Tensor:
    """
    使用 Horner's scheme 计算多项式:
      c[0]*x^(n-1) + c[1]*x^(n-2) + ... + c[n-2]*x + c[n-1]
    其中 coefficients = [c[0], c[1], ..., c[n-1]].
    """
    out = torch.zeros_like(x)
    for c in coefficients:
        out = out * x + c
    return out

def modulate(x, shift, scale):
    """Modulate layer implementation for HunyuanVideo"""
    try:
        # Ensure consistent data types
        shift = shift.to(dtype=x.dtype, device=x.device)
        scale = scale.to(dtype=x.dtype, device=x.device)
        
        # Reshape shift and scale to match x dimensions
        B = x.shape[0]  # batch size
        
        if len(x.shape) == 3:  # [B, L, D]
            shift = shift.view(B, 1, -1)  # [B, 1, D]
            scale = scale.view(B, 1, -1)  # [B, 1, D]
            shift = shift.expand(-1, x.shape[1], -1)  # [B, L, D]
            scale = scale.expand(-1, x.shape[1], -1)  # [B, L, D]
        elif len(x.shape) == 5:  # [B, C, T, H, W]
            shift = shift.view(B, -1, 1, 1, 1)  # [B, C, 1, 1, 1]
            scale = scale.view(B, -1, 1, 1, 1)  # [B, C, 1, 1, 1]
            shift = shift.expand(-1, -1, x.shape[2], x.shape[3], x.shape[4])  # [B, C, T, H, W]
            scale = scale.expand(-1, -1, x.shape[2], x.shape[3], x.shape[4])  # [B, C, T, H, W]
        else:
            raise ValueError(f"Unsupported input shape: {x.shape}")
        
        # Step-by-step calculation to reduce memory usage
        result = x.mul_(1 + scale)  # in-place operation
        result.add_(shift)  # in-place operation
        
        return result
        
    except Exception as e:
        raise RuntimeError(f"Modulation failed: {str(e)}")

def modulate(x, shift, scale):
    """Modulate layer implementation for HunyuanVideo"""
    try:
        # Ensure consistent data types
        shift = shift.to(dtype=x.dtype, device=x.device)
        scale = scale.to(dtype=x.dtype, device=x.device)
        
        # Reshape shift and scale to match x dimensions
        B = x.shape[0]  # batch size
        
        if len(x.shape) == 3:  # [B, L, D]
            shift = shift.view(B, 1, -1)  # [B, 1, D]
            scale = scale.view(B, 1, -1)  # [B, 1, D]
            shift = shift.expand(-1, x.shape[1], -1)  # [B, L, D]
            scale = scale.expand(-1, x.shape[1], -1)  # [B, L, D]
        elif len(x.shape) == 5:  # [B, C, T, H, W]
            shift = shift.view(B, -1, 1, 1, 1)  # [B, C, 1, 1, 1]
            scale = scale.view(B, -1, 1, 1, 1)  # [B, C, 1, 1, 1]
            shift = shift.expand(-1, -1, x.shape[2], x.shape[3], x.shape[4])  # [B, C, T, H, W]
            scale = scale.expand(-1, -1, x.shape[2], x.shape[3], x.shape[4])  # [B, C, T, H, W]
        else:
            raise ValueError(f"Unsupported input shape: {x.shape}")
        
        # Step-by-step calculation to reduce memory usage
        result = x.mul_(1 + scale)  # in-place operation
        result.add_(shift)  # in-place operation
        
        return result
        
    except Exception as e:
        raise RuntimeError(f"Modulation failed: {str(e)}")

class TeaCacheHunyuanVideoSampler:
    @classmethod 
    def INPUT_TYPES(cls):
        return {
            "required": {
                "noise": ("NOISE",),
                "guider": ("GUIDER",),
                "sampler": ("SAMPLER",),
                "sigmas": ("SIGMAS",),
                "latent_image": ("LATENT",),
                "speedup": ([
                    "Original (1x)", 
                    "Fast (1.6x)", 
                    "Faster (2.1x)",
                    "Ultra Fast (3.2x)",
                    "Shapeless Fast (4.4x)"
                ], {
                    "default": "Fast (1.6x)",
                    "tooltip": (
                        "Control TeaCache speed/quality trade-off:\n"
                        "Original: Base quality\n"
                        "Fast: 1.6x speedup\n"
                        "Faster: 2.1x speedup\n"
                        "Ultra Fast: 3.2x speedup\n"
                        "Shapeless Fast: 4.4x speedup"
                    )
                }),
                "enable_custom_speed": ("BOOLEAN", {
                    "default": False,
                    "label": "Enable Custom Speed"
                }),
                "custom_speed": ("FLOAT", {
                    "default": 1.0,
                    "min": 1.0,
                    "max": 4.4,
                    "step": 0.1,
                    "label": "Custom Speed Multiplier"
                })
            }
        }

    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("output", "denoised_output")
    FUNCTION = "sample"
    CATEGORY = "sampling/custom_sampling"

    def calculate_threshold(self, speed_multiplier: float) -> float:
        """根据预设速度点进行线性插值，计算自定义速度对应的阈值"""
        # 预设的速度倍数和对应阈值
        predefined_speeds = [1.0, 1.6, 2.1, 3.2, 4.4]
        predefined_thresholds = [0.0, 0.1, 0.15, 0.25, 0.35]
        
        # 使用 numpy 的线性插值函数
        threshold = np.interp(speed_multiplier, predefined_speeds, predefined_thresholds)
        
        # 确保阈值不超过最大值
        threshold = min(threshold, 0.35)
        
        return threshold

    def teacache_forward(
            self,
            transformer,
            x: torch.Tensor,
            timestep: torch.Tensor,  # Should be in range(0, 1000).
            context: Optional[torch.Tensor] = None,
            y: Optional[torch.Tensor] = None,  # Text embedding for modulation.
            guidance: Optional[torch.Tensor] = None,  # Guidance for modulation, should be cfg_scale x 1000.
            attention_mask: Optional[torch.Tensor] = None,
            control: Any = None,
            transformer_options: Dict = {},
            **kwargs
        ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """TeaCache forward implementation"""
        should_calc = True
        
        if transformer.enable_teacache:
            try:
                # 获取输入维度
                B, C, T, H, W = x.shape
                
                # 准备调制向量
                try:
                    # HunyuanVideo 使用 timestep_embedding 进行时间步编码
                    time_emb = comfy.ldm.flux.layers.timestep_embedding(timestep, 256, time_factor=1.0).to(x.dtype)
                    vec = transformer.time_in(time_emb)  # [B, hidden_size]
                    
                    # 文本调制 - HunyuanVideo 使用 vector_in 处理 y 而不是 context
                    if y is not None:
                        if not hasattr(transformer, 'params') or not hasattr(transformer.params, 'vec_in_dim'):
                            raise AttributeError("Transformer missing required attributes: params.vec_in_dim")
                        vec = vec + transformer.vector_in(y[:, :transformer.params.vec_in_dim])
                    
                    # 指导调制
                    if guidance is not None and getattr(transformer, 'params', None) and transformer.params.guidance_embed:
                        guidance_emb = comfy.ldm.flux.layers.timestep_embedding(guidance, 256).to(x.dtype)
                        guidance_vec = transformer.guidance_in(guidance_emb)
                        vec = vec + guidance_vec
                        
                except Exception as e:
                    raise RuntimeError(f"Failed to prepare modulation vector: {str(e)}")
                
                # 嵌入图像
                try:
                    img = transformer.img_in(x)
                except Exception as e:
                    raise RuntimeError(f"Failed to embed image: {str(e)}")
                
                if transformer.enable_teacache:
                    try:
                        # 使用原地操作减少内存使用
                        inp = img.clone()
                        vec_ = vec.clone()
                        
                        # 获取调制参数
                        modulation_output = transformer.double_blocks[0].img_mod(vec_)
                        
                        # 处理调制输出
                        if isinstance(modulation_output, tuple):
                            if len(modulation_output) >= 2:
                                mod_shift = modulation_output[0]
                                mod_scale = modulation_output[1]
                                if hasattr(mod_shift, 'shift') and hasattr(mod_scale, 'scale'):
                                    img_mod1_shift = mod_shift.shift
                                    img_mod1_scale = mod_scale.scale
                                else:
                                    img_mod1_shift = mod_shift
                                    img_mod1_scale = mod_scale
                            else:
                                raise ValueError(f"Tuple too short, expected at least 2 elements, got {len(modulation_output)}")
                        elif hasattr(modulation_output, 'shift') and hasattr(modulation_output, 'scale'):
                            img_mod1_shift = modulation_output.shift
                            img_mod1_scale = modulation_output.scale
                        elif hasattr(modulation_output, 'chunk'):
                            chunks = modulation_output.chunk(6, dim=-1)
                            img_mod1_shift = chunks[0]
                            img_mod1_scale = chunks[1]
                        else:
                            raise ValueError(f"Unsupported modulation output format: {type(modulation_output)}")
                        
                        # 确保获取到的是张量
                        if not isinstance(img_mod1_shift, torch.Tensor) or not isinstance(img_mod1_scale, torch.Tensor):
                            raise ValueError(f"Failed to get tensor values for shift and scale")
                        
                        # 应用归一化和调制
                        normed_inp = transformer.double_blocks[0].img_norm1(inp)
                        del inp  # 释放内存
                        
                        modulated_inp = modulate(normed_inp, shift=img_mod1_shift, scale=img_mod1_scale)
                        del normed_inp  # 释放内存
                        
                        # 计算相对 L1 距离并决定是否需要计算
                        if transformer.cnt == 0 or transformer.cnt == transformer.num_steps - 1:
                            should_calc = True
                            transformer.accumulated_rel_l1_distance = 0
                        else:
                            try:
                                coefficients = [7.33226126e+02, -4.01131952e+02, 6.75869174e+01, -3.14987800e+00, 9.61237896e-02]
                                rescale_func = np.poly1d(coefficients)
                                rel_l1 = ((modulated_inp - transformer.previous_modulated_input).abs().mean() / 
                                         (transformer.previous_modulated_input.abs().mean() + 1e-6)).cpu().item()
                                transformer.accumulated_rel_l1_distance += rescale_func(rel_l1)
                                
                                if transformer.accumulated_rel_l1_distance < transformer.rel_l1_thresh:
                                    should_calc = False
                                else:
                                    should_calc = True
                                    transformer.accumulated_rel_l1_distance = 0
                            except Exception as e:
                                should_calc = True
                        
                        transformer.previous_modulated_input = modulated_inp
                        transformer.cnt += 1
                        
                    except Exception as e:
                        should_calc = True

            except Exception as e:
                should_calc = True

        # 如果需要计算，调用原始的 forward 方法
        if should_calc:
            try:
                out = transformer.original_forward(x, timestep, context, y, guidance, 
                                                attention_mask=attention_mask,
                                                control=control,
                                                transformer_options=transformer_options,
                                                **kwargs)
                transformer.previous_residual = out
                return out
            except Exception as e:
                raise
        else:
            # 如果不需要计算，返回之前的结果
            return transformer.previous_residual

    def sample(self, noise, guider, sampler, sigmas, latent_image, speedup, enable_custom_speed=False, custom_speed=1.0):
        """Sampling implementation"""
        device = comfy.model_management.get_torch_device()
        
        # 定义预设速度的阈值映射
        predefined_speeds = [1.0, 1.6, 2.1, 3.2, 4.4]
        predefined_thresholds = [0.0, 0.1, 0.15, 0.25, 0.35]
        
        # 根据是否启用自定义速度来决定使用哪个阈值
        if enable_custom_speed:
            if not (1.0 <= custom_speed <= 4.4):
                raise ValueError("Custom speed must be between 1.0 and 4.4")
            threshold = self.calculate_threshold(custom_speed)
        else:
            # 定义预设的速度选项
            thresh_map = {
                "Original (1x)": 0.0,
                "Fast (1.6x)": 0.1,
                "Faster (2.1x)": 0.15,
                "Ultra Fast (3.2x)": 0.25,
                "Shapeless Fast (4.4x)": 0.35
            }
            if speedup not in thresh_map:
                raise ValueError(f"Unsupported speedup option: {speedup}")
            threshold = thresh_map[speedup]
        
        try:
            # 获取 transformer
            transformer = guider.model_patcher.model.diffusion_model
            
            # 初始化 TeaCache 状态
            transformer.enable_teacache = True
            transformer.cnt = 0  
            transformer.num_steps = len(sigmas) - 1
            transformer.rel_l1_thresh = threshold
            transformer.accumulated_rel_l1_distance = 0
            transformer.previous_modulated_input = None
            transformer.previous_residual = None

            latent = latent_image
            latent_image = latent["samples"].clone()
            latent = latent.copy()

            noise_mask = None
            if "noise_mask" in latent:
                noise_mask = latent["noise_mask"].clone()

            # 保存原始 forward 方法
            transformer.original_forward = transformer.forward
            
            # 使用 lambda 替换 forward 方法，确保正确绑定 self
            transformer.forward = lambda x, t, context=None, y=None, guidance=None, attention_mask=None, control=None, transformer_options={}, **kwargs: self.teacache_forward(
                transformer, x, t, context, y, guidance, attention_mask, control, transformer_options, **kwargs
            )
            
            try:
                x0_output = {}
                callback = latent_preview.prepare_callback(guider.model_patcher, sigmas.shape[-1] - 1, x0_output)
                
                disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
                samples = guider.sample(
                    noise.generate_noise(latent), 
                    latent_image, 
                    sampler, 
                    sigmas, 
                    denoise_mask=noise_mask, 
                    callback=callback, 
                    disable_pbar=disable_pbar, 
                    seed=noise.seed
                )
                samples = samples.to(comfy.model_management.intermediate_device())
                
            finally:
                # 恢复原始的 forward 方法
                transformer.forward = transformer.original_forward
                delattr(transformer, 'original_forward')
                transformer.enable_teacache = False

            out = latent.copy()
            out["samples"] = samples
            if "x0" in x0_output:
                out_denoised = latent.copy()
                out_denoised["samples"] = guider.model_patcher.model.process_latent_out(x0_output["x0"].cpu())
            else:
                out_denoised = out
                
            return (out, out_denoised)

        except Exception as e:
            raise RuntimeError(f"Sampling failed: {str(e)}")

NODE_CLASS_MAPPINGS = {
    "TTPlanet_Tile_Preprocessor_Simple": TTPlanet_Tile_Preprocessor_Simple,
    "TTP_Image_Tile_Batch": TTP_Image_Tile_Batch,
    "TTP_Image_Assy": TTP_Image_Assy,
    "TTP_CoordinateSplitter": TTP_CoordinateSplitter,
    "TTP_condtobatch": TTP_condtobatch,
    "TTP_condsetarea_merge": TTP_condsetarea_merge,
    "TTP_Tile_image_size": Tile_imageSize,
    "TTP_condsetarea_merge_test": TTP_condsetarea_merge_test,
    "TTP_Expand_And_Mask": TTP_Expand_And_Mask,
    "TTP_text_mix": TTP_text_mix,
    "TeaCacheHunyuanVideoSampler": TeaCacheHunyuanVideoSampler,
    "TTP_Smart_Tile_Interactive_Crop_Experimental": TTP_Smart_Tile_Interactive_Crop_Experimental,
    "TTP_Smart_Tile_Set_Preview_Experimental": TTP_Smart_Tile_Set_Preview_Experimental,
    "TTP_QwenVL3_Local_Loader_Experimental": TTP_QwenVL3_Local_Loader_Experimental,
    "TTP_Smart_Tile_QwenVL_Prompt_Set_Builder_Experimental": TTP_Smart_Tile_QwenVL_Prompt_Set_Builder_Experimental,
    "TTP_Smart_Tile_Loop_Source_Experimental": TTP_Smart_Tile_Loop_Source_Experimental,
    "TTP_Smart_Tile_Loop_Collect_Experimental": TTP_Smart_Tile_Loop_Collect_Experimental,
    "TTP_Smart_Tile_Image_Upscale_Prep_Experimental": TTP_Smart_Tile_Image_Upscale_Prep_Experimental,
    "TTP_Smart_Tile_Assemble_Experimental": TTP_Smart_Tile_Assemble_Experimental,
    "TTP_Smart_Tile_Save_Final_Image_Experimental": TTP_Smart_Tile_Save_Final_Image_Experimental,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TTPlanet_Tile_Preprocessor_Simple": "TTP Tile Preprocessor Simple",
    "TTP_Image_Tile_Batch": "TTP_Image_Tile_Batch",
    "TTP_Image_Assy": "TTP_Image_Assy",
    "TTP_CoordinateSplitter": "TTP_CoordinateSplitter",
    "TTP_condtobatch": "TTP_cond to batch",
    "TTP_condsetarea_merge": "TTP_condsetarea_merge",
    "TTP_Tile_image_size": "TTP_Tile_image_size",
    "TTP_condsetarea_merge_test": "TTP_condsetarea_merge_test",
    "TTP_Expand_And_Mask": "TTP_Expand_And_Mask",
    "TTP_text_mix": "TTP_text_mix",
    "TeaCacheHunyuanVideoSampler": "TTP_TeaCache HunyuanVideo Sampler",
    "TTP_Smart_Tile_Interactive_Crop_Experimental": "TTP Smart Tile Interactive Crop",
    "TTP_Smart_Tile_Set_Preview_Experimental": "TTP Smart Tile Set Preview",
    "TTP_QwenVL3_Local_Loader_Experimental": "TTP QwenVL3 Local Loader",
    "TTP_Smart_Tile_QwenVL_Prompt_Set_Builder_Experimental": "TTP Smart Tile QwenVL Prompt Set Builder",
    "TTP_Smart_Tile_Loop_Source_Experimental": "TTP Smart Tile Loop Source",
    "TTP_Smart_Tile_Loop_Collect_Experimental": "TTP Smart Tile Loop Collect",
    "TTP_Smart_Tile_Image_Upscale_Prep_Experimental": "TTP Smart Tile Image Upscale Prep",
    "TTP_Smart_Tile_Assemble_Experimental": "TTP Smart Tile Assemble",
    "TTP_Smart_Tile_Save_Final_Image_Experimental": "TTP Smart Tile Save Final Image",
}
