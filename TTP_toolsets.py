import base64
import io
import json
import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageChops, ImageEnhance, ImageDraw
import node_helpers
import torch
import comfy.model_management
import comfy.samplers
import comfy.sample
import comfy.utils
import latent_preview
from typing import Any, List, Tuple, Optional, Union, Dict

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


def _ttp_round_to(value, multiple):
    if multiple <= 1:
        return int(round(value))
    return int(np.ceil(value / multiple) * multiple)


def _ttp_parse_box_value(value, size):
    if isinstance(value, float) and 0.0 <= value <= 1.0:
        return int(round(value * size))
    return int(round(value))


def _ttp_parse_margin_value(value, size):
    if isinstance(value, float) and 0.0 <= value <= 1.0:
        return int(round(value * size))
    return int(round(value))


def _ttp_normalize_tile_box(tile, image_width, image_height, defaults):
    if all(key in tile for key in ("x0", "y0", "x1", "y1")):
        x0 = _ttp_parse_box_value(tile.get("x0", 0), image_width)
        y0 = _ttp_parse_box_value(tile.get("y0", 0), image_height)
        x1 = _ttp_parse_box_value(tile.get("x1", image_width), image_width)
        y1 = _ttp_parse_box_value(tile.get("y1", image_height), image_height)
        x, x_end = sorted((x0, x1))
        y, y_end = sorted((y0, y1))
        w = x_end - x
        h = y_end - y
    else:
        x = _ttp_parse_box_value(tile.get("x", 0), image_width)
        y = _ttp_parse_box_value(tile.get("y", 0), image_height)
        w = _ttp_parse_box_value(tile.get("w", tile.get("width", image_width)), image_width)
        h = _ttp_parse_box_value(tile.get("h", tile.get("height", image_height)), image_height)

    x = _ttp_clamp(x, 0, max(0, image_width - 1))
    y = _ttp_clamp(y, 0, max(0, image_height - 1))
    w = _ttp_clamp(w, 1, max(1, image_width - x))
    h = _ttp_clamp(h, 1, max(1, image_height - y))

    pad_value = tile.get("pad", tile.get("padding", defaults.get("pad", 0)))
    blend_value = tile.get("blend", tile.get("feather", defaults.get("blend", 32)))
    pad = _ttp_parse_margin_value(pad_value, max(image_width, image_height))
    blend = _ttp_parse_margin_value(blend_value, max(w, h))

    return {
        "name": str(tile.get("name", f"tile_{tile.get('id', 0)}")),
        "core_box": [x, y, w, h],
        "pad": max(0, pad),
        "blend": max(0, blend),
        "priority": float(tile.get("priority", defaults.get("priority", 50))),
        "importance": float(tile.get("importance", defaults.get("importance", 1.0))),
        "strength": float(tile.get("strength", defaults.get("strength", 1.0))),
        "prompt_tag": str(tile.get("prompt_tag", tile.get("name", ""))),
        "align": bool(tile.get("align", defaults.get("align", False))),
    }


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
        x, y, w, h = normalized["core_box"]
        pad = normalized["pad"]
        sample_left = _ttp_clamp(x - pad, 0, image_width)
        sample_top = _ttp_clamp(y - pad, 0, image_height)
        sample_right = _ttp_clamp(x + w + pad, 0, image_width)
        sample_bottom = _ttp_clamp(y + h + pad, 0, image_height)
        normalized["sample_box"] = [
            sample_left,
            sample_top,
            max(1, sample_right - sample_left),
            max(1, sample_bottom - sample_top),
        ]
        normalized["paste_box"] = [x, y, w, h]
        normalized_tiles.append(normalized)

    return normalized_tiles


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


def _ttp_crop_smart_tiles_from_meta(pil_image, tiles_meta, round_to):
    image_width, image_height = pil_image.size
    tile_images = []
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

        crop = pil_image.crop((sx, sy, sx + sw, sy + sh))
        tile_images.append(crop)
        tile["tile_canvas_box"] = [0, 0, sw, sh]
        x, y, w, h = tile["paste_box"]
        positions.append((x, y, x + w, y + h))

    max_tile_width = max(tile.width for tile in tile_images)
    max_tile_height = max(tile.height for tile in tile_images)
    max_tile_width = max(1, _ttp_round_to(max_tile_width, round_to))
    max_tile_height = max(1, _ttp_round_to(max_tile_height, round_to))
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


def _ttp_decode_image_data(image_data):
    text = str(image_data or "").strip()
    if not text:
        raise ValueError("No image input was provided. Connect image or use Choose image / Paste image in the interactive tile editor.")

    if "," in text and text.split(",", 1)[0].lower().startswith("data:"):
        text = text.split(",", 1)[1]

    try:
        raw = base64.b64decode(text, validate=False)
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise ValueError("image_data is not a valid base64 image. Choose or paste the image again in the interactive tile editor.") from exc


def _ttp_get_interactive_source_image(image=None, image_data=""):
    if image is not None:
        if not isinstance(image, torch.Tensor) or image.ndim != 4 or int(image.shape[0]) <= 0:
            raise ValueError("image input must be a non-empty ComfyUI IMAGE tensor.")
        return tensor2pil(image[0].unsqueeze(0)).convert("RGB")
    return _ttp_decode_image_data(image_data)


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
    CATEGORY = "TTP/Smart Tile Experimental"

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
    CATEGORY = "TTP/Smart Tile Experimental"

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
    CATEGORY = "TTP/Smart Tile Experimental"

    def _add_tile_meta(self, raw_tiles, image_width, image_height):
        tiles_meta = []
        for index, tile in enumerate(raw_tiles):
            normalized = _ttp_normalize_tile_box({**tile, "id": index}, image_width, image_height, {})
            normalized["id"] = index
            x, y, w, h = normalized["core_box"]
            pad = normalized["pad"]
            sample_left = _ttp_clamp(x - pad, 0, image_width)
            sample_top = _ttp_clamp(y - pad, 0, image_height)
            sample_right = _ttp_clamp(x + w + pad, 0, image_width)
            sample_bottom = _ttp_clamp(y + h + pad, 0, image_height)
            normalized["sample_box"] = [
                sample_left,
                sample_top,
                max(1, sample_right - sample_left),
                max(1, sample_bottom - sample_top),
            ]
            normalized["paste_box"] = [x, y, w, h]
            tiles_meta.append(normalized)
        return tiles_meta

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
                "image_data": ("STRING", {"default": "", "multiline": True}),
                "layout_json": ("STRING", {"default": default_layout, "multiline": False}),
                "default_pad": ("INT", {"default": 128, "min": 0, "max": 2048, "step": 8}),
                "default_blend": ("INT", {"default": 64, "min": 0, "max": 1024, "step": 8}),
                "include_full_image": ("BOOLEAN", {"default": False}),
                "round_to": ("INT", {"default": 8, "min": 1, "max": 128, "step": 1}),
            },
            "optional": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "TTP_SMART_TILE_META", "LIST", "IMAGE", "STRING")
    RETURN_NAMES = ("source_image", "tiles", "tile_meta", "positions", "preview", "layout_json")
    FUNCTION = "interactive_crop_tiles"
    CATEGORY = "TTP/Smart Tile Experimental"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        image_data = str(kwargs.get("image_data", ""))
        fingerprint = {
            "image_data_length": len(image_data),
            "image_data_tail": image_data[-64:],
            "layout_json": kwargs.get("layout_json", ""),
            "default_pad": kwargs.get("default_pad"),
            "default_blend": kwargs.get("default_blend"),
            "include_full_image": kwargs.get("include_full_image"),
            "round_to": kwargs.get("round_to"),
            "image": type(kwargs.get("image")).__name__ if kwargs.get("image") is not None else None,
        }
        return json.dumps(fingerprint, sort_keys=True)

    def interactive_crop_tiles(
        self,
        image_data,
        layout_json,
        default_pad=128,
        default_blend=64,
        include_full_image=False,
        round_to=8,
        image=None,
    ):
        pil_image = _ttp_get_interactive_source_image(image=image, image_data=image_data)
        image_width, image_height = pil_image.size
        normalized_layout_json = _ttp_interactive_layout_with_defaults(
            layout_json,
            int(default_pad),
            int(default_blend),
            bool(include_full_image),
        )
        tiles_meta = _ttp_parse_smart_tile_layout(normalized_layout_json, image_width, image_height)
        tiles, tile_meta, positions, preview = _ttp_crop_smart_tiles_from_meta(pil_image, tiles_meta, int(round_to))
        return (
            pil2tensor(pil_image),
            tiles,
            tile_meta,
            positions,
            preview,
            normalized_layout_json,
        )


class TTP_Smart_Tile_Assemble_Experimental:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sampled_tiles": ("IMAGE",),
                "tile_meta": ("TTP_SMART_TILE_META", {"forceInput": True}),
                "blend_multiplier": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05}),
                "output_scale": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 16.0, "step": 0.05}),
                "use_priority": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "base_image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("image", "weight_preview")
    FUNCTION = "assemble_tiles"
    CATEGORY = "TTP/Smart Tile Experimental"

    def assemble_tiles(self, sampled_tiles, tile_meta, blend_multiplier=1.0, output_scale=0.0, use_priority=True, base_image=None):
        if not isinstance(tile_meta, dict) or tile_meta.get("type") != "ttp_smart_tile":
            raise ValueError("tile_meta must come from TTP Smart Tile Crop (Experimental)")

        tiles_info = tile_meta.get("tiles", [])
        if len(tiles_info) != sampled_tiles.shape[0]:
            raise ValueError(f"sampled_tiles batch ({sampled_tiles.shape[0]}) does not match tile_meta tiles ({len(tiles_info)})")

        original_width, original_height = tile_meta["original_size"]
        if output_scale <= 0:
            inferred_scales = []
            for index, tile in enumerate(tiles_info):
                _, _, sw, sh = tile["sample_box"]
                canvas_width, canvas_height = tile.get("tile_canvas_size", [sw, sh])
                canvas_box = tile.get("tile_canvas_box", [0, 0, sw, sh])
                content_width = float(sampled_tiles[index].shape[1]) * (canvas_box[2] / max(1, canvas_width))
                content_height = float(sampled_tiles[index].shape[0]) * (canvas_box[3] / max(1, canvas_height))
                inferred_scales.append(content_width / max(1, sw))
                inferred_scales.append(content_height / max(1, sh))
            output_scale = float(np.median(inferred_scales)) if inferred_scales else 1.0
            output_scale = max(0.01, output_scale)

        output_width = max(1, int(round(original_width * output_scale)))
        output_height = max(1, int(round(original_height * output_scale)))

        if base_image is not None:
            base_pil = tensor2pil(base_image[0].unsqueeze(0)).convert("RGB")
            if base_pil.size != (output_width, output_height):
                base_pil = base_pil.resize((output_width, output_height), Image.Resampling.LANCZOS)
            canvas = np.array(base_pil).astype(np.float32) / 255.0
            weights = np.ones((output_height, output_width, 1), dtype=np.float32)
        else:
            canvas = np.zeros((output_height, output_width, 3), dtype=np.float32)
            weights = np.zeros((output_height, output_width, 1), dtype=np.float32)

        for index, tile in enumerate(tiles_info):
            tile_pil = tensor2pil(sampled_tiles[index].unsqueeze(0)).convert("RGB")
            sx, sy, sw, sh = tile["sample_box"]
            px, py, pw, ph = tile["paste_box"]

            canvas_width, canvas_height = tile.get("tile_canvas_size", [sw, sh])
            canvas_box = tile.get("tile_canvas_box", [0, 0, sw, sh])
            content_left = int(round(tile_pil.width * canvas_box[0] / max(1, canvas_width)))
            content_top = int(round(tile_pil.height * canvas_box[1] / max(1, canvas_height)))
            content_width = max(1, int(round(tile_pil.width * canvas_box[2] / max(1, canvas_width))))
            content_height = max(1, int(round(tile_pil.height * canvas_box[3] / max(1, canvas_height))))

            tile_scale_x = content_width / max(1, sw)
            tile_scale_y = content_height / max(1, sh)
            local_left = int(round((px - sx) * tile_scale_x))
            local_top = int(round((py - sy) * tile_scale_y))
            local_right = int(round(local_left + pw * tile_scale_x))
            local_bottom = int(round(local_top + ph * tile_scale_y))
            local_left = _ttp_clamp(content_left + local_left, 0, tile_pil.width - 1)
            local_top = _ttp_clamp(content_top + local_top, 0, tile_pil.height - 1)
            local_right = _ttp_clamp(content_left + local_right, local_left + 1, tile_pil.width)
            local_bottom = _ttp_clamp(content_top + local_bottom, local_top + 1, tile_pil.height)

            paste_region = tile_pil.crop((local_left, local_top, local_right, local_bottom))
            out_x = int(round(px * output_scale))
            out_y = int(round(py * output_scale))
            out_w = max(1, int(round(pw * output_scale)))
            out_h = max(1, int(round(ph * output_scale)))
            if paste_region.size != (out_w, out_h):
                paste_region = paste_region.resize((out_w, out_h), Image.Resampling.LANCZOS)

            blend = int(round(tile.get("blend", 0) * output_scale * blend_multiplier))
            mask = _ttp_create_feather_mask(out_w, out_h, blend)
            importance = max(0.0, float(tile.get("importance", 1.0)))
            priority = max(0.0, float(tile.get("priority", 50.0)))
            priority_scale = (1.0 + priority / 100.0) if use_priority else 1.0
            mask = mask * importance * priority_scale

            region = np.array(paste_region).astype(np.float32) / 255.0
            x_end = min(output_width, out_x + out_w)
            y_end = min(output_height, out_y + out_h)
            region_w = x_end - out_x
            region_h = y_end - out_y
            if region_w <= 0 or region_h <= 0:
                continue
            canvas[out_y:y_end, out_x:x_end] += region[:region_h, :region_w] * mask[:region_h, :region_w]
            weights[out_y:y_end, out_x:x_end] += mask[:region_h, :region_w]

        safe_weights = np.maximum(weights, 1e-6)
        output = canvas / safe_weights
        output = np.clip(output, 0.0, 1.0)
        weight_preview = np.clip(weights / max(1e-6, weights.max()), 0.0, 1.0)
        weight_preview = np.repeat(weight_preview, 3, axis=2)

        return (
            torch.from_numpy(output.astype(np.float32)).unsqueeze(0),
            torch.from_numpy(weight_preview.astype(np.float32)).unsqueeze(0),
        )


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
    "TTP_Smart_Tile_Layout_Experimental": TTP_Smart_Tile_Layout_Experimental,
    "TTP_Smart_Tile_Crop_Experimental": TTP_Smart_Tile_Crop_Experimental,
    "TTP_Smart_Tile_Visual_Crop_Experimental": TTP_Smart_Tile_Visual_Crop_Experimental,
    "TTP_Smart_Tile_Interactive_Crop_Experimental": TTP_Smart_Tile_Interactive_Crop_Experimental,
    "TTP_Smart_Tile_Assemble_Experimental": TTP_Smart_Tile_Assemble_Experimental,
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
    "TTP_Smart_Tile_Layout_Experimental": "TTP Smart Tile Layout (Experimental)",
    "TTP_Smart_Tile_Crop_Experimental": "TTP Smart Tile Crop (Experimental)",
    "TTP_Smart_Tile_Visual_Crop_Experimental": "TTP Smart Tile Param Crop (Experimental)",
    "TTP_Smart_Tile_Interactive_Crop_Experimental": "TTP Smart Tile Interactive Crop (Experimental)",
    "TTP_Smart_Tile_Assemble_Experimental": "TTP Smart Tile Assemble (Experimental)",
}
