import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageChops, ImageEnhance
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
    "TeaCacheHunyuanVideoSampler": TeaCacheHunyuanVideoSampler
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
    "TeaCacheHunyuanVideoSampler": "TTP_TeaCache HunyuanVideo Sampler"
}
