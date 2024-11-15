import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageChops, ImageEnhance
import node_helpers
import torch

def pil2tensor(image: Image) -> torch.Tensor:
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def tensor2pil(t_image: torch.Tensor) -> Image:
    return Image.fromarray(np.clip(255.0 * t_image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

        
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
            return ([pil2tensor(image).unsqueeze(0)], [(0, 0, img_width, img_height)], (img_width, img_height), (1, 1))

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
                "padding": ("INT", {"default": 64, "min": 1}),
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

        # 计算左偏移和右偏移，确保裁剪尺寸一致
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
            # 对于垂直方向，进行类似的调整
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
                    row_image.paste(tile_image, (row_image.width, 0))
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
                final_image.paste(row_images[row], (0, final_image.height))

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
            tile_width = int(raw_W / width_factor)
            tile_height = int(raw_H / height_factor)
            # 验证 tile_width 和 tile_height 是否可以被8整除
            if tile_width % 8 != 0:
                tile_width = ((tile_width + 7) // 8) * 8
            if tile_height % 8 != 0:
                tile_height = ((tile_height + 7) // 8) * 8
        
        else:
            # 使用正确的公式计算 tile_width 和 tile_height
            tile_width = int(raw_W / (1 + (width_factor - 1) * (1 - overlap_rate)))
            tile_height = int(raw_H / (1 + (height_factor - 1) * (1 - overlap_rate)))

            # 验证 tile_width 和 tile_height 是否可以被8整除
            if tile_width % 8 != 0:
                tile_width = (tile_width // 8) * 8
            if tile_height % 8 != 0:
                tile_height = (tile_height // 8) * 8

        # 返回结果
        return (tile_width, tile_height)
        
class TTP_Expand_And_Mask:

    def __init__(self, *args, **kwargs):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),  # 输入一张图片
                "fill_mode": (["duplicate", "white"], {"default": "duplicate", "label": "Fill Mode"}),
                # fill_mode是一个字符串列表参数，可以选择"duplicate"或"white"
                "direction": (["left", "right", "top", "bottom"], {"default": "right", "label": "Expansion Direction"}),
                # direction是一个字符串列表参数，可以选择扩展方向："left", "right", "top", "bottom"
                "num_blocks": ("INT", {"default": 1, "min": 1, "max": 2, "step": 1, "label": "Number of Blocks"}),
                # num_blocks表示要添加扩展块的数量，可选择1或2
                "fill_alpha_decision": ("BOOLEAN", {"default": False, "label": "Fill Alpha with White"}),
                # fill_alpha_decision为一个布尔值参数，用来决定是否将输出图像透明区域填充为白色
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("EXPANDED_IMAGE", "MASK")
    FUNCTION = "expand_and_mask"
    CATEGORY = "TTP/Image"

    def expand_and_mask(self, image, fill_mode="duplicate", direction="right", num_blocks=1, fill_alpha_decision=False):
        pil_image = tensor2pil(image)
        width, height = pil_image.size
        has_alpha = (pil_image.mode == 'RGBA')


        def create_fill_image():
            if pil_image.mode == 'RGBA':
                return Image.new("RGBA", (width, height), color=(255, 255, 255, 255))
            elif pil_image.mode == 'RGB':
                return Image.new("RGB", (width, height), color=(255, 255, 255))
            elif pil_image.mode == 'L':
                return Image.new("L", (width, height), color=255)
            else:
                raise ValueError(f"Unsupported image mode for fill: {pil_image.mode}")

        if fill_mode == "duplicate":
            fill_image = pil_image.copy()
        elif fill_mode == "white":
            fill_image = create_fill_image()
        else:
            fill_image = pil_image.copy()

        if direction in ["left", "right"]:
            new_width = width + width * num_blocks
            new_height = height
        else:  # direction in ["top", "bottom"]
            new_width = width
            new_height = height + height * num_blocks

        expanded_image_mode = pil_image.mode
        expanded_image = Image.new(expanded_image_mode, (new_width, new_height))

        if direction == "left":
            for i in range(num_blocks):
                expanded_image.paste(fill_image, (width * i, 0))
            expanded_image.paste(pil_image, (width * num_blocks, 0))
        elif direction == "right":
            expanded_image.paste(pil_image, (0, 0))
            for i in range(num_blocks):
                expanded_image.paste(fill_image, (width + width * i, 0))
        elif direction == "top":
            for i in range(num_blocks):
                expanded_image.paste(fill_image, (0, height * i))
            expanded_image.paste(pil_image, (0, height * num_blocks))
        elif direction == "bottom":
            expanded_image.paste(pil_image, (0, 0))
            for i in range(num_blocks):
                expanded_image.paste(fill_image, (0, height + height * i))
        else:
            raise ValueError(f"Unsupported direction: {direction}")

        mask_array = np.zeros((new_height, new_width), dtype=np.float32)

        if has_alpha:
            alpha_array = np.array(pil_image.getchannel("A"), dtype=np.float32) / 255.0
            alpha_mask_array = 1.0 - alpha_array

            if direction == "left":
                mask_array[:, width * num_blocks:width * num_blocks + width] = alpha_mask_array
            elif direction == "right":
                mask_array[:, :width] = alpha_mask_array
            elif direction == "top":
                mask_array[height * num_blocks:height * num_blocks + height, :] = alpha_mask_array
            elif direction == "bottom":
                mask_array[:height, :] = alpha_mask_array

        if direction == "left":
            mask_array[:, :width * num_blocks] = 1.0
        elif direction == "right":
            mask_array[:, width:] = 1.0
        elif direction == "top":
            mask_array[:height * num_blocks, :] = 1.0
        elif direction == "bottom":
            mask_array[height:, :] = 1.0

        mask_tensor = torch.from_numpy(mask_array).unsqueeze(0).unsqueeze(0)

        if fill_alpha_decision and has_alpha:
            expanded_image = expanded_image.convert('RGBA')  # 确保图像是RGBA模式
            background = Image.new('RGBA', expanded_image.size, (255, 255, 255, 255)) 
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
    "TTP_text_mix": TTP_text_mix
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TTPlanet_Tile_Preprocessor_Simple": "�TTP Tile Preprocessor Simple",
    "TTP_Image_Tile_Batch": "�TTP_Image_Tile_Batch",
    "TTP_Image_Assy": "�TTP_Image_Assy",
    "TTP_CoordinateSplitter": "�TTP_CoordinateSplitter",
    "TTP_condtobatch": "�TTP_cond to batch",
    "TTP_condsetarea_merge": "�TTP_condsetarea_merge",
    "TTP_Tile_image_size": "�TTP_Tile_image_size",
    "TTP_condsetarea_merge_test": "TTP_condsetarea_merge_test",
    "TTP_Expand_And_Mask": "TTP_Expand_And_Mask",
    "TTP_text_mix": "TTP_text_mix"
}
