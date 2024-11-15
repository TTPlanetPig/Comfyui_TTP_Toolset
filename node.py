import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageChops, ImageEnhance
import node_helpers
import torch

def pil2tensor(image: Image) -> torch.Tensor:
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def tensor2pil(t_image: torch.Tensor) -> Image:
    return Image.fromarray(np.clip(255.0 * t_image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

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
    "TTP_Expand_And_Mask": TTP_Expand_And_Mask,
    "TTP_text_mix": TTP_text_mix
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TTP_Expand_And_Mask": "TTP_Expand_And_Mask",
    "TTP_text_mix": "TTP_text_mix"
}
