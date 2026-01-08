"""
LTX First Last Frame Control with Middle Frame and Context Support

包含三个节点:
1. LTXVContext_TTP: 视频续接context节点
2. LTXVMiddleFrame_TTP: 可串联的中间帧节点
3. LTXVFirstLastFrameControl_TTP: 首尾帧控制节点

Author: TTP
"""

import torch
import comfy.utils
from comfy_api.latest import io


class LTXVContext_TTP:
    """
    视频续接Context节点
    
    功能:
    - 从上一个视频提取结尾帧作为context
    - 应用到新视频的开头
    - 实现平滑续接
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "previous_video": ("IMAGE",),  # 上一个视频（IMAGE类型，batch维度是帧）
                "vae": ("VAE",),
                "latent": ("LATENT",),  # 新视频的空白latent
                "context_latent_frames": ("INT", {
                    "default": 6, 
                    "min": 2, 
                    "max": 20, 
                    "step": 1,
                    "tooltip": "Context的latent帧数（推荐6）\n"
                               "注：latent帧0=1原始帧，其余=8原始帧\n"
                               "6帧 = 1+40 = 41原始帧 ≈ 1.64秒@25fps"
                }),
            },
            "optional": {
                "context_strength": ("FLOAT", {
                    "default": 1.0, 
                    "min": 0.0, 
                    "max": 1.0, 
                    "step": 0.05,
                    "tooltip": "Context固定强度（1.0=完全固定）"
                }),
            }
        }
    
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "apply_context"
    CATEGORY = "conditioning/video_models"
    
    def apply_context(self, previous_video, vae, latent, context_latent_frames, context_strength=1.0):
        """
        从上一个视频提取context并应用到新latent
        
        Args:
            previous_video: 上一个视频的帧（IMAGE格式）
            vae: VAE模型
            latent: 新视频的空白latent
            context_latent_frames: 要使用的context latent帧数
            context_strength: context强度
        
        Returns:
            包含context的latent
        """
        samples = latent["samples"].clone()
        batch, channels, latent_frames, latent_height, latent_width = samples.shape
        
        # 获取VAE缩放因子
        _, height_scale_factor, width_scale_factor = vae.downscale_index_formula
        target_width = latent_width * width_scale_factor
        target_height = latent_height * height_scale_factor
        
        # 计算需要多少原始帧来生成context_latent_frames个latent帧
        # LTX的公式: latent_frames = ((original_frames - 1) // 8) + 1
        # 反推: original_frames = (latent_frames - 1) * 8 + 1
        # 8N+1结构：第0帧独立，其余每8帧压缩为1个latent帧
        required_frames = (context_latent_frames - 1) * 8 + 1
        
        # 从previous_video提取最后N帧
        total_video_frames = previous_video.shape[0]
        start_idx = max(0, total_video_frames - required_frames)
        context_frames = previous_video[start_idx:]
        
        # 确保帧数符合要求（可能不足required_frames）
        actual_frames = context_frames.shape[0]
        
        # 调整图像尺寸
        if context_frames.shape[1] != target_height or context_frames.shape[2] != target_width:
            pixels = comfy.utils.common_upscale(
                context_frames.movedim(-1, 1), 
                target_width, 
                target_height, 
                "bilinear", 
                "center"
            ).movedim(1, -1)
        else:
            pixels = context_frames
        
        # 只取RGB通道
        encode_pixels = pixels[:, :, :, :3]
        
        # VAE编码
        context_latent = vae.encode(encode_pixels)
        actual_latent_frames = context_latent.shape[2]
        
        # 将context嵌入到新latent的开头
        embed_frames = min(actual_latent_frames, latent_frames)
        samples[:, :, :embed_frames] = context_latent[:, :, :embed_frames]
        
        # 设置noise_mask
        noise_mask = torch.ones(
            (batch, 1, latent_frames, 1, 1),
            dtype=torch.float32,
            device=samples.device,
        )
        
        # Context帧固定
        noise_mask[:, :, :embed_frames] = 1.0 - context_strength
        
        return ({"samples": samples, "noise_mask": noise_mask},)


class LTXVMiddleFrame_TTP:
    """
    中间帧节点 - 可串联（传统ComfyUI格式）
    
    使用传统的INPUT_TYPES和RETURN_TYPES定义，支持anytype"*"
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "position": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            },
            "optional": {
                "middle_frames": ("*",),  # anytype - 接受任何类型
            }
        }
    
    RETURN_TYPES = ("*",)  # anytype - 返回任何类型
    RETURN_NAMES = ("middle_frames",)
    FUNCTION = "execute"
    CATEGORY = "conditioning/video_models"
    
    def execute(self, image, position, strength, middle_frames=None):
        """
        执行中间帧处理
        
        Args:
            image: 中间帧图像
            position: 位置(0.0-1.0)
            strength: 强度
            middle_frames: 上一个节点的中间帧数据（可选）
        
        Returns:
            累积的中间帧数据
        """
        # 初始化或累积中间帧列表
        if middle_frames is None:
            frames_list = []
        else:
            # 复制现有列表
            frames_list = list(middle_frames.get("frames", []))
        
        # 添加新的中间帧信息
        frames_list.append({
            "image": image,
            "position": position,
            "strength": strength,
        })
        
        # 返回累积的数据
        return ({"frames": frames_list},)


class LTXVFirstLastFrameControl_TTP(io.ComfyNode):
    """
    首尾帧控制节点（io.Schema格式）
    
    功能:
    - 控制首帧和尾帧
    - 接受中间帧数据并融合
    - 输出包含所有关键帧的latent
    """
    
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LTXVFirstLastFrameControl_TTP",
            display_name="LTX First Last Frame Control (TTP)",
            category="conditioning/video_models",
            description="控制LTX视频生成的首帧和尾帧，支持中间帧融合",
            inputs=[
                io.Vae.Input("vae"),
                io.Latent.Input("latent"),
                io.Image.Input("first_image", optional=True, tooltip="首帧图像(可选)"),
                io.Image.Input("last_image", optional=True, tooltip="尾帧图像(可选)"),
                io.Float.Input(
                    "first_strength", 
                    default=1.0, 
                    min=0.0, 
                    max=1.0, 
                    step=0.05,
                    tooltip="首帧嵌入强度 (1.0=完全替换, 0.0=不嵌入)"
                ),
                io.Float.Input(
                    "last_strength", 
                    default=1.0, 
                    min=0.0, 
                    max=1.0, 
                    step=0.05,
                    tooltip="尾帧嵌入强度 (1.0=完全替换, 0.0=不嵌入)"
                ),
            ],
            # 不定义middle_frames输入，改用可选参数在execute中接收
            outputs=[
                io.Latent.Output(display_name="latent"),
            ],
        )
    
    @classmethod  
    def INPUT_TYPES(cls):
        """添加传统格式的INPUT_TYPES以支持middle_frames anytype"""
        base_inputs = {
            "required": {
                "vae": ("VAE",),
                "latent": ("LATENT",),
                "first_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "last_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            },
            "optional": {
                "first_image": ("IMAGE",),
                "last_image": ("IMAGE",),
                "middle_frames": ("*",),  # anytype
            }
        }
        return base_inputs
    
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "execute"
    CATEGORY = "conditioning/video_models"
    
    @classmethod
    def execute(
        cls, 
        vae, 
        latent, 
        first_strength=1.0, 
        last_strength=1.0,
        first_image=None, 
        last_image=None,
        middle_frames=None
    ) -> io.NodeOutput:
        """
        执行首尾帧控制和中间帧融合
        
        Args:
            vae: VAE模型
            latent: 输入的视频latent
            first_strength: 首帧强度
            last_strength: 尾帧强度
            first_image: 首帧图像(可选)
            last_image: 尾帧图像(可选)
            middle_frames: 中间帧数据(可选)
            
        Returns:
            包含处理后latent和noise_mask的字典
        """
        # 如果没有提供任何图像和中间帧，直接返回
        has_middle = middle_frames is not None and len(middle_frames.get("frames", [])) > 0
        if first_image is None and last_image is None and not has_middle:
            return (latent,)
        
        # 获取latent信息
        samples = latent["samples"]
        batch, _, latent_frames, latent_height, latent_width = samples.shape
        
        # 获取VAE的缩放因子
        _, height_scale_factor, width_scale_factor = vae.downscale_index_formula
        width = latent_width * width_scale_factor
        height = latent_height * height_scale_factor
        
        # 初始化noise_mask (全1表示不mask)
        noise_mask = torch.ones(
            (batch, 1, latent_frames, 1, 1),
            dtype=torch.float32,
            device=samples.device,
        )
        
        # 处理首帧
        if first_image is not None and first_strength > 0.0:
            first_latent = cls._encode_image(vae, first_image, height, width)
            first_latent_frames = first_latent.shape[2]
            
            # 嵌入首帧
            samples[:, :, :first_latent_frames] = first_latent
            
            # 设置首帧的noise_mask
            noise_mask[:, :, :first_latent_frames] = 1.0 - first_strength
        
        # 处理尾帧
        if last_image is not None and last_strength > 0.0:
            last_latent = cls._encode_image(vae, last_image, height, width)
            last_latent_frames = last_latent.shape[2]
            
            # 计算尾帧起始位置
            last_start_idx = latent_frames - last_latent_frames
            if last_start_idx < 0:
                last_latent = last_latent[:, :, :latent_frames]
                last_start_idx = 0
                last_latent_frames = latent_frames
            
            # 嵌入尾帧
            samples[:, :, last_start_idx:] = last_latent
            
            # 设置尾帧的noise_mask
            noise_mask[:, :, last_start_idx:] = 1.0 - last_strength
        
        # 处理中间帧（如果有）
        if has_middle:
            frames_list = middle_frames["frames"]
            for frame_data in frames_list:
                image = frame_data["image"]
                position = frame_data["position"]
                strength = frame_data["strength"]
                
                if strength <= 0.0:
                    continue
                
                # 编码中间帧图像
                middle_latent = cls._encode_image(vae, image, height, width)
                middle_latent_frames = middle_latent.shape[2]
                
                # 根据百分比计算中间帧位置
                middle_frame_idx = round(position * (latent_frames - 1))
                
                # 确保不超出范围
                middle_frame_idx = max(0, min(middle_frame_idx, latent_frames - middle_latent_frames))
                
                # 嵌入中间帧
                samples[:, :, middle_frame_idx:middle_frame_idx+middle_latent_frames] = middle_latent
                
                # 设置中间帧的noise_mask（使用min确保不覆盖更强的约束）
                current_mask = noise_mask[:, :, middle_frame_idx:middle_frame_idx+middle_latent_frames, :, :]
                new_mask = 1.0 - strength
                # 保留更强的约束（更小的mask值）
                noise_mask[:, :, middle_frame_idx:middle_frame_idx+middle_latent_frames, :, :] = torch.minimum(
                    current_mask, 
                    torch.full_like(current_mask, new_mask)
                )
        
        return ({"samples": samples, "noise_mask": noise_mask},)
    
    @staticmethod
    def _encode_image(vae, image, target_height, target_width):
        """
        编码图像到latent空间
        
        Args:
            vae: VAE模型
            image: 输入图像张量
            target_height: 目标高度
            target_width: 目标宽度
            
        Returns:
            编码后的latent张量
        """
        # 调整图像尺寸
        if image.shape[1] != target_height or image.shape[2] != target_width:
            pixels = comfy.utils.common_upscale(
                image.movedim(-1, 1), 
                target_width, 
                target_height, 
                "bilinear", 
                "center"
            ).movedim(1, -1)
        else:
            pixels = image
        
        # 只取RGB通道(去掉alpha)
        encode_pixels = pixels[:, :, :, :3]
        
        # VAE编码
        latent = vae.encode(encode_pixels)
        
        return latent
    
    # 兼容旧API
    generate = execute
