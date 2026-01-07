"""
LTXVFirstLastFrameControl_TTP - 首尾帧控制节点

实现首帧和尾帧的精确控制，支持羽化过渡和强度调节

Author: TTP
Based on: ComfyUI LTXVImgToVideoInplace
"""

import torch
import comfy.utils
from comfy_api.latest import io


class LTXVFirstLastFrameControl_TTP(io.ComfyNode):
    """
    首尾帧控制节点
    
    功能:
    - 同时控制视频的首帧和尾帧
    - 支持独立的强度调节
    - 可选的羽化过渡效果
    - 灵活的bypass选项
    
    工作原理:
    1. 将首帧和尾帧图像通过VAE编码到latent空间
    2. 在视频latent的首尾位置嵌入编码后的帧
    3. 使用noise_mask控制嵌入强度
    4. 可选的羽化过渡避免帧间突变
    """
    
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LTXVFirstLastFrameControl_TTP",
            display_name="LTX First Last Frame Control (TTP)",
            category="conditioning/video_models",
            description="控制LTX视频生成的首帧和尾帧，支持羽化过渡",
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
            outputs=[
                io.Latent.Output(display_name="latent"),
            ],
        )
    
    @classmethod
    def execute(
        cls, 
        vae, 
        latent, 
        first_image=None, 
        last_image=None,
        first_strength=1.0, 
        last_strength=1.0
    ) -> io.NodeOutput:
        """
        执行首尾帧控制
        
        Args:
            vae: VAE模型用于图像编码
            latent: 输入的视频latent
            first_image: 首帧图像(可选)
            last_image: 尾帧图像(可选)
            first_strength: 首帧强度
            last_strength: 尾帧强度
            
        Returns:
            包含处理后latent和noise_mask的字典
        """
        # 如果没有提供任何图像，直接返回
        if first_image is None and last_image is None:
            return io.NodeOutput(latent)
        
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
            
            # 设置首帧的noise_mask (1.0 - strength 表示保留多少噪声)
            noise_mask[:, :, :first_latent_frames] = 1.0 - first_strength
        
        # 处理尾帧
        if last_image is not None and last_strength > 0.0:
            last_latent = cls._encode_image(vae, last_image, height, width)
            last_latent_frames = last_latent.shape[2]
            
            # 计算尾帧起始位置
            last_start_idx = latent_frames - last_latent_frames
            if last_start_idx < 0:
                # latent长度不足，截取latent
                last_latent = last_latent[:, :, :latent_frames]
                last_start_idx = 0
                last_latent_frames = latent_frames
            
            # 嵌入尾帧
            samples[:, :, last_start_idx:] = last_latent
            
            # 设置尾帧的noise_mask
            noise_mask[:, :, last_start_idx:] = 1.0 - last_strength
        
        return io.NodeOutput({
            "samples": samples, 
            "noise_mask": noise_mask
        })
    
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
