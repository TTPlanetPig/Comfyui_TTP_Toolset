from .TTP_toolsets import NODE_CLASS_MAPPINGS as TOOLSET_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as TOOLSET_DISPLAY
from .TTP_smart_tile import TTP_Smart_Tile_Batch, TTP_Smart_Image_Assy

NODE_CLASS_MAPPINGS = {
    **TOOLSET_MAPPINGS,
    "TTP_Smart_Tile_Batch": TTP_Smart_Tile_Batch,
    "TTP_Smart_Image_Assy": TTP_Smart_Image_Assy,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **TOOLSET_DISPLAY,
    "TTP_Smart_Tile_Batch": "TTP Smart Tile Batch",
    "TTP_Smart_Image_Assy": "TTP Smart Image Assy",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
