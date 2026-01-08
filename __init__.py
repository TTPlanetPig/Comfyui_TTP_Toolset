from .TTP_toolsets import NODE_CLASS_MAPPINGS as TOOLSET_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as TOOLSET_DISPLAY
from .LTXVFirstLastFrameControl_TTP import LTXVFirstLastFrameControl_TTP, LTXVMiddleFrame_TTP, LTXVContext_TTP

NODE_CLASS_MAPPINGS = {
    **TOOLSET_MAPPINGS,
    "LTXVFirstLastFrameControl_TTP": LTXVFirstLastFrameControl_TTP,
    "LTXVMiddleFrame_TTP": LTXVMiddleFrame_TTP,
    "LTXVContext_TTP": LTXVContext_TTP,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **TOOLSET_DISPLAY,
    "LTXVFirstLastFrameControl_TTP": "LTX First Last Frame Control (TTP)",
    "LTXVMiddleFrame_TTP": "LTX Middle Frame (TTP)",
    "LTXVContext_TTP": "LTX Video Context (TTP)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
