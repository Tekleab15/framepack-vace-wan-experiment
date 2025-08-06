import torch
import os
import comfy.model_management as mm
import comfy.utils
import comfy.latent_formats
import comfy.sd 

from ..models.wan.framepack_vace import FramepackVace
from ..models.utils.preprocessor import VaceVideoProcessor

class VACE_FRAMEPACK_MODEL_LOADER:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_dir": ("STRING", {"default": "models/Wan2.1-VACE-1.3B/"}),
                "model_config_name": (list(FramepackVace.get_supported_configs().keys()), {"default": "vace-1.3B"}), 
                "precision": (["bf16", "fp32", "fp16"], {"default": "bf16"}),
                "device_id": ("INT", {"default": 0, "min": 0, "max": torch.cuda.device_count() - 1}),
                # Optional: Add arguments for FSDP, USP if configurable at load time
                "t5_cpu": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("FRAMEPACK_VACE_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "VACE/Framepack"
    DESCRIPTION = "Loads the VACE Wan model with Framepack capabilities."

    def load_model(self, ckpt_dir, model_config_name, precision, device_id, t5_cpu):
        from models.wan.configs.shared_config import WAN_CONFIGS # Assuming path
        config = WAN_CONFIGS[model_config_name]
        
        # Determine dtype
        param_dtype = {
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
            "fp16": torch.float16
        }[precision]
        config.param_dtype = param_dtype

        # Instantiate FramepackVace
        model_instance = FramepackVace(
            config=config,
            checkpoint_dir=ckpt_dir,
            device_id=device_id,
            rank=0, # Assuming single-node, rank 0 for ComfyUI
            t5_cpu=t5_cpu,
            
            # FSDP/USP arguments might need to be passed if the init handles them
            t5_fsdp=False,
            dit_fsdp=False,
            use_usp=False,
        )
        return (model_instance,)
