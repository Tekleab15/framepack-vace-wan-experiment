import torch
import os
import comfy.model_management as mm
import comfy.utils
import comfy.latent_formats
import comfy.sd 

from models.wan.framepack_vace import FramepackVace
from models.utils.preprocessor import VaceVideoProcessor

class VACE_FRAMEPACK_MODEL_TYPE:
    pass

class VACE_SOURCE_DATA_TYPE:
    pass

class VACE_FRAMEPACK_MODEL_LOADER:
    @classmethod
    def INPUT_TYPES(s):
        model_configs = ["vace-1.3B", "vace-14B"]
        return {
            "required": {
                "ckpt_dir": ("STRING", {"default": "models/Wan2.1-VACE-1.3B/"}),
                "model_config_name": (model_configs, {"default": "vace-1.3B"}),
                "precision": (["bf16", "fp32", "fp16"], {"default": "bf16"}),
                "device_id": ("INT", {"default": 0, "min": 0, "max": torch.cuda.device_count() - 1}),
                "t5_cpu": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("VACE_FRAMEPACK_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "VACE/Framepack"
    DESCRIPTION = "Loads the VACE Wan model with Framepack capabilities."

    def load_model(self, ckpt_dir, model_config_name, precision, device_id, t5_cpu):
        from vace.models.wan.configs import WAN_CONFIGS
        config = WAN_CONFIGS[model_config_name]
        
        param_dtype = {
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
            "fp16": torch.float16
        }[precision]
        config.param_dtype = param_dtype

        model_instance = FramepackVace(
            config=config,
            checkpoint_dir=ckpt_dir,
            device_id=device_id,
            rank=0,
            t5_cpu=t5_cpu,
            t5_fsdp=False,
            dit_fsdp=False,
            use_usp=False,
        )
        
        return (model_instance,)
    
class VACE_FRAMEPACK_VIDEO_INPUT_PREPROCESSOR:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("VACE_FRAMEPACK_MODEL",),
                "num_frames": ("INT", {"default": 81, "min": 1}),
                "output_width": ("INT", {"default": 768, "min": 64, "step": 8}),
                "output_height": ("INT", {"default": 512, "min": 64, "step": 8}),
            },
            "optional": {
                "video_path": ("STRING", {"default": "", "optional": True}),
                "mask_path": ("STRING", {"default": "", "optional": True}),
                "ref_image_paths": ("STRING", {"default": "", "optional": True, "tooltip": "Comma-separated paths to reference images."}),
            }
        }

    RETURN_TYPES = ("VACE_SOURCE_DATA",)
    RETURN_NAMES = ("source_data",)
    FUNCTION = "preprocess_input"
    CATEGORY = "VACE/Framepack"
    DESCRIPTION = "Prepares source video, mask, and reference images for VACE Framepack generation."

    def preprocess_input(self, model, video_path, mask_path, ref_image_paths, num_frames, output_width, output_height):
        device = model.device
        
        src_video = [video_path if video_path else None]
        src_mask = [mask_path if mask_path else None]
        src_ref_images = [ref_image_paths.split(',') if ref_image_paths else None]

        preprocessed_video, preprocessed_mask, preprocessed_ref_images = model.prepare_source(
            src_video=src_video,
            src_mask=src_mask,
            src_ref_images=src_ref_images,
            num_frames=num_frames,
            image_size=(output_height, output_width),
            device=device
        )
        
        source_data = {
            "src_video": preprocessed_video,
            "src_mask": preprocessed_mask,
            "src_ref_images": preprocessed_ref_images,
            "output_width": output_width,
            "output_height": output_height,
            "num_frames": num_frames,
            "device": device
        }

        return (source_data,)

class VACE_VAELOADER:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vae_name": (comfy.folder_paths.get_filename_list("vae"),),
                "gpu_id": ("INT", {"default": 0, "min": 0, "max": torch.cuda.device_count() - 1, "tooltip": "GPU device ID to load VAE onto"}),
            }
        }
    RETURN_TYPES = ("VAE",)
    FUNCTION = "load_vae"
    CATEGORY = "VACE/Framepack"
    DESCRIPTION = "Loads a VAE model onto a specific GPU."

    def load_vae(self, vae_name, gpu_id):
        vae_path = comfy.folder_paths.get_full_path_or_raise("vae", vae_name)
        target_device = torch.device(f"cuda:{gpu_id}")
        
        vae_sd = comfy.utils.load_torch_file(vae_path, safe_load=True) 
        
        # Using ComfyUI's VAE class for consistency
        vae = comfy.sd.VAE(sd=vae_sd).eval()
        
        return (vae,)

class VACE_FRAMEPACK_SAMPLER:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("VACE_FRAMEPACK_MODEL",),
                "source_data": ("VACE_SOURCE_DATA",),
                "prompt": ("STRING", {"multiline": True}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "sampling_steps": ("INT", {"default": 50, "min": 1}),
                "guide_scale": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 30.0}),
                "sample_solver": (["unipc", "dpm++"], {"default": "unipc"}),
                "sample_shift": ("FLOAT", {"default": 16.0, "min": 0.0}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent_video",)
    FUNCTION = "sample_video"
    CATEGORY = "VACE/Framepack"
    DESCRIPTION = "Generates video using VACE Framepack model."
    

    def sample_video(self, model, source_data, prompt, negative_prompt, sampling_steps, guide_scale,
                     sample_solver, sample_shift, seed):
        
        src_video = source_data["src_video"]
        src_mask = source_data["src_mask"]
        src_ref_images = source_data["src_ref_images"]
        output_width = source_data["output_width"]
        output_height = source_data["output_height"]
        num_frames = source_data["num_frames"]

        # ComfyUI often expects a dict for latent, so we convert src_video to that format
        generated_video = model.generate_with_framepack(
            input_prompt=prompt,
            input_frames=src_video,
            input_masks=src_mask,
            input_ref_images=src_ref_images,
            size=(output_width, output_height),
            frame_num=num_frames,
            sampling_steps=sampling_steps,
            guide_scale=guide_scale,
            sample_solver=sample_solver,
            shift=sample_shift,
            seed=seed,
            n_prompt=negative_prompt,
            offload_model=True
        )
        
        # The ComfyUI LATENT format expects a dictionary with a 'samples' key
        return ({"samples": generated_video},)


NODE_CLASS_MAPPINGS = {
    "VACE_Framepack_ModelLoader": VACE_FRAMEPACK_MODEL_LOADER,
    "VACE_Framepack_VideoInputPreprocessor": VACE_FRAMEPACK_VIDEO_INPUT_PREPROCESSOR,
    "VACE_Framepack_Sampler": VACE_FRAMEPACK_SAMPLER,
    "VACE_VAELoader": VACE_VAELOADER,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VACE_Framepack_ModelLoader": "VACE Framepack Model Loader",
    "VACE_Framepack_VideoInputPreprocessor": "VACE Framepack Video Input Preprocessor",
    "VACE_Framepack_Sampler": "VACE Framepack Sampler",
    "VACE_VAELoader": "VACE VAE Loader",
}