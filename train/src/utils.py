from pathlib import Path
from peft import PeftModel
import torch
from transformers import BitsAndBytesConfig, Qwen2VLForConditionalGeneration, AutoProcessor, AutoConfig, Qwen2_5_VLForConditionalGeneration
import warnings
import os
import json
import importlib
import inspect
from types import ModuleType
import base64
from io import BytesIO
from PIL import Image
import decord


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)

# This code is borrowed from LLaVA
def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, 
                          device_map="auto", device="cuda", use_flash_attn=False, **kwargs):
    kwargs = {"device_map": device_map}
    
    if device != "cuda":
        kwargs['device_map'] = {"":device}
    
    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    if use_flash_attn:
        kwargs['_attn_implementation'] = 'flash_attention_2'

    if is_lora_model(model_path) and model_base is None:
        warnings.warn('There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument.')
    if is_lora_model(model_path) and model_base is not None:
        lora_cfg_pretrained = AutoConfig.from_pretrained(model_path)
        if hasattr(lora_cfg_pretrained, 'quantization_config'):
            del lora_cfg_pretrained.quantization_config
        processor = AutoProcessor.from_pretrained(model_base)
        print('Loading Qwen2-VL from base model...')
        if "Qwen2.5" in model_base:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
        else:
            model = Qwen2VLForConditionalGeneration.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
        token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
        if model.lm_head.weight.shape[0] != token_num:
            model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
            model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

        print('Loading additional Qwen2-VL weights...')
        non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_state_dict.bin'), map_location='cpu')
        non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
        if any(k.startswith('model.model.') for k in non_lora_trainables):
            non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
        model.load_state_dict(non_lora_trainables, strict=False)
    
        print('Loading LoRA weights...')
        model = PeftModel.from_pretrained(model, model_path)

        print('Merging LoRA weights...')
        model = model.merge_and_unload()

        print('Model Loaded!!!')

    else:
        print(f"Loading model from {model_path} as a standard model. Adapter files were not found, so it can't be merged")
        config_path = Path(model_path) / 'config.json'
        with open(config_path, 'r') as f:
            config = json.load(f)

        if "Qwen2_5" in config["architectures"][0]:
            processor = AutoProcessor.from_pretrained(model_path)
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

        else:
            processor = AutoProcessor.from_pretrained(model_path)
            model = Qwen2VLForConditionalGeneration.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    return processor, model

def is_lora_model(model_path: str | Path) -> bool:
    """
    Check if a model directory contains LoRA adapter files.
    
    Args:
        model_path: Path to the model directory
        
    Returns:
        bool: True if the directory contains LoRA adapter files
    """
    model_dir = Path(model_path)
    return (model_dir / 'adapter_config.json').exists() and (model_dir / 'adapter_model.safetensors').exists()

def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]
    
def load_reward_funcs(
    module_path: str = "train.reward_funcs",
    training_args=None,
    *,
    name_pred=lambda n: n.endswith("_reward"),
    obj_pred=lambda o: callable(o),
    keep_order: bool = True,
):
    mod: ModuleType = importlib.import_module(module_path)
    members = inspect.getmembers(mod, predicate=obj_pred)
    available_funcs = {n: o for n, o in members if name_pred(n)}

    selected_funcs = []
    weights = []

    if training_args is not None:
        func_names = training_args.reward_func_names.split(",") if training_args.reward_func_names else []
        weight_vals = (
            [float(w) for w in training_args.reward_weights.split(",")]
            if training_args.reward_weights
            else [1.0] * len(func_names)
        )
        if len(func_names) != len(weight_vals):
            raise ValueError(
                f"Mismatch: {len(func_names)} reward_func_names but {len(weight_vals)} reward_weights"
            )

        for name, weight in zip(func_names, weight_vals):
            if name not in available_funcs:
                raise ValueError(f"Reward function '{name}' not found in {module_path}")
            if name == "reasoning_reward" and not getattr(training_args, "reward_llm_judge", None):
                # skip reasoning reward if judge is disabled
                continue
            selected_funcs.append((name, available_funcs[name]))
            weights.append(weight)

    else:
        # fallback: load all
        selected_funcs = list(available_funcs.items())
        weights = [1.0] * len(selected_funcs)

    return [o for _, o in selected_funcs], weights


def video_to_first_frame_base64(video_path: str) -> str:
    """
    Given a video path, return the Base64 encoded version of the first frame using decord.
    """
    vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
    frame = vr[0].asnumpy()  # First frame (H, W, C)
    img = Image.fromarray(frame)

    buffer = BytesIO()
    img.save(buffer, format="PNG")
    base64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{base64_str}"


def replace_video_with_first_frame(inputs: list[dict]) -> list[dict]:
    """
    Replace video entries in inputs with first-frame base64 images.
    """
    new_inputs = []
    for sample in inputs:
        new_sample = sample.copy()
        new_prompt = []
        for message in sample["prompt"]:
            if message["role"] == "user" and isinstance(message["content"], list):
                new_content = []
                for ele in message["content"]:
                    if ele.get("type") == "video" and "video" in ele:
                        # Convert first frame to base64
                        b64_img = video_to_first_frame_base64(ele["video"])
                        # Replace video entry with image entry
                        new_content.append({
                            "type": "image",
                            "image": b64_img
                        })
                    else:
                        new_content.append(ele)
                message = {**message, "content": new_content}
            new_prompt.append(message)
        new_sample["prompt"] = new_prompt
        new_inputs.append(new_sample)
    return new_inputs


def mask_key_frames(video_inputs, video_kwargs):
    """
    Replace key frames with white frames (all ones), keeping video length intact.

    Args:
        video_inputs (list[torch.Tensor]):
            Each entry is a video tensor (T, C, H, W) with values in [0,1].
        video_kwargs (dict):
            Must contain 'key_frame_indices', a list of index lists (one per video).

    Returns:
        tuple: (mask_key_frames_image_inputs, mask_key_frames_video_inputs)
            where image_inputs = None, video_inputs = list of masked videos
    """
    key_frame_indices_list = video_kwargs.get("key_frame_indices", [])
    masked_videos = []

    for video, key_indices in zip(video_inputs, key_frame_indices_list):
        if not isinstance(video, torch.Tensor):
            raise TypeError(f"Expected video tensor, got {type(video)}")

        masked_video = video.clone()
        for idx in key_indices:
            if 0 <= idx < masked_video.shape[0]:
                masked_video[idx] = torch.ones_like(masked_video[idx]) * 255  # white frame in [0, 255]
        masked_videos.append(masked_video)

    return None, masked_videos
