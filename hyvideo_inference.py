import os
import time
import math
import json
from pathlib import Path
from loguru import logger
from datetime import datetime

import torch
from svg.models.hyvideo.utils.file_utils import save_videos_grid
from svg.models.hyvideo.config import parse_args
from svg.models.hyvideo.inference import HunyuanVideoSampler
import gc


def sparsity_to_width(sparsity, context_length, num_frame, frame_size):
    seq_len = context_length + num_frame * frame_size
    total_elements = seq_len ** 2
    
    sparsity = (sparsity * total_elements - 2 * seq_len * context_length) / total_elements
      
    width = seq_len * (1 - math.sqrt(1 - sparsity))
    width_frame = width / frame_size
    
    return width_frame


if __name__ == "__main__":
    args = parse_args()
    print(args)
    models_root_path = Path(args.model_base)
    if not models_root_path.exists():
        raise ValueError(f"`models_root` not exists: {models_root_path}")

    # Load models
    hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(models_root_path, args=args,device="cpu")
    pipe = hunyuan_video_sampler.pipeline
    hunyuan_video_sampler.model.enable_teacache = True
    hunyuan_video_sampler.model.rel_l1_thresh = 0.15
    hunyuan_video_sampler.model.num_steps =args.infer_steps 
    hunyuan_video_sampler.model.cnt = 0
    
    

def get_linear_split_map():
    hidden_size = 3072
    split_linear_modules_map =  {
                                "img_attn_qkv" : {"mapped_modules" : ["img_attn_q", "img_attn_k", "img_attn_v"] , "split_sizes": [hidden_size, hidden_size, hidden_size]},
                                "linear1" : {"mapped_modules" : ["linear1_attn_q", "linear1_attn_k", "linear1_attn_v", "linear1_mlp"] , "split_sizes":  [hidden_size, hidden_size, hidden_size, 7*hidden_size- 3*hidden_size]}
                                }
    return split_linear_modules_map
try:
    from xformers.ops.fmha.attn_bias import BlockDiagonalPaddedKeysMask
except ImportError:
    BlockDiagonalPaddedKeysMask = None

    from mmgp import offload
    kwargs = { "extraModelsToQuantize": None}
    profile = 2
    preload = 0
    if profile == 2 or profile == 4:
        kwargs["budgets"] = { "transformer" : 100 if preload  == 0 else preload, "text_encoder" : 100, "*" : 1000 }
    elif profile == 3:
        kwargs["budgets"] = { "*" : "70%" }

    split_linear_modules_map = get_linear_split_map()
    offload.split_linear_modules(pipe.transformer, split_linear_modules_map )
    offload.profile(pipe,profile_no=4,quantizeTransformer=False,**kwargs)
    
    # Get the updated args
    args = hunyuan_video_sampler.args

    # Sparsity Related
    transformer = hunyuan_video_sampler.pipeline.transformer
    for _, block in enumerate(transformer.double_blocks):
        block.sparse_args = args
    for _, block in enumerate(transformer.single_blocks):
        block.sparse_args = args
    transformer.sparse_args = args

    if args.pattern == "SVG":
        # We need to get the prompt len in advance, since HunyuanVideo handle the attention mask in a special way
        prompt_mask = hunyuan_video_sampler.get_prompt_mask(
            prompt=args.prompt, 
            height=args.video_size[0],
            width=args.video_size[1],
            video_length=args.video_length,
            negative_prompt=args.neg_prompt,
            infer_steps=args.infer_steps,
            guidance_scale=args.cfg_scale,
            num_videos_per_prompt=args.num_videos,
            embedded_guidance_scale=args.embedded_cfg_scale
        )
        prompt_len = prompt_mask.sum()

        print(f"Memory: {torch.cuda.memory_allocated() // 1024 ** 2} / {torch.cuda.max_memory_allocated() // 1024 ** 2} MB before Inference")

    cfg_size, num_head, head_dim, dtype, device = 1, 24, 128, torch.bfloat16, "cuda"
    context_length, num_frame, frame_size = 256, args.video_length//4 + 1, 3600

    # Calculation
    spatial_width = temporal_width = sparsity_to_width(args.sparsity, context_length, num_frame, frame_size)
                
    print(f"Spatial_width: {spatial_width}, Temporal_width: {temporal_width}. Sparsity: {args.sparsity}")

    save_path = args.output_path
        
    if args.pattern == "SVG":
        masks = ["spatial", "temporal"]

        def get_attention_mask(mask_name):

            context_length = 256
            num_frame = args.video_length // 4 + 1
            frame_size = 3600
            attention_mask = torch.zeros((context_length + num_frame * frame_size, context_length + num_frame * frame_size), device="cpu")

            # TODO: fix hard coded mask
            if mask_name == "spatial":
                pixel_attn_mask = torch.zeros_like(attention_mask[:-context_length, :-context_length], dtype=torch.bool, device="cpu")
                block_size, block_thres = 128, frame_size * 1.5
                num_block = math.ceil(num_frame * frame_size / block_size)
                for i in range(num_block):
                    for j in range(num_block):
                        if abs(i - j) < block_thres // block_size:
                            pixel_attn_mask[i * block_size : (i + 1) * block_size, j * block_size : (j + 1) * block_size] = 1
                attention_mask[:-context_length, :-context_length] = pixel_attn_mask

                attention_mask[-context_length:, :] = 1
                attention_mask[:, -context_length:] = 1
                # attention_mask = torch.load(f"/data/home/xihaocheng/andy_develop/tmp_data/hunyuanvideo/I2VSparse/sparseattn/v5/mask_tensor/mask_spatial.pt", map_location="cpu")

            else:
                pixel_attn_mask = torch.zeros_like(attention_mask[:-context_length, :-context_length], dtype=torch.bool, device="cpu")

                block_size, block_thres = 128, frame_size * 1.5
                num_block = math.ceil(num_frame * frame_size / block_size)
                for i in range(num_block):
                    for j in range(num_block):
                        if abs(i - j) < block_thres // block_size:
                            pixel_attn_mask[i * block_size : (i + 1) * block_size, j * block_size : (j + 1) * block_size] = 1

                pixel_attn_mask = pixel_attn_mask.reshape(frame_size, num_frame, frame_size, num_frame).permute(1, 0, 3, 2).reshape(frame_size * num_frame, frame_size * num_frame)
                attention_mask[:-context_length, :-context_length] = pixel_attn_mask

                attention_mask[-context_length:, :] = 1
                attention_mask[:, -context_length:] = 1
                # attention_mask = torch.load(f"/data/home/xihaocheng/andy_develop/tmp_data/hunyuanvideo/I2VSparse/sparseattn/v5/mask_tensor/mask_temporal.pt", map_location="cpu")
            attention_mask = attention_mask[:args.sample_mse_max_row].cuda()
            return attention_mask


        from svg.models.hyvideo.modules.attenion import Hunyuan_SparseAttn, prepare_flexattention
        from svg.models.hyvideo.modules.custom_models import replace_sparse_forward

        AttnModule = Hunyuan_SparseAttn
        AttnModule.num_sampled_rows = args.num_sampled_rows
        AttnModule.num_frame = num_frame
        AttnModule.sample_mse_max_row = args.sample_mse_max_row
        AttnModule.attention_masks = [get_attention_mask(mask_name) for mask_name in masks]
        AttnModule.first_layers_fp = args.first_layers_fp
        AttnModule.first_times_fp = args.first_times_fp

        block_mask = prepare_flexattention(
                cfg_size, num_head, head_dim, dtype, device, 
                context_length, prompt_len, num_frame, frame_size, 
                diag_width=spatial_width, multiplier=temporal_width
            )
        AttnModule.block_mask = block_mask
        replace_sparse_forward()


    # Start sampling
    # TODO: batch inference check
    torch.cuda.empty_cache()
    gc.collect()
    outputs = hunyuan_video_sampler.predict(
        prompt=args.prompt, 
        height=args.video_size[0],
        width=args.video_size[1],
        video_length=args.video_length,
        seed=args.seed,
        negative_prompt=args.neg_prompt,
        infer_steps=args.infer_steps,
        guidance_scale=args.cfg_scale,
        num_videos_per_prompt=args.num_videos,
        flow_shift=args.flow_shift,
        batch_size=args.batch_size,
        embedded_guidance_scale=args.embedded_cfg_scale
    )
    samples = outputs['samples']
    
    # Save samples
    if 'LOCAL_RANK' not in os.environ or int(os.environ['LOCAL_RANK']) == 0:
        for i, sample in enumerate(samples):
            sample = samples[i].unsqueeze(0)
            save_videos_grid(sample, save_path, fps=24)
            logger.info(f'Sample save to: {save_path}')

