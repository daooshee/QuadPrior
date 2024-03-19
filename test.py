from cldm.hack import disable_verbosity, enable_sliced_attention
disable_verbosity()

import cv2
import einops
import numpy as np
import torch
import random
import glob
import os
import argparse

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from cldm.model import create_model, load_state_dict
from ldm.models.diffusion.dpm_solver import DPMSolverSampler

# 4090: 14G

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", default='checkpoints/COCO-final.ckpt', type=str)
parser.add_argument("--same_folder", default='output_QuadPrior', type=str)
parser.add_argument("--input_folder", default='test_data', type=str)

parser.add_argument("--use_float16", default=True, type=bool)
parser.add_argument("--save_memory", default=False, type=bool) # Cannot use. Has bugs

if __name__ == '__main__':
    args = parser.parse_args()
    checkpoint_file = args.checkpoint

    if args.save_memory:
        enable_sliced_attention()

    print("====== Load parameters ======")

    # Load pretrained Stable Diffusion v1.5
    model = create_model('./models/cldm_v15.yaml').cpu()

    state_dict = load_state_dict('./models/control_sd15_ini.ckpt', location='cpu')
    new_state_dict = {}
    for s in state_dict:
        if "cond_stage_model.transformer" not in s:
            new_state_dict[s] = state_dict[s]
    model.load_state_dict(new_state_dict)

    # Insert new layers in ControlNet (sorry for the ugliness
    model.add_new_layers()

    # Load trained checkpoint
    state_dict = load_state_dict(checkpoint_file, location='cpu')
    new_state_dict = {}
    for sd_name, sd_param in state_dict.items():
        if '_forward_module.control_model' in sd_name:
            new_state_dict[sd_name.replace('_forward_module.control_model.', '')] = sd_param
    model.control_model.load_state_dict(new_state_dict)

    # Load bypass decoder
    ae_checkpoint = './checkpoints/main-epoch=00-step=7000.ckpt'
    model.change_first_stage(ae_checkpoint)

    print("====== Finish loading parameters ======")

    if args.use_float16:
        model = model.cuda().to(dtype=torch.float16)
    else:
        model = model.cuda()
    diffusion_sampler = DPMSolverSampler(model)

    def process(input_image, prompt="", num_samples=1, image_resolution=512, diffusion_steps=10, guess_mode=False, strength=1.0, scale=9.0, seed=0, eta=0.0):
        with torch.no_grad():
            detected_map = resize_image(HWC3(input_image), image_resolution)
            H, W, C = detected_map.shape

            if args.use_float16:
                control = torch.from_numpy(detected_map.copy()).cuda().to(dtype=torch.float16) / 255.0
            else:
                control = torch.from_numpy(detected_map.copy()).cuda() / 255.0
            control = torch.stack([control for _ in range(num_samples)], dim=0)
            control = einops.rearrange(control, 'b h w c -> b c h w').clone()
            ae_hs = model.encode_first_stage(control*2-1)[1]

            if seed == -1:
                seed = random.randint(0, 65535)
            seed_everything(seed)

            if args.save_memory:
                model.low_vram_shift(is_diffusing=False)

            cond = {"c_concat": [control], "c_crossattn": [model.get_unconditional_conditioning(num_samples)]}
            un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_unconditional_conditioning(num_samples)]}
            shape = (4, H // 8, W // 8)

            if args.save_memory:
                model.low_vram_shift(is_diffusing=True)

            model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
            samples, intermediates = diffusion_sampler.sample(diffusion_steps, num_samples,
                                                        shape, cond, verbose=False, eta=eta,
                                                        unconditional_guidance_scale=scale,
                                                        unconditional_conditioning=un_cond,
                                                        dmp_order=3)

            if args.save_memory:
                model.low_vram_shift(is_diffusing=False)
            
            if args.use_float16:
                x_samples = model.decode_new_first_stage(samples.to(dtype=torch.float16), ae_hs)
            else:
                x_samples = model.decode_new_first_stage(samples, ae_hs)
            x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

            results = [x_samples[i] for i in range(num_samples)]
        return results

    # Low-light Enhancement
    img_list = glob.glob(f"{args.input_folder}/*.*")
    print(f"Find {len(img_list)} files in {args.input_folder}")

    for img_path in img_list:
        save_name = os.path.split(img_path)[1]
        save_name = os.path.splitext(save_name)[0] + ".png"
        save_path = os.path.join(args.same_folder, save_name)

        if os.path.exists(save_path):
            print(f"Exists {save_path}, skip.")
            continue

        input_image = cv2.imread(img_path)
        # if you set num_samples > 1, process will return multiple results
        output = process(input_image, num_samples=1)[0]

        os.makedirs(args.same_folder, exist_ok=True)
        cv2.imwrite(save_path, output)