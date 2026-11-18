import numpy as np
from torchvision.transforms.functional import to_tensor
from infinity.utils.dynamic_resolution import dynamic_resolution_h_w, h_div_w_templates
import PIL.Image as PImage
import torch
from typing import List
import torch.nn.functional as F
from infinity.models.basic import CrossAttnBlock
from infinity.models.sampling import sample_with_top_k_top_p_also_inplace_modifying_logits_
from tools.run_infinity import encode_prompt


def transform(pil_img, tgt_h, tgt_w):
    width, height = pil_img.size
    if width / height <= tgt_w / tgt_h:
        resized_width = tgt_w
        resized_height = int(tgt_w / (width / height))
    else:
        resized_height = tgt_h
        resized_width = int((width / height) * tgt_h)
    pil_img = pil_img.resize((resized_width, resized_height), resample=PImage.LANCZOS)
    # crop the center out
    arr = np.array(pil_img)
    crop_y = (arr.shape[0] - tgt_h) // 2
    crop_x = (arr.shape[1] - tgt_w) // 2
    im = to_tensor(arr[crop_y: crop_y + tgt_h, crop_x: crop_x + tgt_w])
    return im.add(im).add_(-1)

def load_img(img_path, args):
    # h_div_w = 1.0
    # h_div_w_template = h_div_w_templates[np.argmin(np.abs(h_div_w - h_div_w_templates))]
    with open(img_path, 'rb') as f:
        img: PImage.Image = PImage.open(f)
        img = img.convert('RGB')
        w,h = img.size
        h_div_w = h/w
        h_div_w_template = h_div_w_templates[np.argmin(np.abs(h_div_w - h_div_w_templates))]
        tgt_h, tgt_w = dynamic_resolution_h_w[h_div_w_template][args.pn]['pixel']
        img_B3HW = transform(img, tgt_h, tgt_w)

    return img_B3HW.unsqueeze(0)

def mask_quant(vae, vae_scale_schedule, raw_features, device):
    with torch.amp.autocast('cuda', enabled = False):
        B = raw_features.shape[0]
        if raw_features.dim() == 4:
            codes_out = raw_features.unsqueeze(2) #[bs, vocabulary_size, 1, h, w]
        else:
            codes_out = raw_features
        cum_var_input = 0 #当前编码了的feature，用于计算residual
        gt_all_bit_indices = []
        x_BLC_wo_prefix = []
        for si, (pt, ph, pw, pm) in enumerate(vae_scale_schedule):
            residual = codes_out - cum_var_input
            if si != len(vae_scale_schedule)-1:
                residual = F.interpolate(residual, size=vae_scale_schedule[si][:3], mode=vae.quantizer.z_interplote_down).contiguous()
            quantized, _, bit_indices, loss = vae.quantizer.lfq(residual) # quantized shape: [B, d_vae, 1, h, w], bit_indices shape: [B,1,h,w,d_vae]
            gt_all_bit_indices.append(bit_indices)  
            cum_var_input = cum_var_input + F.interpolate(quantized, size=vae_scale_schedule[-1][:3], mode=vae.quantizer.z_interplote_up).contiguous()
            if si < len(vae_scale_schedule)-1:
                this_scale_input = F.interpolate(cum_var_input, size=vae_scale_schedule[si+1][:3], mode=vae.quantizer.z_interplote_up).contiguous()
                x_BLC_wo_prefix.append(this_scale_input.reshape(*this_scale_input.shape[:2], -1).permute(0,2,1)) # (B,H/2*W/2,4C) or (B,H*W,C)
        gt_ms_idx_Bl = [item.reshape(B, -1, vae.codebook_dim) for item in gt_all_bit_indices]
        x_BLC_wo_prefix = torch.cat(x_BLC_wo_prefix, 1)
    
    return x_BLC_wo_prefix, gt_ms_idx_Bl

