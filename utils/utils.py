import random

import cv2
import numpy as np
import torch
from PIL import Image


def rem_duplicates(orig):
    res = []
    res_set = set()
    for item in orig:
        if item not in res_set:
            res.append(item)
            res_set.add(item)
    return


def sample_frames_uniformly(video_path, num_samples=8):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []

    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        frames.append(Image.fromarray(frame))

    cap.release()

    indices = np.linspace(0, len(frames) - 1, num_samples).astype(int)
    sampled_frames = [frames[index] for index in indices]
    return sampled_frames


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def overlay_attention(attn_map, pixel_values, processor, index):
    mean = torch.tensor(processor.image_mean).reshape(3, 1, 1)
    std = torch.tensor(processor.image_std).reshape(3, 1, 1)
    attn_resized = cv2.resize(attn_map[index], (224, 224))
    heatmap = cv2.applyColorMap(np.uint8(255 * attn_resized), cv2.COLORMAP_HOT)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
    img = pixel_values[0, index].cpu() * std + mean
    img = torch.clamp(img, 0, 1)
    img = img.permute(1, 2, 0).numpy()
    img_gray = np.dot(img, [0.2989, 0.5870, 0.1140])
    img_gray = np.stack([img_gray] * 3, axis=-1)
    overlay = 0.5 * img_gray + 0.5 * heatmap
    overlay = np.clip(overlay, 0, 1)

    return overlay


def roll_out_attn_map(attentions, num_attn_maps, patch_num_height, patch_num_width):
    R = (
        torch.eye(patch_num_height * patch_num_width + 1, device=attentions[0].device)
        .unsqueeze(0)
        .repeat(num_attn_maps, 1, 1)
    )
    for i in range(11, -1, -1):
        A = attentions[i].mean(1)
        A = A + torch.eye(A.size(1), device=A.device).unsqueeze(0).repeat(
            num_attn_maps, 1, 1
        )
        A = A / A.sum(dim=-1, keepdim=True)
        R = R @ A
    attn_map = R[:, 0, 1:]

    attn_map = attn_map.cpu().numpy()
    attn_map = attn_map.reshape(num_attn_maps, patch_num_height, patch_num_width)
    attn_map = (attn_map - attn_map.min(axis=(1, 2), keepdims=True)) / (
        attn_map.max(axis=(1, 2), keepdims=True)
        - attn_map.min(axis=(1, 2), keepdims=True)
    )
    return attn_map
