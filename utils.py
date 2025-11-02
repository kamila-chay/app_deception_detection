import cv2
from PIL import Image
import numpy as np
import random
import torch

def rem_duplicates(orig):
    res = []
    res_set = set()
    for item in orig:
        if item not in res_set:
            res.append(item)
            res_set.add(item)
    return 

def sample_frames_uniformly(video_path, num_samples = 8):
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
