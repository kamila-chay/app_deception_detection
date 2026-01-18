# Developed as part of a BSc thesis at the Faculty of Computer Science, Bialystok Univesity of Technology

import os
import random

import cv2
import numpy as np
import torch
from PIL import Image


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
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def concatenate_token_ids(token_ids1, token_ids2, pad_token_id):
    if token_ids1.size(1) == token_ids2.size(1):
        return torch.concat([token_ids1, token_ids2], dim=0)
    pad_token_id = torch.tensor([[pad_token_id]], device=token_ids1.device)
    if token_ids1.size(1) > token_ids2.size(1):
        extra_padding = pad_token_id.repeat(
            token_ids2.size(0), token_ids1.size(1) - token_ids2.size(1)
        )
        token_ids2 = torch.concat([token_ids2, extra_padding], dim=1)
    else:
        extra_padding = pad_token_id.repeat(
            token_ids1.size(0), token_ids2.size(1) - token_ids1.size(1)
        )
        token_ids1 = torch.concat([token_ids1, extra_padding], dim=1)

    return torch.concat([token_ids1, token_ids2], dim=0)


def make_conversation_for_separate_configuration(video_path, *args, completion=None):
    return (
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "url": str(video_path),
                    },
                    {
                        "type": "text",
                        "text": "Explain different cues to deception/truthfulness in this video and how they could be interpreted.",
                    },
                ],
            }
        ]
        if not completion
        else [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "url": str(video_path),
                    },
                    {
                        "type": "text",
                        "text": "Explain different cues to deception/truthfulness in this video and how they could be interpreted.",
                    },
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": completion}]},
        ]
    )
