from torch.utils.data import Dataset
import pandas as pd
from utils import sample_frames_uniformly
import os
import torch

def create_conv_template(video_path, completion=""):
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "url": str(video_path),
                },
                {
                    "type": "text", 
                    "text": "Would you say that the person in the video is lying or telling the truth? Explain your reasoning."},
            ],
        } 
    ] if not completion else [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "url": str(video_path),
                },
                {
                    "type": "text", 
                    "text": "Would you say that the person in the video is lying or telling the truth? Explain your reasoning."},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": completion
                }
            ]
        }
    ]

class DolosDataset(Dataset):
    def __init__(self, info, folder):
        self.info = pd.read_csv(info, header=None)
        self.folder = folder

    def __len__(self):
        return len(self.info)

    def __getitem__(self, index):
        filename = self.info.iloc[index, 0]
        filepath = self.folder / "video" / f"{filename}.mp4"
        labelpath = self.folder / "gen_labels" / f"{filename}.txt"
        with open(labelpath, "r") as f:
            label = f.read()
        return create_conv_template(filepath), create_conv_template(filepath, completion=label)
    

class DolosClassificationDataset(Dataset):
    def __init__(self, csv_file, directory, transform):
        self.data = pd.read_csv(csv_file)
        self.directory = directory
        self.transform = transform
        self.curr_iter_idx = -1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        name = row[0]
        label = row[1]
        video = sample_frames_uniformly(os.path.join(self.directory, name + ".mp4"))
        video = self.transform(video, return_tensors="pt")
        def map_label(label):
            label = label.lower().strip()
            if label == "deception" or label == "lie":
                return torch.tensor(0, dtype=torch.long)
            elif label == "truth":
                return torch.tensor(1, dtype=torch.long)
            else:
                raise ValueError(f"Invalid label: {label}")
        return video["pixel_values"][0], map_label(label)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.curr_iter_idx == len(self) - 1:
            self.curr_iter_idx = -1
            raise StopIteration
        self.curr_iter_idx += 1
        return self[self.curr_iter_idx], self.data.iloc[self.curr_iter_idx, 0]