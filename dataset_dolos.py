from torch.utils.data import Dataset, Subset
import pandas as pd
from collections import Counter
import random
from utils import rem_duplicates

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
        return create_conv_template(filepath), create_conv_template(filepath, completion=label), 