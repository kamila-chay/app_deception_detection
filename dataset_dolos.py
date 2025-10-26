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
        self.info = pd.read_excel(info)
        self.folder = folder
        self.subjects = rem_duplicates(self.info["Participants name"].tolist())
        self.subject_counts = Counter(self.info["Participants name"].tolist())

    def iter_subjects(self):
        for subject in self.subjects:
            test_indices = self.info.index[self.info["Participants name"] == subject]
            temp_subjects = [s for s in self.subjects if s != subject]
            val_subject = random.choice(temp_subjects)
            while self.subject_counts[val_subject] not in {4, 5, 6, 7}: # optimal for validation
                val_subject = random.choice(list(temp_subjects))
            val_indices = self.info.index[self.info["Participants name"] == val_subject]
            train_indices = self.info.index[(self.info["Participants name"] != subject) & (self.info["Participants name"] != val_subject)]
            yield Subset(self, train_indices.tolist()), Subset(self, val_indices.tolist()), Subset(self, test_indices.tolist())

    def __len__(self):
        return len(self.info)

    def __getitem__(self, index):
        filename = self.info.loc[index, "Filename"]
        filepath = self.folder / "video" / f"{filename}.mp4"
        labelpath = self.folder / "gen_labels" / f"{filename}.txt"
        with open(labelpath, "r") as f:
            label = f.read()
        return create_conv_template(filepath), create_conv_template(filepath, completion=label), 