# Developed as part of a BSc thesis at the Faculty of Computer Science, Bialystok Univesity of Technology

import json
import os
import random

import pandas as pd
import torch
from torch.utils.data import Dataset

from thesis.utils.utils import sample_frames_uniformly


def make_conversation_for_joint_configuration(video_path, completion=""):
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
                        "text": "Would you say that the person in the video is lying or telling the truth? Explain your reasoning.",
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
                        "text": "Would you say that the person in the video is lying or telling the truth? Reason and arrive at a tentatve conclusion even if it's not conclusive.",
                    },
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": completion}]},
        ]
    )


class DolosDataset(Dataset):
    def __init__(
        self,
        info,
        folder,
        label_folder="joint_configuration_reasoning_labels",
        conversation_making_func=make_conversation_for_joint_configuration,
    ):
        self.info = pd.read_csv(info, header=None)
        self.folder = folder
        self.label_folder = label_folder
        self.include_raw_cues = False
        self.include_opposing = False
        self.conversation_making_func = conversation_making_func

    def __len__(self):
        return len(self.info)

    def include_raw_cues_(self, value):
        self.include_raw_cues = value

    def include_dispreferred_(self, value):
        self.include_opposing = value

    def __getitem__(self, index):
        filename = self.info.iloc[index, 0]
        filepath = self.folder / "video" / f"{filename}.mp4"
        labelpath = self.folder / self.label_folder / f"{filename}.txt"

        with open(labelpath, "r") as f:
            label = f.read()

        ret_values = (
            self.conversation_making_func(filepath),
            self.conversation_making_func(filepath, completion=label),
        )

        if self.include_raw_cues:
            with open(
                self.folder / self.label_folder / f"{filename}_raw_cues.json", "r"
            ) as f:
                raw_cues = json.load(f)
            ret_values = (*ret_values, raw_cues)

        if self.include_opposing:
            with open(
                self.folder / self.label_folder / f"{filename}_opposing.txt", "r"
            ) as f:
                opposing_label = f.read()
            ret_values = (
                *ret_values,
                self.conversation_making_func(
                    filepath, completion=opposing_label
                ),
            )
        return ret_values


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
