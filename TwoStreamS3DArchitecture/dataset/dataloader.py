import os
import torch
import random
import argparse
import cv2 as cv
import numpy as np

from torchvision import transforms
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedShuffleSplit

class VideoDataset(Dataset):
    
    def __init__(self, root_path: str, num_frames: int, phase: str="train") -> None:
        self.root_path = root_path
        self.label_path = os.path.join(root_path, "label.txt")
        self.num_frames = num_frames
        self.phase = phase

        self.vocab_dict = self._load_vocab()
        self.data_dict = self._load_data()

        all_video_paths = self.data_dict["train"][0] + self.data_dict["val"][0] + self.data_dict["test"][0]
        all_labels = self.data_dict["train"][1] + self.data_dict["val"][1] + self.data_dict["test"][1]

        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
        train_indices, test_indices = next(sss.split(all_video_paths, all_labels))

        if phase == "train":
            self.data = ([all_video_paths[i] for i in train_indices], [all_labels[i] for i in train_indices])
        else:
            self.data = ([all_video_paths[i] for i in test_indices], [all_labels[i] for i in test_indices])

    def _load_vocab(self):
        vocab_dict = {}
        with open(self.label_path, 'r') as file:
            for count, vocab in enumerate(file.read().splitlines()):
                vocab_dict[vocab] = count
        return vocab_dict

    def _load_data(self):
        data_dict = {"train": ([], []), "test": ([], []), "val": ([], [])}
        for phase in ["train", "test", "val"]:
            phase_path = os.path.join(self.root_path, phase)
            for label in os.listdir(phase_path):
                label_path = os.path.join(phase_path, label)
                video_paths = [os.path.join(label_path, video) for video in os.listdir(label_path)]
                labels = [self.vocab_dict[label]] * len(video_paths)
                data_dict[phase][0].extend(video_paths)
                data_dict[phase][1].extend(labels)
        return data_dict

    def _preprocess_frame(self, frame: np.array) -> np.array:
        frame = cv.resize(frame.astype(np.float32) / 255.0, (224, 224))
        return frame

    def _load_and_preprocess_frames(self, path: str, num_frames: int) -> torch.tensor:
        frame_list = sorted(os.listdir(path), key=lambda x: int(x.split('_')[-1].split('.')[0]))
        if len(frame_list) > num_frames:
            frame_list = random.sample(frame_list, num_frames)
            frame_list.sort()

        frames = [self._preprocess_frame(cv.imread(os.path.join(path, frame))) for frame in frame_list]
        return torch.tensor(np.array(frames)).permute(0, 3, 1, 2)

    def get_frames(self, video_path: str) -> torch.tensor:
        rgb_frames = self._load_and_preprocess_frames(os.path.join(video_path, "rgb"), self.num_frames)
        kp_frames = self._load_and_preprocess_frames(os.path.join(video_path, "kp"), self.num_frames)
        return rgb_frames, kp_frames

    def __len__(self) -> int:
        return len(self.data[0])

    def __getitem__(self, index: int) -> torch.Tensor:
        video_path = self.data[0][index]
        label = self.data[1][index]
        rgb_frames, kp_frames = self.get_frames(video_path)

        if self.phase == "train" and random.random() < 0.5:
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomRotation(degrees=(0, 15)),
                transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                transforms.RandomPerspective(distortion_scale=0.7, p=0.7),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        rgb_frames = torch.stack([transform(frame.permute(1, 2, 0).numpy()) for frame in rgb_frames])
        kp_frames = torch.stack([transform(frame.permute(1, 2, 0).numpy()) for frame in kp_frames])

        rgb_frames = rgb_frames.permute(1, 0, 2, 3)
        kp_frames = kp_frames.permute(1, 0, 2, 3)

        return rgb_frames, kp_frames, label