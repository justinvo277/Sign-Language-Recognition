import os
import shutil
import argparse
import cv2 as cv
import numpy as np

from utils import interpolate_frames, select_random_frames


def augmentation_video(video_path: str, video_name: str, dir_label_path: str, target_frames: int=32):
    
    cap = cv.VideoCapture(video_path)
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

    new_video_name = None
    dir_save_path = None

    if total_frames < target_frames:
        new_video_name = video_name + "_interpolate.mp4"
        dir_save_path = os.path.join(dir_label_path, new_video_name)
        interpolate_frames(video_path=video_path, output_path=dir_save_path, target_frames=target_frames)
    else:
        num_video_aug = (total_frames // target_frames) + 1
        for idx in range(num_video_aug):
            new_video_name = f"{video_name}_{str(idx + 1)}_select_random.mp4"
            dir_save_path = os.path.join(dir_label_path, new_video_name)
            select_random_frames(video_path=video_path, output_path=dir_save_path, target_frames=target_frames)
        new_video_name =  f"{video_name}_copy.mp4"
        dir_save_path = os.path.join(dir_label_path, new_video_name)
        shutil.copy(video_path, dir_save_path)


def augmentation_dataset(root_folder: str, dir_folder: str, target_frames: int) -> None:

    if not os.path.exists(dir_folder):
        os.makedirs(dir_folder)

    for phase in os.listdir(root_folder):

        phase_path = os.path.join(root_folder, phase)
        dir_phase_path = os.path.join(dir_folder, phase)

        if not os.path.exists(dir_phase_path):
            os.makedirs(dir_phase_path)

        for label in os.listdir(phase_path):
            print(f"Processing in {label} folder in {phase} folder !!")
            label_path = os.path.join(phase_path, label)
            dir_label_path = os.path.join(dir_phase_path, label)

            if not os.path.exists(dir_label_path):
                os.makedirs(dir_label_path)

            for video in os.listdir(label_path):
                video_path = os.path.join(label_path, video)
                augmentation_video(video_path=video_path, video_name=video[:-4], dir_label_path=dir_label_path, target_frames=target_frames)
            print(f"Done {label} folder !!")
            print("\n")
