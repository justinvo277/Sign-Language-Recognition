import os
import argparse
import cv2 as cv

from utils import human_detection, calculate_frame_difference

def drop_frames(video_path: str, threshold: int = 2500) -> None:

    cap = cv.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if len(frames) == 0:
        print(f"Error: No frames were read from video {video_path}. Deleting the file.")
        os.remove(video_path)
        return
    
    diff_lst = [0]
    for frame_idx in range(1, len(frames)):
        diff = calculate_frame_difference(frames[frame_idx-1], frames[frame_idx])
        diff_lst.append(diff)

    unique_frames = [frames[0]]
    for frame_idx in range(1, len(frames)):
        if diff_lst[frame_idx] > threshold:
            unique_frames.append(frames[frame_idx])

    height, width, layers = unique_frames[0].shape
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter(video_path, fourcc, 30, (width, height))

    for frame in unique_frames:
        out.write(frame)

    out.release()
    cv.destroyAllWindows()

def is_video_openable(file_path: str) -> bool:
    cap = cv.VideoCapture(file_path)
    if not cap.isOpened():
        return False
    
    ret, frame = cap.read()
    cap.release()
    return ret

def preprocessing_folder(root_folder: str, dir_folder: str) -> None:

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
                dir_video_path = os.path.join(dir_label_path, video)
                if is_video_openable(video_path):
                    human_detection(root_video_path=video_path, dir_video_path=dir_video_path)
                    drop_frames(video_path=dir_video_path)
                else:
                    print(f"Can not open the video {video_path}")
            print(f"Done {label} folder !!")
            print("\n")