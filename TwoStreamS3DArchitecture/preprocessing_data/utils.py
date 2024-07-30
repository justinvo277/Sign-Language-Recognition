import os
import random
import cv2 as cv
import numpy as np

from ultralytics import YOLO

def check_video(video_path: str) -> int:
        cap = cv.VideoCapture(video_path)
        if cap.isOpened() == 1:
            cap.release()
            return 1
        cap.release()
        return 0

def interpolate_frames(video_path: str, output_path: str, target_frames: int) -> None:

    cap = cv.VideoCapture(video_path)
    original_frames = []
    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv.CAP_PROP_FPS))

    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        original_frames.append(frame)
    cap.release()

    N = len(original_frames)
    if target_frames <= N:
        return
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    for i in range(target_frames):
        alpha = i * (N - 1) / (target_frames - 1)
        lower = int(np.floor(alpha))
        upper = int(np.ceil(alpha))
        if lower == upper:
            frame = original_frames[lower]
        else:
            weight_upper = alpha - lower
            weight_lower = 1 - weight_upper
            frame = cv.addWeighted(original_frames[lower], weight_lower, original_frames[upper], weight_upper, 0)
        out.write(frame)
    out.release()

def select_random_frames(video_path: str, output_path: str, target_frames: int) -> None:

    cap = cv.VideoCapture(video_path)
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    fps = int(cap.get(cv.CAP_PROP_FPS))

    if target_frames > total_frames:
        return

    random_frames = sorted(random.sample(range(total_frames), target_frames))

    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    current_frame = 0
    frame_index = 0

    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        if current_frame == random_frames[frame_index]:
            out.write(frame)
            frame_index += 1
            if frame_index >= target_frames:
                break
        current_frame += 1

    cap.release()

def human_detection(root_video_path: str, dir_video_path: str) -> None:

    model = YOLO('yolov8n.pt') 

    cap = cv.VideoCapture(root_video_path)

    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv.CAP_PROP_FPS))
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    out = cv.VideoWriter(dir_video_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].tolist() if box.conf is not None else 0
                cls = box.cls[0].tolist() if box.cls is not None else -1
                label = model.names[int(cls)]
                if label == 'person' and conf >= 0.89: 
                    out.write(frame)
                else:
                    print(root_video_path)


def calculate_frame_difference(frame1: np.array, frame2: np.array) -> float:
    diff = cv.absdiff(frame1, frame2)
    gray = cv.cvtColor(diff, cv.COLOR_BGR2GRAY)
    _, thresh = cv.threshold(gray, 30, 255, cv.THRESH_BINARY)
    return np.sum(thresh)