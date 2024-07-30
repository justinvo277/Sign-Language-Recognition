import os
import shutil
import argparse
import cv2 as cv
import numpy as np
import mediapipe as mp


from augmentation import augmentation_dataset
from preprocessing import preprocessing_folder


parser = argparse.ArgumentParser(description="Config")

parser.add_argument("--dataset_name", type=str, help="Dataset name", default="WLASL100")
parser.add_argument("--target_frames", type=int, help="Frames each video", default=32)
parser.add_argument("--root_path", type=str, help="Folder dataset", default="/home/hieu/dev/SLData/SLR/dataset/WLASL_100")
parser.add_argument("--dir_path", type=str, help="Folder save dât after preprocessing", default="/home/hieu/dev/SLData/SLR")

args = parser.parse_args()


def check_video(video_path: str) -> int:
        cap = cv.VideoCapture(video_path)
        if cap.isOpened() == 1:
            cap.release()
            return 1
        cap.release()
        return 0


def preprocessing_video(video_path: str, video_name: str, save_folder_path: str) -> None:

    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    rgb_save_folder_path = os.path.join(save_folder_path, "rgb")
    kp_save_folder_path = os.path.join(save_folder_path, "kp") #kp = keypoint;

    if not os.path.exists(rgb_save_folder_path):
        os.makedirs(rgb_save_folder_path)
    if not os.path.exists(kp_save_folder_path):
        os.makedirs(kp_save_folder_path)

    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection, \
                mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose_detection:
            
            cap = cv.VideoCapture(video_path)

            count_frame = 0
            num_count_frame = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
            selected_frames = [i for i in range(num_count_frame)]

            for frame_index in selected_frames:

                cap.set(cv.CAP_PROP_POS_FRAMES, frame_index)
                success, image = cap.read()

                if image is None:
                    break

                image_tmp = image.copy()

                image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
                w, h, c = image_rgb.shape
                image_black = np.zeros((w, h, c), dtype=np.uint8)

                face_results = face_detection.process(image_rgb)
                if face_results.detections:
                    for detection in face_results.detections:
                        mp_drawing.draw_detection(image, detection)

                pose_results = pose_detection.process(image_rgb)
                if pose_results.pose_landmarks:
                    mp_drawing.draw_landmarks(image_black, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                    connections = [
                        (mp_pose.PoseLandmark.NOSE, mp_pose.PoseLandmark.LEFT_SHOULDER),
                        (mp_pose.PoseLandmark.NOSE, mp_pose.PoseLandmark.RIGHT_SHOULDER)]

                    for connection in connections:
                        start_point = pose_results.pose_landmarks.landmark[connection[0]]
                        end_point = pose_results.pose_landmarks.landmark[connection[1]]
                        start_x, start_y = int(start_point.x * image.shape[1]), int(start_point.y * image.shape[0])
                        end_x, end_y = int(end_point.x * image.shape[1]), int(end_point.y * image.shape[0])
                        cv.line(image_black, (start_x, start_y), (end_x, end_y), (255, 255, 255), 2)

                image_name = video_name + "_" + str(count_frame) + ".png"
                count_frame += 1

                folder_frame = os.path.join(rgb_save_folder_path, image_name)
                folder_kp = os.path.join(kp_save_folder_path, image_name)

                cv.imwrite(folder_frame, image_tmp)
                cv.imwrite(folder_kp, image_black)

            cap.release()
            cv.destroyAllWindows()

if __name__ == "__main__":

    print("DATA CLEANING PHASE !!!")
    data_name_pre = args.dataset_name + "_preprocessing"
    dir_preprocessing_path = os.path.join(args.dir_path, data_name_pre)
    if not os.path.exists(dir_preprocessing_path):
        os.makedirs(dir_preprocessing_path)
    preprocessing_folder(root_folder=args.root_path, dir_folder=dir_preprocessing_path)
    print("\n")

    print("DATA AUGMENT PHASE !!!")
    data_name_aug = args.dataset_name + "_augmentation"
    dir_augmentation_path = os.path.join(args.dir_path, data_name_aug)
    if not os.path.exists(dir_augmentation_path):
        os.makedirs(dir_augmentation_path)
    augmentation_dataset(root_folder=dir_preprocessing_path, dir_folder=dir_augmentation_path, target_frames=args.target_frames)
    print("\n")


    print("VIDEO TO FRAMES PHASE !!!")
    data_train_name = args.dataset_name + "_train"
    new_dataset = os.path.join(args.dir_path, data_train_name)

    if not os.path.exists(new_dataset):
        os.makedirs(new_dataset)

    label_path = os.path.join(new_dataset, "label.txt")
    label_data_path = os.path.join(args.root_path, "train")
    label_lst = []

    for label in os.listdir(label_data_path):
        label_lst.append(label)

    with open(label_path, 'w') as file:
        for label in label_lst:
            file.write(label)
            file.write("\n")

    for data_type in ["train", "test", "val"]:
         
        data_path = os.path.join(dir_augmentation_path, data_type)
        save_path = os.path.join(new_dataset, data_type)

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for label in os.listdir(data_path):
            
            label_path = os.path.join(data_path, label)
            new_label_path = os.path.join(save_path, label)

            if not os.path.exists(new_label_path):
                os.makedirs(new_label_path)

            for video in os.listdir(label_path):

                video_path = os.path.join(label_path, video)
                new_video_path = os.path.join(new_label_path, video.split(".")[0])

                if not os.path.exists(new_video_path):
                    os.makedirs(new_video_path)

                if check_video(video_path):
                    preprocessing_video(video_path, video.split(".")[0], new_video_path)

    shutil.rmtree(dir_preprocessing_path)
    shutil.rmtree(dir_augmentation_path)
