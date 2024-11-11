from PyFaceDet import facedetectcnn
import cv2
import os
import random
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def is_sharp(image, threshold):
    """检查图像是否清晰"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()
    return variance > threshold

def clip_face_from_frame(faces, output_folder, frame_count, frame):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # 筛选出得分最高的人脸
    if faces:
        max_confidence = 90
        best_face = None
        for face in faces:
            x, y, L, W, confidence, angle = face
            if confidence > max_confidence:
                max_confidence = confidence
                best_face = face
        if best_face:
            x, y, L, W, confidence, angle = best_face
            face_region = frame[y:y+W, x:x+L]
            if face_region is not None and face_region.size > 0:
                cv2.imwrite(f'{output_folder}/frame_{frame_count}.jpg', face_region)
            else:
                return False
        return True  # 成功保存人脸
    return False  # 没有检测到人脸

def process_video(file, folder_video):
    video_path = os.path.join(folder_video, file)
    if not video_path.lower().endswith(('.mp4', '.avi', '.flv', '.mov', '.wmv')):  # 如果不是视频文件则跳过
        return

    # 打开视频文件
    video_capture = cv2.VideoCapture(video_path)
    video_name = os.path.basename(video_path)
    fwe = os.path.splitext(video_name)[0]

    # 检查视频是否成功打开
    if not video_capture.isOpened():
        print(f"Error: Could not open video {video_path}.")
        return

    # 设置输出文件夹路径为视频同名文件夹
    output_folder = f'/home/zongtianyu/data/wangyuanxiang/xyy/reproduce/Track2/English/traVal_face_frame/{fwe}'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 检查文件夹是否已有至少4张图片
    existing_images = [f for f in os.listdir(output_folder) if f.endswith('.jpg')]
    if len(existing_images) >= 1:
        return  # 如果文件夹已有至少4张图片，跳过此视频的处理
    else:
        print(f"{video_name}: has less 1 images.")

    # 初始化帧计数器和成功保存人脸计数
    frame_count = 0
    face_detected_frames = []
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))  # 获取视频总帧数

    while len(face_detected_frames) < 4:
        # 读取视频流的下一帧
        ret, frame = video_capture.read()
        if not ret:
            break

        # 检测人脸
        if frame is not None:
            faces = facedetectcnn.facedetect_cnn(frame)
            if faces:
                # 保存包含人脸的帧
                if clip_face_from_frame(faces, output_folder, frame_count, frame):
                    face_detected_frames.append(frame_count)  # 记录成功检测到人脸的帧

        # 更新帧计数器
        frame_count += 1
    # 如果检测到的人脸帧不足4帧，从剩余帧中随机选择
    face_detected_frames = [f for f in os.listdir(output_folder) if f.endswith('.jpg')]
    while len(face_detected_frames) < 4:
        remaining_frames_needed = 4 - len(face_detected_frames)
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 重置视频到开头
        random_frames = random.sample(range(total_frames), remaining_frames_needed)
        print(f"需要采集的图片数量:{remaining_frames_needed}")
        for frame_index in random_frames:
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = video_capture.read()
            if frame is not None:
                suc = cv2.imwrite(f'{output_folder}/frame_{frame_count + 1}.jpg', frame)
                print(os.path.exists(f'{output_folder}/frame_{frame_count + 1}.jpg'))
                frame_count += 1
                face_detected_frames.append(frame_count)
                print(face_detected_frames)

    # 释放视频捕获对象
    video_capture.release()
    cv2.destroyAllWindows()

# 读取视频文件夹和CSV
folder_video = "/home/zongtianyu/data/wangyuanxiang/xyy/reproduce/Track2/English/Videos"
csv_path = "/home/zongtianyu/data/wangyuanxiang/xyy/reproduce/Track2/English/transcription.csv"
video_list = pd.read_csv(csv_path)["FileName"].tolist()

# 使用多线程处理视频
with ThreadPoolExecutor(max_workers=16) as executor:  # 根据系统配置调整 max_workers
    # 使用 tqdm 进度条显示进度
    list(tqdm(executor.map(lambda file: process_video(file, folder_video), video_list), total=len(video_list)))
