import pickle
import numpy as np
import cv2
import torch
from torchvision.models import resnet50
from torchvision.transforms import transforms
import os
import tqdm
import torch.utils.data as data
import glob
import shutil
import argparse
import numpy as np
from PIL import Image
# import config

class FrameDataset(data.Dataset):
    def __init__(self, vid, face_dir, transform=None):
        super(FrameDataset, self).__init__()
        self.vid = vid
        self.path = os.path.join(face_dir, vid)
        self.transform = transform
        self.frames = self.get_frames()

    def get_frames(self):
        frames = glob.glob(os.path.join(self.path, '*'))
        return frames

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        path = self.frames[index]
        img = Image.open(path)
        if self.transform is not None:
            img = self.transform(img)
        name = os.path.basename(path)[:-4]
        return img, name

def frame_extract(video_path, root_save_path, frame_to_save=5):
    video_name = os.path.basename(video_path)[:-4]
    save_dir = os.path.join(root_save_path, video_name)
    
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    video = cv2.VideoCapture(video_path)
    frame_to_save = 5  # We want to save frame 5 (0-indexed is 4)
    count = 0
    
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        if count == frame_to_save:
            save_path = os.path.join(save_dir, f'frame{frame_to_save}.jpg')
            cv2.imwrite(save_path, frame)
            break  # Exit after saving the desired frame

        count += 1

    video.release()
    cv2.destroyAllWindows()


def extract(data_loader, model):
    model.eval()
    with torch.no_grad():
        features, timestamps = [], []
        for images, names in data_loader:
            images = images.cuda()
            embedding = model(images)
            features.append(embedding.cpu().detach().numpy())
            timestamps.extend(names)
        features, timestamps = np.row_stack(features), np.array(timestamps)
        return features, timestamps


def feature_extract(frame_dir, save_dir, feature_level='FRAME'):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    model = resnet50(pretrained=True).cuda()
    transform = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    vids = os.listdir(frame_dir)
    EMBEDDING_DIM = -1
    print(f'Find total "{len(vids)}" videos.')
    for i, vid in enumerate(vids, 1):
        print(f"Processing video '{vid}' ({i}/{len(vids)})...")
        csv_file = os.path.join(save_dir, f'{vid}.npy')
        if os.path.exists(csv_file):
            continue

        # forward
        dataset = FrameDataset(vid, frame_dir, transform=transform)
        if len(dataset) == 0:
            print("Warning: number of frames of video {} should not be zero.".format(vid))
            embeddings, framenames = [], []
        else:
            data_loader = torch.utils.data.DataLoader(dataset,
                                                      batch_size=32,
                                                      num_workers=4,
                                                      pin_memory=True)
            embeddings, framenames = extract(data_loader, model)

        # save results
        indexes = np.argsort(framenames)
        embeddings = embeddings[indexes]
        EMBEDDING_DIM = max(EMBEDDING_DIM, np.shape(embeddings)[-1])

        if feature_level == 'FRAME':
            embeddings = np.array(embeddings).squeeze()
            if len(embeddings) == 0:
                embeddings = np.zeros((1, EMBEDDING_DIM))
            elif len(embeddings.shape) == 1:
                embeddings = embeddings[np.newaxis, :]
            np.save(csv_file, embeddings)   # shape = (frame_num, 1000)
        else:
            embeddings = np.array(embeddings).squeeze()
            if len(embeddings) == 0:
                embeddings = np.zeros((EMBEDDING_DIM, ))
            elif len(embeddings.shape) == 2:
                embeddings = np.mean(embeddings, axis=0)
            np.save(csv_file, embeddings)



def visual_extraction():

    sample_rate = 5


    English_drama_path = r'/home/zongtianyu/data/wangyuanxiang/xyy/ChallengeData/Track2/English/Testing'
    video_name = os.listdir(English_drama_path)
    for video in tqdm.tqdm(video_name):
        if 'mp4' in video:
            video_path = os.path.join(English_drama_path, video)
            if not os.path.exists(f'/home/ccip/data/wangyuanxiang/xyy/reproduce/Track2/English/reframe_testing_video_English_{sample_rate}'):
                os.mkdir(f'/home/ccip/data/wangyuanxiang/xyy/reproduce/Track2/English/reframe_testing_video_English_{sample_rate}')
            frame_extract(video_path, f'/home/ccip/data/wangyuanxiang/xyy/reproduce/Track2/English/reframe_testing_video_English_{sample_rate}', frame_to_save=sample_rate)
    
    print('Finished extracting English frame!')



    # # resnet50 feature_extract
    # english_frame_dir = r'/home/ccip/data/wangyuanxiang/xyy/reproduce/Track2/English/reframe_testing_video_English_5'
    # save_dir = r'/home/ccip/data/wangyuanxiang/xyy/reproduce/Track2/English/features/resnet50_frame_10_testing'
    # if not os.path.exists(save_dir):
    #     os.mkdir(save_dir)
    # feature_extract(english_frame_dir, save_dir, feature_level='FRAME')






if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run.')
    parser.add_argument('--overwrite', action='store_true', default=True, help='whether overwrite existed feature folder.')
    parser.add_argument('--dataset', type=str, default='BoxOfLies', help='input dataset')
    params = parser.parse_args()

    print(f'==> Extracting openface features...')




    visual_extraction()



    pass
