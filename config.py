# *_*coding:utf-8 *_*
import os
import sys
import socket


############ For LINUX ##############
DATA_DIR = {
	# 'Track1_English_Annotation': r'G:\数据集\ChallengeData\Track1\English\Annotation',
	# 'Track1_English_NoAnnotation': r'G:\数据集\ChallengeData\Track1\English\NoAnnotation',
	# 'Track1_Mandarin_Annotation': r'G:\数据集\ChallengeData\Track1\Mandarin\Annotation',
	# 'Track1_Mandarin_NoAnnotation': r'G:\数据集\ChallengeData\Track1\Mandarin\NoAnnotation',
	'Track2_English': r'/home/ccip/data/wangyuanxiang/xyy/reproduce/Track2/English',
	# 'Track2_Mandarin': r'G:\数据集\ChallengeData\Track2\Mandarin',
}

PATH_TO_RAW_AUDIO = {
	# 'Track1_English_Annotation': os.path.join(DATA_DIR['Track1_English_Annotation'], 'Audios'),
	# 'Track1_English_NoAnnotation': os.path.join(DATA_DIR['Track1_English_NoAnnotation'], 'Audios'),
	# 'Track1_Mandarin_Annotation': os.path.join(DATA_DIR['Track1_Mandarin_Annotation'], 'Audios'),
	# 'Track1_Mandarin_NoAnnotation': os.path.join(DATA_DIR['Track1_Mandarin_NoAnnotation'], 'Audios'),
    
    # Training and Validation
	'Track2_English': os.path.join(DATA_DIR['Track2_English'], 'TraVal_Audios_resampled_16000'),
    # Testing
	# 'Track2_English': os.path.join(DATA_DIR['Track2_English'], 'Testing_Audios_resampled_16000'),

	# 'Track2_Mandarin': os.path.join(DATA_DIR['Track2_Mandarin'], 'Audios'),
}

PATH_TO_RAW_FACE = {
	# 'Track1_English_Annotation': os.path.join(DATA_DIR['Track1_English_Annotation'], 'openface_face'),
	# 'Track1_English_NoAnnotation': os.path.join(DATA_DIR['Track1_English_NoAnnotation'], 'openface_face'),
	# 'Track1_Mandarin_Annotation': os.path.join(DATA_DIR['Track1_Mandarin_Annotation'], 'openface_face'),
	# 'Track1_Mandarin_NoAnnotation': os.path.join(DATA_DIR['Track1_Mandarin_NoAnnotation'], 'openface_face'),
	# 'Track2_English': os.path.join(DATA_DIR['Track2_English'], 'openface_face'),
    
	# tra and val data
    # use frame0
    'Track2_English': os.path.join(DATA_DIR['Track2_English'], 'TraVal_video_frame_English_10'),
    'Track2_English_trainval': os.path.join(DATA_DIR['Track2_English'], 'TraVal_video_frame_English_10'),
    'Track2_English_test': os.path.join(DATA_DIR['Track2_English'], 'testing_video_frame_English_10'),
    # test data
    # 'Track2_English': os.path.join(DATA_DIR['Track2_English'], 'testing_video_frame_English_10'),   # frame0

	

	# 'Track2_Mandarin': os.path.join(DATA_DIR['Track2_Mandarin'], 'openface_face'),
}

PATH_TO_TRANSCRIPTIONS = {
	# 'MER2023': os.path.join(DATA_DIR['MER2023'], 'transcription.csv'),
	# 'Track1_English_Annotation': os.path.join(DATA_DIR['Track1_English_Annotation'], 'transcription.csv'),
	# 'Track1_English_NoAnnotation': os.path.join(DATA_DIR['Track1_English_NoAnnotation'], 'transcription.csv'),
	# 'Track1_Mandarin_Annotation': os.path.join(DATA_DIR['Track1_Mandarin_Annotation'], 'transcription.csv'),
	# 'Track1_Mandarin_NoAnnotation': os.path.join(DATA_DIR['Track1_Mandarin_NoAnnotation'], 'transcription.csv'),
    
    # Training and Validation
	'Track2_English': os.path.join(DATA_DIR['Track2_English'], 'transcription.csv'),
    # Testing
	# 'Track2_English': os.path.join(DATA_DIR['Track2_English'], 'Testing_transcription.csv'),

	# 'Track2_Mandarin': os.path.join(DATA_DIR['Track2_Mandarin'], 'transcription.csv'),
}
PATH_TO_FEATURES = {
	# 'MER2023': r'G:\数据集\IEMOCAP\RoBERTa'  #os.path.join(DATA_DIR['MER2023'], 'features'),
	# 'Track1_English_Annotation': os.path.join(DATA_DIR['Track1_English_Annotation'], 'features'),
	# 'Track1_English_NoAnnotation': os.path.join(DATA_DIR['Track1_English_NoAnnotation'], 'features'),
	# 'Track1_Mandarin_Annotation': os.path.join(DATA_DIR['Track1_Mandarin_Annotation'], 'features'),
	# 'Track1_Mandarin_NoAnnotation': os.path.join(DATA_DIR['Track1_Mandarin_NoAnnotation'], 'features'),
	'Track2_English': os.path.join(DATA_DIR['Track2_English'], 'features'),
	'Track2_English_trainval': os.path.join(DATA_DIR['Track2_English'], 'features'),
	'Track2_English_test': os.path.join(DATA_DIR['Track2_English'], 'features'),
	# 'Track2_Mandarin': os.path.join(DATA_DIR['Track2_Mandarin'], 'features'),
}
# PATH_TO_LABEL = {
# 	'MER2023': os.path.join(DATA_DIR['MER2023'], 'label-6way.npz'),
# }

PATH_TO_PRETRAINED_MODELS = r'/home/ccip/data/wangyuanxiang/xyy/reproduce/all_models'
PATH_TO_OPENSMILE = os.path.join(PATH_TO_PRETRAINED_MODELS, r'opensmile-2.3.0')
PATH_TO_FFMPEG = os.path.join(PATH_TO_PRETRAINED_MODELS, r'ffmpeg-4.4.1-i686-static', r'ffmpeg')
PATH_TO_NOISE = os.path.join(PATH_TO_PRETRAINED_MODELS, r'musan', r'audio-select')

SAVED_ROOT = os.path.join('./saved')
DATA_DIR = os.path.join(SAVED_ROOT, 'data')
MODEL_DIR = os.path.join(SAVED_ROOT, 'model')
LOG_DIR = os.path.join(SAVED_ROOT, 'log')
PREDICTION_DIR = os.path.join(SAVED_ROOT, 'prediction')
FUSION_DIR = os.path.join(SAVED_ROOT, 'fusion')
SUBMISSION_DIR = os.path.join(SAVED_ROOT, 'submission')
