## track2 ------------------------------------------------------------------------------------------------------------------------

## 第一步：数据格式化
python feature_extraction_main.py normalize_dataset_format --data_root='/home/ccip/data/wangyuanxiang/xyy/ChallengeData/Track2/English' --save_root='/home/ccip/data/wangyuanxiang/xyy/reproduce/Track2/English' --track=2

## 第二步：抽取特征（audio，text， video）
# audio
conda activate emotion
cd feature_extraction/audio
# 抽取audio之前需转换MP4为wav格式，然后设置采样率为16000
python -u extract_wav2vec_embedding.py       --dataset=Track2_English --feature_level=FRAME --gpu=0


# text
cd feature_extraction/text
python extract_text_embedding_LZ.py --dataset=Track2_English --feature_level=FRAME --model_name=roberta-base             --gpu=0


# video
cd feature_extraction/visual
# 使用feature_extraction/visual/extract_frame.py抽帧，并用Resnet50提前特征存为npy格式
python extract_frame_tra_val.py
# python -u extract_ferplus_embedding.py  --dataset=Track2_English --feature_level='UTTERANCE' --model_name='resnet50_ferplus_dag' --gpu=0




### track1 -------------------------------------------------------------------------------------------------------------------------

## 第一步：数据格式化
python feature_extraction_main.py normalize_dataset_format --data_root='/home/ccip/data/wangyuanxiang/xj/Track1_data/English/' \
    --save_root='/home/ccip/data/wangyuanxiang/xyy/reproduce/Track1/English' \
    --track=1


## 第二步：抽取特征（audio，text， video）
# audio
cd feature_extraction/audio
# 抽取audio之前需转换MP4为wav格式，然后设置采样率为16000
python -u extract_wav2vec_embedding.py       --dataset=Track1_English --feature_level=FRAME --gpu=0


# text
cd feature_extraction/text
python extract_text_embedding_LZ.py --dataset=Track2_English --feature_level=FRAME --model_name=roberta-base             --gpu=0


# video
cd feature_extraction/visual
# 使用feature_extraction/visual/extract_frame.py抽帧，并用Resnet50提前特征存为npy格式
python extract_frame_tra_val.py
# python -u extract_ferplus_embedding.py  --dataset=Track2_English --feature_level='UTTERANCE' --model_name='resnet50_ferplus_dag' --gpu=0




