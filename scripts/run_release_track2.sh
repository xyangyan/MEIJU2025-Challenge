
########################################################################
######################## step1: dataset preprocess #####################
########################################################################
### Processing training set and validation set

python feature_extraction_main.py normalize_dataset_format --data_root='/home/ccip/data/wangyuanxiang/xyy/ChallengeData/Track2/English' --save_root='/home/ccip/data/wangyuanxiang/xyy/reproduce/Track2/English' --track=2

### Processing test set
#python feature_extraction_main.py normalize_dataset_format --data_root='G:\数据集\ChallengeData\Track1\English' --save_root='G:\数据集\ChallengeData\Track1\English' --isTest=True

############################################################################
################# step2: multimodal feature extraction #####################
# you can also extract utterance-level features setting --feature_level='UTTERANCE'#
############################################################################
## visual feature extraction
cd feature_extraction/visual
/home/ccip/data/wangyuanxiang/xyy/reproduce/MEIJU2025-baseline-master/feature_extraction/visual/extract_frame.py
# python -u extract_ferplus_embedding.py  --dataset=Track2_English --feature_level='UTTERANCE' --model_name='resnet50_ferplus_dag' --gpu=0



## acoustic feature extraction
cd feature_extraction/audio
#python -u extract_wav2vec_embedding.py       --dataset=MEIJU --feature_level=UTTERANCE --gpu=0
python -u extract_wav2vec_embedding.py       --dataset=Track2_English --feature_level=FRAME --gpu=0



## lexical feature extraction
cd feature_extraction/text
python extract_text_embedding_LZ.py --dataset=Track2_English --feature_level=FRAME --model_name=roberta-base             --gpu=0

