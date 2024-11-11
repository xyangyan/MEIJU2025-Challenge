set -e
run_idx=$1
gpu=$2

for i in `seq 1`;
do

cmd="python train_baseline.py --dataset_mode=multimodal --model=pretrain
--log_dir=./logs --checkpoints_dir=./checkpoints --gpu_ids=$gpu
--input_dim_a=1024 --embd_size_a=128 --embd_method_a=maxpool
--input_dim_v=768 --embd_size_v=128  --embd_method_v=maxpool
--input_dim_l=4096 --embd_size_l=128  --hidden_size=128
--A_type=wavlm-large-FRA --V_type=clip-vit-large-patch14-FRA-org-0.5773 --L_type=baichuan2-7b-base-4-FRA
--num_thread=8 --corpus=MEIJU_English
--emo_output_dim=7 --int_output_dim=8 --track=2
--ce_weight=1.0 --cls_layers=128,64 --dropout_rate=0.5
--niter=20 --niter_decay=40 --print_freq=10
--batch_size=32 --lr=2e-4 --run_idx=$run_idx --weight_decay=1e-5
--name=pretrain_English --suffix=run_{gpu_ids}_{run_idx}
--cvNo=$i"


echo "\n-------------------------------------------------------------------------------------"
echo "Execute command: $cmd"
echo "-------------------------------------------------------------------------------------\n"
echo $cmd | sh

done
