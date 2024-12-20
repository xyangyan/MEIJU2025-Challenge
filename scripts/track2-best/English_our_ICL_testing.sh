set -e
gpu=$1


cmd="python test_baseline.py --model=our
--checkpoints_dir=./checkpoints-best --gpu_ids=$gpu
--input_dim_a=1280 --embd_size_a=128 --embd_method_a=maxpool
--input_dim_v=768 --embd_size_v=128  --embd_method_v=maxpool
--input_dim_l=4096 --embd_size_l=128  --hidden_size=128
--A_type=wavlm-large-FRA --V_type=clip-vit-large-patch14-FRA-org-0.5773 --L_type=baichuan2-7b-base-4-FRA
--num_thread=8 --corpus=MEIJU_English
--emo_output_dim=7 --int_output_dim=8 --track=2
--cls_layers=128,64 --dropout_rate=0.2 --use_ICL=True
--batch_size=32 --lr=2e-4 --weight_decay=1e-5
--name=our_English_ICL_run_3_1
--cvNo=1"

# --name=our_English_ICL_run_4_1

echo "\n-------------------------------------------------------------------------------------"
echo "Execute command: $cmd"
echo "-------------------------------------------------------------------------------------\n"
echo $cmd | sh

