set -e
gpu=$1


cmd="python extract_frame_clip.py
--model_name=clip-vit-large-patch14 
--gpu=$gpu"


echo "\n-------------------------------------------------------------------------------------"
echo "Execute command: $cmd"
echo "-------------------------------------------------------------------------------------\n"
echo $cmd | sh

