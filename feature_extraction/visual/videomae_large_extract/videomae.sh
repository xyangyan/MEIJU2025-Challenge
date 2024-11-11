set -e
gpu=$1


cmd="python extract_frame_videomae.py
--model_name=videomae-large 
--gpu=$gpu"


echo "\n-------------------------------------------------------------------------------------"
echo "Execute command: $cmd"
echo "-------------------------------------------------------------------------------------\n"
echo $cmd | sh

