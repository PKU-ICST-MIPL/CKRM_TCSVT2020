export PYTHONPATH=/home/wenzhang/CKRM
export LD_LIBRARY_PATH=/usr/local/cuda-9.0/
## If you train Q->A
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3.6 train.py -params multiatt/default.json -folder saves/flagship_answer

## If you train QA->R
#export CUDA_VISIBLE_DEVICES=4,5,6,7,8,9
#python3.6 train.py -params multiatt/default.json -folder saves/flagship_rationale -rationale


