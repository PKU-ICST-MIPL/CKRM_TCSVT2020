export PYTHONPATH=/home/wenzhang/CKRM
export LD_LIBRARY_PATH=/usr/local/cuda-9.0/
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
# Evaluate on VCR validation set
python3.6 eval_q2ar.py 
# Get VCR testing set prediction result
#python3.6 eval_for_leaderboard.py

