#! /bin/bash
export MKL_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

source scripts/parallelize.sh
set -u
echo "Starting at $(date)"
cmds=""
njobs=7
gpu_t=0
dataset_t="femnist"
local_data_distrb='iid'
model_t="mclr" # "cnn"
name_t="SigMdl"
valid_rate=0
num_aggregate_locals_t=35

local_epochs_t=25 # local batch
learning_rate_t=0.01
mu=0.9 # momentum coefficient
batch_size_t=32
total_epochs_t=1500
corp_rate=0.6

logf='results/'$(date +"%m-%d-%H:%M")_${name_t}_${dataset_t}_${local_data_distrb}_${model_t}_${total_epochs_t}r_corp${corp_rate}
if [ ! -d $logf ];then
  mkdir $logf
fi
for ra in "CWM" "GM" #"FedAvg"  "norm"
do
for at in "signflip" "labelflip" "gauss" #'empire' 'omniscient' 'little'
do
for times_t in 0 1 2 3 4
    do
        cmds="$cmds ; time python -u main_fl.py --gpu $gpu_t --dataset $dataset_t --name $name_t --model $model_t \
        --learning_rate $learning_rate_t --num_aggregate_locals $num_aggregate_locals_t --batch_size $batch_size_t \
        --local_epochs $local_epochs_t --total_epochs $total_epochs_t --times $times_t \
        --ra $ra --local_data_distrb $local_data_distrb --at $at --corp_rate $corp_rate \
        --valid_rate $valid_rate --mu $mu --logf $logf 2>&1 "
    done
done
done


set +u # for parallel exec to work (unbound variables)
f_ParallelExec $njobs "$cmds"

echo "Done at $(date)"