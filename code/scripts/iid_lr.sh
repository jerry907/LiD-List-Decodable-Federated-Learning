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
valid_rate=0.2
model_t="lr" # "cnn"
name_t="ListDec" # algorithm
local_epochs_t=25
optimizer_name_t="CustomSGD"

learning_rate_t=0.01
mu=0.9 # global momentum
batch_size_t=32 # same as FLTrust. 16-39 batches = 1 epoch
num_aggregate_locals_t=1 # number of clients used for training and aggregating
total_epochs_t=1500 # FLTrust: 2500

# Byzantine attack
corp_rate=0.6
ra="ListDec"

logf='results/'$(date +"%m-%d-%H:%M")_${dataset_t}_${local_data_distrb}_${model_t}_${total_epochs_t}r
if [ ! -d "$logf" ];then
  mkdir $logf
fi
for times_t in 0 1 2 3 4
do
for at in "empire" "signflip" "gauss" "little" "omniscient" "labelflip" # 'clean' # # # --vote_at $vote_at 
    do
        cmds="$cmds ; time python -u main_fl.py --gpu $gpu_t --dataset $dataset_t --name $name_t --model $model_t --ra $ra \
        --learning_rate $learning_rate_t --num_aggregate_locals $num_aggregate_locals_t --batch_size $batch_size_t \
        --local_epochs $local_epochs_t --total_epochs $total_epochs_t --optimizer_name $optimizer_name_t --times $times_t \
        --local_data_distrb $local_data_distrb --at $at --corp_rate $corp_rate --valid_rate $valid_rate \
        --mu $mu --logf $logf 2>&1 "
    done
done

set +u # for parallel exec to work (unbound variables)
f_ParallelExec $njobs "$cmds"

echo "Done at $(date)"