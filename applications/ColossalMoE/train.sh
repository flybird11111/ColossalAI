NUM_GPU=8
# MODEL="mistralai/Mixtral-8x7B-v0.1"
MODEL="/home/data/models/Mixtral-8x7B-Instruct-v0.1"
SEQ_LENGTH=2048
BATCH_SIZE=1
LR=0.00001

# hybrid
# colossalai run --nproc_per_node $NUM_GPU --hostfile "hostfile" \
torchrun --standalone --nproc_per_node $NUM_GPU \
    train.py \
    --num_epoch 1 \
    --model_name $MODEL \
    --plugin "hybrid" \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --zero_stage 1 \
    --pp_size 1 \
    --dp_size 1 \
    --ep_size 8 \
