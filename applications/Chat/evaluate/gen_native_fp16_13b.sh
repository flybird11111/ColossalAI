export CUDA_VISIBLE_DEVICES="5"
device_number=1
model_name="opt"
model_path="/home/lczxl/.cache/huggingface/hub/models--facebook--opt-13b/snapshots/e515202d1e7750da62d245fbccb2723b9c1790f5"
dataset="/home/lcxyc/data1/evaluate/ColossalAI/applications/Chat/evaluate/table/evaluation_dataset/cn/evaluate_cn_v4_300.json"
answer_path="opt-13b"


torchrun --standalone --nproc_per_node=$device_number generate_answers.py \
    --model 'a' \
    --strategy ddp \
    --model_path $model_path \
    --model_name $model_name \
    --dataset $dataset \
    --batch_size 4 \
    --answer_path $answer_path \
    --max_length 1024
    # --max_datasets_size 80 \