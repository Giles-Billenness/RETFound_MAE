python -m torch.distributed.launch --nproc_per_node=1 --master_port=48792 main_finetune.py \
    --batch_size 32 \
    --world_size 1 \
    --model swinv2_large_224_META \
    --epochs 50 \
    --blr 5e-3 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.2 \
    --nb_classes 2 \
    --data_path /home/gbillenn/DissProj/Data/mainCSV/filteredConvertedMain/N70 \
    --dataset stroke \
    --task FilStrSwin224_noAug_META \
    --log_dir ../Results/Filtered_Stroke_07_noAug_META/FilStrSwin224_noAug_META/ \
    --output_dir ../Results/Filtered_Stroke_07_noAug_META/ \
    --finetune ../Weights/RETFound_cfp_weights.pth \
    --input_size 224


# default
# --batch_size 32 \
# --input_size 224

# large 512
# --batch_size 3 \
# --input_size 512

# 40 -> 21481
# 44 -> 23137* this one