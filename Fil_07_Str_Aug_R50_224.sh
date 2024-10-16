python -m torch.distributed.launch --nproc_per_node=1 --master_port=48771 main_finetune.py \
    --batch_size 32 \
    --world_size 1 \
    --model resnet50 \
    --epochs 50 \
    --blr 5e-3 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.2 \
    --nb_classes 2 \
    --mixup 0.2 \
    --cutmix 1.0 \
    --data_path /home/gbillenn/DissProj/Data/mainCSV/filteredConvertedMain/N70 \
    --dataset stroke \
    --task FilStrR50224 \
    --log_dir ../Results/Filtered_Stroke_07_Aug02/FilStrR50224/ \
    --output_dir ../Results/Filtered_Stroke_07_Aug02/ \
    --finetune ../Weights/RETFound_cfp_weights.pth \
    --clip_grad 1 \
    --input_size 224


# default
# --batch_size 32 \
# --input_size 224

# large 512
# --batch_size 3 \
# --input_size 512

# 40 -> 21481
# 44 -> 23137* this one