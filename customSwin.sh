python -m torch.distributed.launch --nproc_per_node=2 --master_port=48795 main_finetune.py \
    --batch_size 32 \
    --world_size 2 \
    --model swinv2_large \
    --epochs 50 \
    --blr 5e-3 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.2 \
    --nb_classes 5 \
    --data_path ../Data/Kaggle/augmented_resized_V2/ \
    --task CustomSwinV2 \
    --log_dir ../Results/CustomSwinV2/ \
    --output_dir ../Results/ \
    --finetune ../Weights/RETFound_cfp_weights.pth \
    --input_size 224

# 40 -> 21481
# 44 -> 23137* this one