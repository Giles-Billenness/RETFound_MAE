python -m torch.distributed.launch --nproc_per_node=1 --master_port=47292 main_finetune.py \
    --batch_size 32 \
    --world_size 1 \
    --model swin_large_patch4_window7_224NEWDECAY \
    --epochs 50 \
    --blr 5e-3 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.2 \
    --nb_classes 2 \
    --data_path /home/gbillenn/DissProj/Data/mainCSV/filePathChanged/ \
    --dataset stroke \
    --task BinStr224SwinV1NewDecay \
    --log_dir ../Results/Full_Stroke_Test/BinStr224SwinV1NewDecay/ \
    --output_dir ../Results/Full_Stroke_Test/ \
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