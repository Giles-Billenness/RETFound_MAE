python -m torch.distributed.launch --nproc_per_node=1 --master_port=48793 main_finetune.py \
    --batch_size 32 \
    --world_size 1 \
    --model swinv2_largeNODECAY \
    --epochs 50 \
    --blr 5e-3 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.2 \
    --nb_classes 2 \
    --data_path /home/gbillenn/DissProj/Data/mainCSV/filePathChanged/ \
    --dataset stroke \
    --task BinStr224SwinNODECAY \
    --log_dir ../Results/BinStr224SwinNODECAY/ \
    --output_dir ../Results/ \
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