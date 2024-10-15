python -m torch.distributed.launch --nproc_per_node=1 --master_port=48798 main_finetune.py \
    --batch_size 16 \
    --world_size 1 \
    --model vit_large_patch16 \
    --epochs 50 \
    --blr 5e-3 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.2 \
    --nb_classes 5 \
    --data_path ../Data/APTOS2019/ \
    --task ../Results/APTOS2019/ \
    --finetune ../Weights/RETFound_cfp_weights.pth \
    --input_size 224