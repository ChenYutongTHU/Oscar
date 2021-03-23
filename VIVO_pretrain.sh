export WORLD_SIZE="8"
srun -N 1 -G 8 python -m torch.distributed.launch --nproc_per_node=8 oscar/run_VIVO_pretrain.py \
    --model_name_or_path ../pretrained_models/bert_base_uncased \
    --do_train \
    --do_lower_case \
    --add_od_labels \
    --learning_rate 5e-5 --weight_decay 0.05 --adam_epsilon 1e-8 --max_grad_norm 1 --warmup_steps 0 --scheduler linear \
    --per_gpu_train_batch_size 128 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 100 \
    --tie_weights \
    --freeze_embedding \
    --data_dir /data/private/NocapsData/VIVO_pretrain_data \
    --num_workers 4 \
    --output_dir ../output/pretrain \
    --max_seq_length 15  --max_seq_a_length 15 --max_img_seq_length 50 \
    --mask_prob 0.15 --max_masked_tokens 3 \
    --logging_steps 100 --save_steps 5000 \
    --evaluate_during_training --whole_word
    #--evaluate_during_training
    
# python oscar/run_captioning.py \
#     --model_name_or_path pretrained_models/image_captioning/pretrained_base \
#     --do_train \
#     --do_lower_case \
#     --add_od_labels \
#     --learning_rate 3e-5 \
#     --per_gpu_train_batch_size 64 \
#     --num_train_epochs 60 \
#     --tie_weights \
#     --freeze_embedding \
#     --label_smoothing 0.1 \
#     --drop_worst_ratio 0.2 \
#     --drop_worst_after 20000 \
#     --output_dir output/

# python -m torch.distributed.launch --nproc_per_node=8 oscar/run_oscarplus_pretrain.py \
#     --use_b 1 \
#     --max_grad_norm 10.0 --gradient_accumulation_steps 1 \
#     --use_img_layernorm 1 \
#     --output_dir <your output folder> \
#     --bert_model bert --model_name_or_path bert-base-uncased \
#     --do_lower_case --learning_rate 5e-05 
#     --warmup_steps 0 --do_train --max_seq_length 35 --on_memory \
#     --max_img_seq_length 50 --img_feature_dim 2054 \
#     --drop_out 0.1 --train_batch_size 8 \
#     --ckpt_period 10000 --max_iters 2000000 --log_period 100 \
#     --data_dir <The input data dir that contain the .yaml files> --dataset_file coco_flickr30k_googlecc_gqa_sbu_oi_x152c4big2exp168.yaml
#     --textb_sample_mode 1 --texta_false_prob 0.25 
