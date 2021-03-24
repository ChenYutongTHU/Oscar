export WORLD_SIZE="1"
srun -N 1 -G 1 python oscar/run_VIVO_pretrain.py \
    --model_name_or_path ../pretrained_models/checkpoint-24-50 \
    --do_train \
    --do_lower_case \
    --add_od_labels \
    --learning_rate 5e-5 --weight_decay 0.05 --adam_epsilon 1e-8 --max_grad_norm 10 --warmup_steps 0 --scheduler linear \
    --per_gpu_train_batch_size 128 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 100 \
    --tie_weights \
    --data_dir /data/private/NocapsData/VIVO_pretrain_demo \
    --num_workers 4 \
    --output_dir ../output/debug \
    --max_seq_length 17  --max_seq_a_length 17 --max_img_seq_length 50 \
    --mask_prob 0.15 --max_masked_tokens 3 \
    --logging_steps 10 --save_steps 50 \
    --evaluate_during_training --whole_word --use_img_layernorm 1