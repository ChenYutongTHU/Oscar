model_path=./VIVO_output/pretrain_128_8_resume31_noloadoptim/checkpoint-99-163600.0
save_path=./caption_output/64_8_epoch60_5e-5/
datadir=/data/private/NocapsData/coco_caption_kara/
nocaps_dir=/data/private/NocapsData/nocaps1k_kara
num_epoch=60 #30

export WORLD_SIZE="8"
python3.6 -m torch.distributed.launch --nproc_per_node=${WORLD_SIZE} oscar/run_captioning.py \
    --model_name_or_path ${model_path} \
    --data_dir ${datadir} \
    --do_train --evaluate_during_training \
    --do_lower_case \
    --add_od_labels \
    --max_seq_a_length 40 --max_seq_length 70 \
    --learning_rate 5e-5 \
    --per_gpu_train_batch_size 64 --gradient_accumulation_steps 1 --num_workers 1 \
    --num_train_epochs ${num_epoch} \
    --tie_weights \
    --freeze_embedding \
    --label_smoothing 0.1 \
    --drop_worst_ratio 0.2 \
    --drop_worst_after 20000 \
    --output_dir ${save_path} --logging_steps 100 \
    --evaluate_during_training --eval_model_dir ${save_path}/eval_output \
    --max_gen_length 20 --num_beams 1 --num_keep_best 1 --repetition_penalty 1 --length_penalty 1 \
    --evaluate_nocaps --nocaps_evaluate_dir ${nocaps_dir} \
    --amp 
    
###only load ckpt!!