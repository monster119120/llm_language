CUDA_VISIBLE_DEVICES=0 L=8 LENGTH=8 nohup python src/train_ae.py \
    --dataset_name wikitext  --dataset_config_name wikitext-2-raw-v1 \
    --encoder_config_path meta-llama/Llama-2-7b-hf \
    --decoder_config_path meta-llama/Llama-2-7b-hf \
    --tokenizer_config_path meta-llama/Llama-2-7b-hf \
    --per_device_train_batch_size 16    \
    --output_dir ./outputs/L8_LENGTH8/     \
    --block_size 99     \
    --checkpointing_steps 100     \
    --learning_rate 5e-5 \
    --num_train_epochs 100 >  logs/L8_LENGTH8.log 2>&1 &


CUDA_VISIBLE_DEVICES=1 L=100 LENGTH=3 nohup python src/train_ae.py \
    --dataset_name wikitext  --dataset_config_name wikitext-2-raw-v1 \
    --encoder_config_path meta-llama/Llama-2-7b-hf \
    --decoder_config_path meta-llama/Llama-2-7b-hf \
    --tokenizer_config_path meta-llama/Llama-2-7b-hf \
    --per_device_train_batch_size 16    \
    --output_dir ./outputs/L100_LENGTH3_AllInfo/     \
    --block_size 99     \
    --checkpointing_steps 100     \
    --learning_rate 5e-5 \
    --num_train_epochs 100 >  logs/L100_LENGTH3.log 2>&1 &