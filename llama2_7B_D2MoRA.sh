export model_path='meta-llama/Llama-2-7B-hf'
export weight_path='checkpoints/D2MoRA'

export device_id=0,1

CUDA_VISIBLE_DEVICES=$device_id torchrun --nproc_per_node 2 --master_port 34517 finetune.py \
   --base_model $model_path \
   --data_path 'LLM-Adapters/commonsense_170k.json' \
   --output_dir $weight_path \
   --batch_size 16 --micro_batch_size 4 --num_epochs 4 \
   --learning_rate 3e-4 --cutoff_len 256 --val_set_size 2000 \
   --eval_step 200 --save_step 200 \
   --use_orth_loss \
   --adapter_name d2mora --expert_down 4 --expert_up 3 \
   --target_modules '["q_proj", "k_proj", "v_proj"]' \
   --lora_r $1 --lora_alpha $2 --use_gradient_checkpointing
