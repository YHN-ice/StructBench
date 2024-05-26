CUDA_VISIBLE_DEVICES=0 ./venv/Scripts/python src/train_bash.py \
	--stage sft \
	--do_train \
	# need customize
	--model_name_or_path ./open_llama_7b \
	# need customize
	--dataset mixrulehint \
	--template default \
	--finetuning_type freeze \
	--lora_target q_proj,v_proj \
	# need customize
	--output_dir mixrulehint_finetune \
	--overwrite_cache \
	--per_device_train_batch_size 2 \
	--gradient_accumulation_steps 4 \
	--lr_scheduler_type cosine \
	--logging_steps 10 \
	--save_steps 300  \
	--learning_rate 5e-5 \
	--num_train_epochs 10 \
	--plot_loss \
	--fp16 \
	--overwrite_output_dir

