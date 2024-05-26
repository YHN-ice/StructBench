eval_on_single_run_single_chkpt() {
	CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
		--stage sft \
		--split test \
		--do_predict \
		--model_name_or_path $2/checkpoint-$1 \
		--dataset stmini \
		--template default \
		--finetuning_type freeze \
		--output_dir $3/checkpoint-$1 \
		--per_device_eval_batch_size 4 \
		--max_samples 300 \
		--predict_with_generate \
		--fp16

}
eval_on_single_run_multiple_chkpts() {
	for i in $(seq 300 300 5700);
	do eval_on_single_run_single_chkpt $i $1 $2
	done
}

CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
  --stage sft \
  --split test \
  --do_predict \
  # https://huggingface.co/openlm-research/open_llama_7b
  --model_name_or_path './open_llama_7b' \
  --dataset stmini \
  --template default \
  --finetuning_type freeze \
  --output_dir eval_chkpts_minitest/before_finetune \
  --per_device_eval_batch_size 2 \
  --max_samples 300 \
  --predict_with_generate \
  --fp16

MODEL_PATH='base_finetune'
OUTPUT_PATH='eval_chkpts_minitest/base_finetune'
eval_on_single_run_multiple_chkpts $MODEL_PATH $OUTPUT_PATH

MODEL_PATH='mix_finetune'
OUTPUT_PATH='eval_chkpts_minitest/mix_finetune'
eval_on_single_run_multiple_chkpts $MODEL_PATH $OUTPUT_PATH

MODEL_PATH='rulehint_finetune'
OUTPUT_PATH='eval_chkpts_minitest/rulehint_finetune'
eval_on_single_run_multiple_chkpts $MODEL_PATH $OUTPUT_PATH

MODEL_PATH='mixrulehint_finetune'
OUTPUT_PATH='eval_chkpts_minitest/mixrulehint_finetune'
eval_on_single_run_multiple_chkpts $MODEL_PATH $OUTPUT_PATH

MODEL_PATH='fewshot_finetune'
OUTPUT_PATH='eval_chkpts_minitest/fewshot_finetune'
eval_on_single_run_multiple_chkpts $MODEL_PATH $OUTPUT_PATH
