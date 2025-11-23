CUDA_VISIBLE_DEVICES=0 python src/run_dataset.py \
--original_dataset_path "./original_dataset" \
--new_dataset_path "./new_dataset" \
--prompt "photo of a crack defect image" \
--neg_prompt " " \
--datacheck