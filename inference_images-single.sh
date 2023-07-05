last_checkpoint=$(cat train_2023_04_20/last_checkpoint)

python inference.py \
       --config-file "train_2023_04_20/config.yaml" \
       --input \
       /home/sho/Desktop/ANNOTATION_BATCHES/test\ file/20230203_070001.jpg \
       --output "inference_test" \
       --opts \
       MODEL.WEIGHTS "train_2023_04_20/$last_checkpoint"
