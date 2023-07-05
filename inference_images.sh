last_checkpoint=$(cat train_2023_04_20/last_checkpoint)

python inference.py \
       --config-file "train_2023_04_20/config.yaml" \
       --input \
       /home/sho/Desktop/ANNOTATION_BATCHES/test\ file/* \
       --output "inference_2023_04_20" \
       --opts \
       MODEL.WEIGHTS "train_2023_04_20/$last_checkpoint"
