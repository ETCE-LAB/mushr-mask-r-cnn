last_checkpoint=$(cat train_2023_06_29_04/last_checkpoint)

./train_net.py --num-gpus 1 \
               --config-file train_2023_06_29_04/config.yaml \
	       MODEL.WEIGHTS "train_2023_06_29_04/$last_checkpoint" \
	       INPUT.CROP.ENABLED True \
               DATASETS.TRAIN \(\"MushR_Dataset_2023_06_29\",\) \
               SOLVER.MAX_ITER 1000000 \
               SOLVER.CHECKPOINT_PERIOD 1000 \
               SOLVER.IMS_PER_BATCH 8 \
               DATASETS.TEST \(\"MushR_Dataset_2023_06_29_TEST\",\) \
               SOLVER.BASE_LR 0.001 \
               OUTPUT_DIR "train_2023_06_29_04/" \
