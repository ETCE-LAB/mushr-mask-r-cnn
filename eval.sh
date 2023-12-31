last_checkpoint=$(cat train_2023_04_04/last_checkpoint)

./train_net.py --num-gpus 1 \
	       --eval-only \
	       --config-file "train_2023_04_04/config.yaml" \
	       MODEL.WEIGHTS "train_2023_04_04/$last_checkpoint"\
               DATASETS.TEST \(\"MushR_Dataset_2023_06_29_TEST\",\) \
               OUTPUT_DIR "eval_2023_07_19_new" \
