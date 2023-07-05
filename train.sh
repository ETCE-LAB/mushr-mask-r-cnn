last_checkpoint=$(cat train_2023_04_20/last_checkpoint)

./train_net.py --num-gpus 1 \
               --config-file ~/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml \
	       INPUT.CROP.ENABLED True \
               DATASETS.TRAIN \(\"MushR_Dataset_2023_06_29\",\) \
               SOLVER.MAX_ITER 4000 \
               SOLVER.IMS_PER_BATCH 8 \
               DATASETS.TEST \(\"MushR_Dataset_2023_06_29_TEST\",\) \
               SOLVER.BASE_LR 0.0001 \
               OUTPUT_DIR "train_2023_06_29_03/" \
