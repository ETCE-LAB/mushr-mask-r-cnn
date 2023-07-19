This repository hosts the MushR Mask R CNN model for maturity
detection of Oyster mushrooms.

# Dependencies

To run inference, please follow the official instructions to [install
detectron2](https://detectron2.readthedocs.io/en/latest/tutorials/install.html).

# Annotations

1. Train: [MushR_Dataset_V1.1_2023_04_04-COCO.json](MushR_Dataset_V1.1_2023_04_04-COCO.json)
2. Test: [MushR_Dataset_2023_06_29_TEST-COCO.json](MushR_Dataset_2023_06_29_TEST-COCO.json)

# Training 

All training related configuration can be found in [train_2023_04_04/](train_2023_04_04/).

We used the [train.sh](train.sh) script to train the model.

To continue training from the last checkpoint, use: [continue_training.sh](continue_training.sh)

## NOTE

The model weights of our trained model can be found in <https://github.com/ETCE-LAB/mushr-mask-r-cnn/releases>


# Inference

The scripts, [inference_images.sh](inference_images.sh) and [inference_images-single.sh](inference_images-single.sh)
can be used to run inference.

Please check [inference_2023_04_20](inference_2023_04_20) for examples of expected results.

# Evaluation

To evaluate the model, please use the [eval.sh](eval.sh) script.

Our evaluation results of the trained model are given in [eval_2023_07_19](eval_2023_07_19).
