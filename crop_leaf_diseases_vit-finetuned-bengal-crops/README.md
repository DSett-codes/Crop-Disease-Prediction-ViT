---
library_name: transformers
license: mit
base_model: wambugu71/crop_leaf_diseases_vit
tags:
- generated_from_trainer
metrics:
- accuracy
model-index:
- name: crop_leaf_diseases_vit-finetuned-bengal-crops
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# crop_leaf_diseases_vit-finetuned-bengal-crops

This model is a fine-tuned version of [wambugu71/crop_leaf_diseases_vit](https://huggingface.co/wambugu71/crop_leaf_diseases_vit) on an unknown dataset.
It achieves the following results on the evaluation set:
- Loss: 0.2703
- Accuracy: 0.9060

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 64
- eval_batch_size: 64
- seed: 42
- gradient_accumulation_steps: 4
- total_train_batch_size: 256
- optimizer: Use OptimizerNames.ADAMW_TORCH with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: linear
- lr_scheduler_warmup_ratio: 0.1
- num_epochs: 3

### Training results

| Training Loss | Epoch | Step | Validation Loss | Accuracy |
|:-------------:|:-----:|:----:|:---------------:|:--------:|
| 0.4845        | 1.0   | 304  | 0.4700          | 0.8456   |
| 0.3312        | 2.0   | 608  | 0.3128          | 0.8933   |
| 0.2491        | 3.0   | 912  | 0.2703          | 0.9060   |


### Framework versions

- Transformers 4.57.1
- Pytorch 2.8.0+cu126
- Datasets 4.4.2
- Tokenizers 0.22.1
