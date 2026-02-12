# Crop Disease Prediction with ViT

Vision Transformer fine-tuning for 94 crop disease classes (banana, cauliflower, corn, cotton, guava, jute, mango, papaya, potato, rice, sugarcane, tea, tomato, wheat). The project ships a local Hugging Face-compatible model folder plus a lightweight inference script.

## What you get
- Fine-tuned ViT classifier with ~90.6% validation accuracy (see [crop_leaf_diseases_vit-finetuned-bengal-crops/README.md](crop_leaf_diseases_vit-finetuned-bengal-crops/README.md)).
- Ready-to-run inference example in [test.py](test.py) that reads `image.png` and prints the predicted label.
- Training and evaluation walkthrough in the notebook [crop-disease-prediction-vit-fine-tuning.ipynb](crop-disease-prediction-vit-fine-tuning.ipynb).
- Full label map and processor/model configs in [crop_leaf_diseases_vit-finetuned-bengal-crops/config.json](crop_leaf_diseases_vit-finetuned-bengal-crops/config.json) and [crop_leaf_diseases_vit-finetuned-bengal-crops/preprocessor_config.json](crop_leaf_diseases_vit-finetuned-bengal-crops/preprocessor_config.json).

## Project layout
- [crop_leaf_diseases_vit-finetuned-bengal-crops/](crop_leaf_diseases_vit-finetuned-bengal-crops/) – model weights (`model.safetensors`), config, processor, and trainer metadata.
- [crop_leaf_diseases_vit-finetuned-bengal-crops.pth](crop_leaf_diseases_vit-finetuned-bengal-crops.pth) – PyTorch checkpoint export.
- [test.py](test.py) – minimal inference script using local weights.
- [crop-disease-prediction-vit-fine-tuning.ipynb](crop-disease-prediction-vit-fine-tuning.ipynb) – end-to-end fine-tuning and evaluation.
- [requirements.txt](requirements.txt) – Python dependencies.

## Setup
1) Create a virtual environment (Python 3.10+ recommended) and activate it.
2) Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick inference
1) Place an RGB leaf image at `image.png` (or update the path below).
2) Run the bundled script:
```bash
python test.py
```

Or use the model folder directly in your own code:
```python
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image

model_dir = "./crop_leaf_diseases_vit-finetuned-bengal-crops"
processor = AutoImageProcessor.from_pretrained(model_dir)
model = AutoModelForImageClassification.from_pretrained(model_dir)

image = Image.open("path/to/leaf.jpg").convert("RGB")
inputs = processor(image, return_tensors="pt")
with torch.no_grad():
	logits = model(**inputs).logits
	probs = logits.softmax(dim=-1)[0]
	pred_id = int(probs.argmax())

print("Predicted class:", model.config.id2label[str(pred_id)])
print("Confidence:", float(probs[pred_id]))
```

## Model notes
- Architecture: ViTForImageClassification (hidden size 192, 12 layers, 3 heads, patch size 16).
- Classes: 94 crop disease and healthy categories (see `id2label` in [config.json](crop_leaf_diseases_vit-finetuned-bengal-crops/config.json)).
- Image size: 224x224, normalized to mean/std = 0.5; rescale factor 1/255 (see [preprocessor_config.json](crop_leaf_diseases_vit-finetuned-bengal-crops/preprocessor_config.json)).
- Training summary: 3 epochs, effective batch size 256, learning rate 5e-5, linear scheduler with 10% warmup. Best val accuracy: 0.9060, val loss: 0.2703.

## Training and evaluation
- The notebook [crop-disease-prediction-vit-fine-tuning.ipynb](crop-disease-prediction-vit-fine-tuning.ipynb) covers data loading, augmentation, training, and evaluation logging.
- Trainer artifacts (loss/accuracy per epoch and TensorBoard run) live in [crop_leaf_diseases_vit-finetuned-bengal-crops/runs](crop_leaf_diseases_vit-finetuned-bengal-crops/runs).
- To re-train on your own data, adjust the dataset paths and label names in the notebook, then re-export the model directory.

## Tips
- Ensure input images are clear leaf close-ups; mixed backgrounds can reduce accuracy.
- Keep class names consistent with the existing `id2label` mapping when adding new data.
- If you are GPU-limited, lower `train_batch_size` and `gradient_accumulation_steps` in the notebook; keep the effective batch size similar.

## License
Model artifacts are released under the MIT license (see model card in [crop_leaf_diseases_vit-finetuned-bengal-crops/README.md](crop_leaf_diseases_vit-finetuned-bengal-crops/README.md)).
