# Crop Disease Prediction with ViT

Vision Transformer fine-tuning for 94 crop disease classes (banana, cauliflower, corn, cotton, guava, jute, mango, papaya, potato, rice, sugarcane, tea, tomato, wheat). The project ships a local Hugging Face-compatible model folder, PyTorch inference, and an ONNX export for lightweight deployment.

## Why this exists
- Early, low-cost disease triage for smallholder farmers where bandwidth is limited.
- Focused on Bengali-region crops and diseases to improve relevance over generic plant models.
- Runs fully offline once weights are downloaded (PyTorch or ONNX), reducing cloud dependence.

## What is included
- Fine-tuned ViT classifier (~90.6% validation accuracy; see [crop_leaf_diseases_vit-finetuned-bengal-crops/README.md](crop_leaf_diseases_vit-finetuned-bengal-crops/README.md)).
- Local model folder with configs/label map in [crop_leaf_diseases_vit-finetuned-bengal-crops/](crop_leaf_diseases_vit-finetuned-bengal-crops/) and a PyTorch checkpoint [crop_leaf_diseases_vit-finetuned-bengal-crops.pth](crop_leaf_diseases_vit-finetuned-bengal-crops.pth).
- PyTorch inference script [test.py](test.py) and ONNX pipeline ([onnx_model_conversion.py](onnx_model_conversion.py), [onnx_model_test.py](onnx_model_test.py)) with exported weights in [onnx_output/](onnx_output/).
- Training and evaluation notebook [crop-disease-prediction-vit-fine-tuning.ipynb](crop-disease-prediction-vit-fine-tuning.ipynb).

## Setup
1) Create a virtual environment (Python 3.10+ recommended) and activate it.
2) Install dependencies:
```bash
pip install -r requirements.txt
```

## Use the model (PyTorch)
1) Place an RGB leaf image at `image.png` (or edit the path in [test.py](test.py)).
2) Run inference:
```bash
python test.py
```

Minimal inline example:
```python
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch

model_dir = "crop_leaf_diseases_vit-finetuned-bengal-crops"
processor = AutoImageProcessor.from_pretrained(model_dir)
model = AutoModelForImageClassification.from_pretrained(model_dir)

image = Image.open("image.png").convert("RGB")
inputs = processor(image, return_tensors="pt")
with torch.no_grad():
    probs = model(**inputs).logits.softmax(dim=-1)[0]
pred_id = int(probs.argmax())
print(model.config.id2label[str(pred_id)], float(probs[pred_id]))
```

## Use the model (ONNX)
- Export (already done in [onnx_output/](onnx_output/), but you can regenerate):
```bash
python onnx_model_conversion.py
```
- Run ONNX inference (expects `image.png`):
```bash
python onnx_model_test.py
```
The test script resizes to 224x224, applies ImageNet mean/std normalization, and uses CPUExecutionProvider for portability.

## Project layout
- [crop_leaf_diseases_vit-finetuned-bengal-crops/](crop_leaf_diseases_vit-finetuned-bengal-crops/) – safetensors weights, configs, processor, trainer metadata.
- [onnx_output/](onnx_output/) – ONNX model, config, and processor for lightweight runtimes.
- [test.py](test.py) – minimal PyTorch inference.
- [onnx_model_conversion.py](onnx_model_conversion.py) / [onnx_model_test.py](onnx_model_test.py) – ONNX export + CPU inference.
- [crop-disease-prediction-vit-fine-tuning.ipynb](crop-disease-prediction-vit-fine-tuning.ipynb) – training and evaluation walkthrough.
- [requirements.txt](requirements.txt) – Python dependencies.

## Model notes
- Architecture: ViTForImageClassification (hidden size 192, 12 layers, 3 heads, patch size 16).
- Classes: 94 crop disease and healthy categories (see `id2label` in [crop_leaf_diseases_vit-finetuned-bengal-crops/config.json](crop_leaf_diseases_vit-finetuned-bengal-crops/config.json)).
- Image size and preprocessing: 224x224; rescale 1/255; normalize to mean/std = 0.5 in the HF processor, or ImageNet mean/std in the ONNX test script.
- Training: 3 epochs, effective batch size 256, lr 5e-5, linear scheduler with 10% warmup; best val accuracy 0.9060, val loss 0.2703.

## Re-train or adapt
- Use the notebook [crop-disease-prediction-vit-fine-tuning.ipynb](crop-disease-prediction-vit-fine-tuning.ipynb) to swap in your dataset paths and labels, then re-export the model directory and (optionally) rerun ONNX export.
- Trainer logs and TensorBoard runs live in [crop_leaf_diseases_vit-finetuned-bengal-crops/runs](crop_leaf_diseases_vit-finetuned-bengal-crops/runs).

## Good practices
- Capture clear leaf close-ups; heavy background clutter can hurt predictions.
- Keep class names aligned with the existing `id2label` mapping when extending datasets.
- If GPU-limited, reduce `train_batch_size` and increase `gradient_accumulation_steps` to keep the effective batch size stable.

## License
Model artifacts are released under the MIT license (see model card in [crop_leaf_diseases_vit-finetuned-bengal-crops/README.md](crop_leaf_diseases_vit-finetuned-bengal-crops/README.md)).
