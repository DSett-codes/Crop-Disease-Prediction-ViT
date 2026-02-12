from optimum.onnxruntime import ORTModelForImageClassification
from transformers import ViTImageProcessor

model_id = "crop_leaf_diseases_vit-finetuned-bengal-crops"
save_dir = "onnx_output"

# Load and export the model in one step
model = ORTModelForImageClassification.from_pretrained(model_id, export=True)
processor = ViTImageProcessor.from_pretrained(model_id)

# Save the ONNX model
model.save_pretrained(save_dir)
processor.save_pretrained(save_dir)
