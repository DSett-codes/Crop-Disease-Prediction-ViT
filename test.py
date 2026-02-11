import torch
import transformers
from transformers import  AutoModelForImageClassification, AutoImageProcessor # use the correct AutoModel class
from PIL import Image

device = torch.device(0 if torch.cuda.is_available() else -1)
transformers.pipelines.torch = torch
# 1. Define the path to your output directory
output_directory = "./crop_leaf_diseases_vit-finetuned-bengal-crops" # Replace with your actual directory path


# 2. Load the model
# The correct AutoModel class depends on your task (e.g., AutoModelForCausalLM, AutoModelForSequenceClassification)
image_processor = AutoImageProcessor.from_pretrained(output_directory)
model = AutoModelForImageClassification.from_pretrained(output_directory)


image = Image.open('image.png')
# prepare image for the model
encoding = image_processor(image.convert("RGB"), return_tensors="pt")
print(encoding.pixel_values.shape)

import torch

# forward pass
with torch.no_grad():
    outputs = model(**encoding)
    logits = outputs.logits
predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])