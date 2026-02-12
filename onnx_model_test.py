import numpy as np
import onnxruntime as ort
from PIL import Image
import json

onnx_dir = "onnx_output"


def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224))
    img_data = np.array(img).transpose(2, 0, 1) # Change to (C, H, W)
    
    # Normalize: (x - mean) / std (values typical for ImageNet)
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    img_data = (img_data / 255.0 - mean) / std
    
    return img_data.astype(np.float32)[np.newaxis, :] # Add batch dimension




# load label map
with open(f"{onnx_dir}/config.json", "r", encoding="utf-8") as f:
    id2label = json.load(f)["id2label"]   # keys are strings: "0", "1", ...

session = ort.InferenceSession(f"{onnx_dir}/model.onnx", providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
processed_img = preprocess_image("image.png")
# after preprocessing image -> processed_img
logits = session.run(None, {input_name: processed_img})[0]   # shape [1, num_classes]
pred_id = int(np.argmax(logits))

pred_label = id2label[str(pred_id)]

print("Predicted class:", pred_label)
