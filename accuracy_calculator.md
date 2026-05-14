# 📊 Accuracy Calculation Guide

This guide explains how accuracy is calculated for the Healthcare DL models and provides a step-by-step procedure for you to manually verify the results.

---

## 1. The Core Logic

Accuracy is the percentage of **correct predictions** made on a **held-out test set** (data the model has never seen before).

**The Formula:**
Accuracy = (Correct Predictions / Total Test Images) × 100

---

## 2. Step-by-Step Manual Calculation Procedure

If you want to verify the accuracy yourself without relying on automated scripts, follow these steps:

### Phase A: Setup & Sampling

1. **Choose a Model:** Pick one model (e.g., Pneumonia).
2. **Locate the Dataset:** Go to `main/datasets/pneumonia/`.
3. **Identify Test Images:** Since the project uses a 15% test split, select roughly 15% of images from each subfolder (e.g., `NORMAL` and `PNEUMONIA`).
4. **Create a Table:** Open Excel or a notebook and create three columns:
   - `Image Name`
   - `Actual Label`
   - `Model Prediction`

### Phase B: Running Inference

For each image in your list:

1. Run a single-image prediction (you can use the script below).
2. Record what the model predicts.
3. Compare it to the actual label.

### Phase C: Final Calculation

1. **Count Total:** Total number of images you tested.
2. **Count Correct:** Number of times `Actual Label == Model Prediction`.
3. **Divide:** (Correct / Total) = Decimal.
4. **Multiply by 100:** To get the final percentage.

---

## 3. Tool: Manual Accuracy Tester Script

To help you with Phase B, I have created a simplified script. You can run this to see the model's prediction for any specific image.

**Create a file named `manual_check.py` in `main/training/` and paste this:**

```python
import torch
from PIL import Image
from torchvision import transforms
from config import DEVICE, build_model

# 1. SETUP: Change these for the model you want to check
MODEL_NAME = "pneumonia"  # Options: pneumonia, covid, brain_tumor, etc.
IMAGE_PATH = "path/to/your/test/image.jpg"
CLASS_NAMES = ["Normal", "Pneumonia"] # Check config.py for correct order

# 2. LOAD MODEL
model = build_model(MODEL_NAME, len(CLASS_NAMES))
model.load_state_dict(torch.load(f"../backend/trained_models/{MODEL_NAME}.pth", map_location=DEVICE))
model.eval()

# 3. PREPROCESS
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

img = Image.open(IMAGE_PATH).convert('RGB')
tensor = transform(img).unsqueeze(0).to(DEVICE)

# 4. PREDICT
with torch.no_grad():
    output = model(tensor)
    prob = torch.softmax(output, dim=1)
    conf, pred = torch.max(prob, 1)

print(f"\n--- Results for {MODEL_NAME} ---")
print(f"Prediction: {CLASS_NAMES[pred.item()]}")
print(f"Confidence: {conf.item()*100:.2f}%")
```
