# 🏋️ Model Training Guide

Complete step-by-step guide to train all 7 deep learning models for the Healthcare DL project.

---

## Prerequisites

Before training, make sure:
- ✅ All datasets are downloaded and organized in `main/datasets/`
- ✅ Backend virtual environment exists at `main/backend/venv/`
- ✅ Your NVIDIA GPU driver is installed (RTX 3050)

---

## Step 1: Open a Terminal

Open **PowerShell** or **Terminal** on your computer.

---

## Step 2: Navigate to the Training Folder

```powershell
cd d:\healthcareDeep_Learning\main\training
```

---

## Step 3: Install Training Dependencies (One Time Only)

Copy-paste this entire command:

```powershell
d:\healthcareDeep_Learning\main\backend\venv\Scripts\pip.exe install scikit-learn matplotlib seaborn tqdm
```

Wait for it to finish. You should see `Successfully installed ...`

---

## Step 4: Verify GPU is Working

Copy-paste this command:

```powershell
d:\healthcareDeep_Learning\main\backend\venv\Scripts\python.exe -c "import torch; print('GPU Available:', torch.cuda.is_available()); print('GPU Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

You should see:
```
GPU Available: True
GPU Name: NVIDIA GeForce RTX 3050 Laptop GPU
```

If it says `False`, your GPU is not configured properly. Training will still work on CPU but will be much slower.

---

## Step 5: Prepare Image Classifier Dataset (One Time Only)

This creates the `image_classifier` dataset by copying sample images from your other datasets:

```powershell
d:\healthcareDeep_Learning\main\backend\venv\Scripts\python.exe d:\healthcareDeep_Learning\main\training\prepare_image_classifier_dataset.py
```

You should see:
```
✅ xray: 300 images copied
✅ mri: 300 images copied
✅ ct_scan: 300 images copied
✅ skin: 300 images copied
```

---

## Step 6: Train All 7 Models (One by One)

> ⚠️ **IMPORTANT**: Run these commands **ONE AT A TIME**. Wait for each training to fully complete before starting the next one. Each training will show epoch-by-epoch progress and will automatically stop when the model is done learning (early stopping).

> 💡 **TIP**: Each model saves its best weights automatically. If training gets interrupted (laptop sleeps, terminal closes), just re-run the same command — it will start fresh and overwrite the previous attempt.

---

### Model 1: Image Type Classifier (~5 minutes)

This model learns to tell the difference between X-ray, MRI, CT Scan, and Skin images.

```powershell
d:\healthcareDeep_Learning\main\backend\venv\Scripts\python.exe d:\healthcareDeep_Learning\main\training\train_image_classifier.py
```

**What you'll see:**
```
🚀 Using GPU: NVIDIA GeForce RTX 3050 Laptop GPU
📊 Dataset: ...image_classifier
   Total images: 1200
   Classes: ['ct_scan', 'mri', 'skin', 'xray']
   Split: train=840, val=180, test=180

🏋️ Training: image_classifier
Epoch   1/50 │ Train Loss: 0.5294 Acc:  92.14% │ Val Loss: 0.0350 Acc: 100.00%
  💾 Best model saved!
Epoch   2/50 │ Train Loss: 0.0700 Acc:  99.64% │ Val Loss: 0.0091 Acc: 100.00%
...
⏹️  Early stopping triggered at epoch X

✅ Training complete: image_classifier
   Best Val Accuracy: ~99-100%
   Model saved to: .../trained_models/image_classifier.pth

📊 EVALUATION RESULTS: image_classifier
   Accuracy:  ~99%
```

**Expected accuracy: 95-100%**

---

### Model 2: Pneumonia Detector (~10 minutes)

This model detects pneumonia from chest X-rays.

```powershell
d:\healthcareDeep_Learning\main\backend\venv\Scripts\python.exe d:\healthcareDeep_Learning\main\training\train_pneumonia.py
```

**Expected accuracy: 90-95%**

---

### Model 3: Brain Tumor Detector (~15 minutes)

This model classifies brain MRI scans into 4 types: glioma, meningioma, pituitary tumor, or no tumor.

```powershell
d:\healthcareDeep_Learning\main\backend\venv\Scripts\python.exe d:\healthcareDeep_Learning\main\training\train_brain_tumor.py
```

**Expected accuracy: 92-97%**

---

### Model 4: Lung Cancer Detector (~10 minutes)

This model classifies CT scans as normal, benign tumor, or malignant tumor.

```powershell
d:\healthcareDeep_Learning\main\backend\venv\Scripts\python.exe d:\healthcareDeep_Learning\main\training\train_lung_cancer.py
```

**Expected accuracy: 85-92%**

---

### Model 5: COVID-19 Detector (~20 minutes)

This model detects COVID-19, viral pneumonia, or normal from chest X-rays. Larger dataset = longer training.

```powershell
d:\healthcareDeep_Learning\main\backend\venv\Scripts\python.exe d:\healthcareDeep_Learning\main\training\train_covid.py
```

**Expected accuracy: 90-95%**

---

### Model 6: Cardiomegaly Detector (~15 minutes)

This model detects enlarged hearts from chest X-rays.

```powershell
d:\healthcareDeep_Learning\main\backend\venv\Scripts\python.exe d:\healthcareDeep_Learning\main\training\train_cardiomegaly.py
```

**Expected accuracy: 85-92%**

---

### Model 7: Skin Lesion Classifier (~25 minutes)

This is the hardest model — 7 different skin conditions. Takes the longest.

```powershell
d:\healthcareDeep_Learning\main\backend\venv\Scripts\python.exe d:\healthcareDeep_Learning\main\training\train_skin_lesion.py
```

**Expected accuracy: 75-85%**

---

## Step 7: Verify All Models are Saved

After training all models, run this to check:

```powershell
Get-ChildItem d:\healthcareDeep_Learning\main\backend\trained_models\*.pth | Format-Table Name, @{Label="Size (MB)"; Expression={[math]::Round($_.Length/1MB, 1)}}
```

You should see **7 files**:
```
Name                     Size (MB)
----                     ---------
image_classifier.pth          16
pneumonia.pth                 16
brain_tumor.pth               94
lung_cancer.pth               94
covid.pth                     16
cardiomegaly.pth              16
skin_lesion.pth               16
```

(ResNet50 models are ~94MB, EfficientNet-B0 models are ~16MB)

---

## Step 8: Test with the Backend

After all models are trained, restart the backend to load the real trained models:

```powershell
cd d:\healthcareDeep_Learning\main\backend
.\venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000
```

In the backend logs, you should now see:
```
✓ image_classifier: loaded trained weights
✓ pneumonia: loaded trained weights
✓ brain_tumor: loaded trained weights
...
Model registry ready: 7/7 models loaded
```

Now upload a real medical image through the frontend — predictions should be meaningful!

---

## Troubleshooting

### "CUDA out of memory" Error
Your GPU ran out of memory. Fix: Open the training script and reduce `batch_size`:
```python
# In the train_model() call, add:
result = train_model(
    ...
    # Add this line:
)
```
Or edit `config.py` line 40 and change `"batch_size": 16` to `"batch_size": 8`.

### Training is Very Slow (No GPU)
If you see `⚠️ Using CPU`, PyTorch is not detecting your GPU. Run:
```powershell
d:\healthcareDeep_Learning\main\backend\venv\Scripts\pip.exe install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Training Got Interrupted
Just re-run the same command. The script will start fresh and overwrite the old `.pth` file.

### "ModuleNotFoundError: No module named 'sklearn'"
Run Step 3 again to install the training dependencies.

---

## Understanding the Output

During training, each line looks like this:

```
Epoch  5/50 │ Train Loss: 0.1234 Acc: 95.00% │ Val Loss: 0.2345 Acc: 92.00% │ LR: 1.00e-04
```

| Column | Meaning |
|--------|---------|
| **Epoch 5/50** | Currently on training round 5 out of maximum 50 |
| **Train Loss: 0.1234** | How wrong the model is on training data (lower = better) |
| **Train Acc: 95.00%** | Percentage of training images correctly classified |
| **Val Loss: 0.2345** | How wrong the model is on unseen validation data (lower = better) |
| **Val Acc: 92.00%** | Percentage of validation images correctly classified — **this is the important number!** |
| **LR: 1.00e-04** | Current learning rate (automatically decreases if model plateaus) |

### Key Messages
- `💾 Best model saved!` → The model just hit a new best validation accuracy, weights are saved.
- `⏹️ Early stopping triggered` → The model stopped improving, training ends automatically.
- `✅ Training complete` → Done! Final results are printed below.

---

## Total Estimated Training Time

| Model | Time | GPU Memory Used |
|-------|------|----------------|
| Image Classifier | ~5 min | ~1.5 GB |
| Pneumonia | ~10 min | ~1.5 GB |
| Brain Tumor | ~15 min | ~3.0 GB |
| Lung Cancer | ~10 min | ~3.0 GB |
| COVID-19 | ~20 min | ~1.5 GB |
| Cardiomegaly | ~15 min | ~1.5 GB |
| Skin Lesion | ~25 min | ~1.5 GB |
| **TOTAL** | **~1.5 hours** | |

> 💡 You can take breaks between models! Each model trains independently.
