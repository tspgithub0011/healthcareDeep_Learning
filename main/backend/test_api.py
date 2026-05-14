"""
Comprehensive test — one image per model, verifying correct routing + predictions.
"""
import requests, os, sys

BACKEND = "http://localhost:8000"

# Test cases: (image_path, expected_image_type, expected_top_disease, label)
TESTS = [
    # Test 1: Chest X-ray with Pneumonia → should route to xray models
    {
        "label": "Pneumonia X-ray",
        "path": r"d:\healthcareDeep_Learning\main\datasets\chest_xray\pneumonia",
        "expect_type": "xray",
        "expect_disease": "Pneumonia",
    },
    # Test 2: Normal chest X-ray → should show Normal
    {
        "label": "Normal Chest X-ray",
        "path": r"d:\healthcareDeep_Learning\main\datasets\chest_xray\normal",
        "expect_type": "xray",
        "expect_disease": "Normal",
    },
    # Test 3: Brain MRI with Glioma
    {
        "label": "Brain MRI (Glioma)",
        "path": r"d:\healthcareDeep_Learning\main\datasets\brain_tumor\glioma",
        "expect_type": "mri",
        "expect_disease": "Glioma",
    },
    # Test 4: Brain MRI with no tumor
    {
        "label": "Brain MRI (No Tumor)",
        "path": r"d:\healthcareDeep_Learning\main\datasets\brain_tumor\no_tumor",
        "expect_type": "mri",
        "expect_disease": "No Tumor",
    },
    # Test 5: Lung Cancer CT (malignant)
    {
        "label": "CT Scan (Malignant Lung Cancer)",
        "path": r"d:\healthcareDeep_Learning\main\datasets\lung_cancer\malignant",
        "expect_type": "ct_scan",
        "expect_disease": "Malignant",
    },
    # Test 6: Lung Cancer CT (normal)
    {
        "label": "CT Scan (Normal Lung)",
        "path": r"d:\healthcareDeep_Learning\main\datasets\lung_cancer\normal",
        "expect_type": "ct_scan",
        "expect_disease": "Normal",
    },
    # Test 7: Skin lesion (melanoma)
    {
        "label": "Skin (Melanoma)",
        "path": r"d:\healthcareDeep_Learning\main\datasets\skin_lesion\mel",
        "expect_type": "skin",
        "expect_disease": "Melanoma",
    },
    # Test 8: Skin lesion (nv - benign mole)
    {
        "label": "Skin (Melanocytic Nevus)",
        "path": r"d:\healthcareDeep_Learning\main\datasets\skin_lesion\nv",
        "expect_type": "skin",
        "expect_disease": "Melanocytic Nevus",
    },
    # Test 9: COVID X-ray
    {
        "label": "X-ray (COVID-19)",
        "path": r"d:\healthcareDeep_Learning\main\datasets\covid_radiography\covid",
        "expect_type": "xray",
        "expect_disease": "COVID",
    },
    # Test 10: Cardiomegaly X-ray
    {
        "label": "X-ray (Cardiomegaly)",
        "path": r"d:\healthcareDeep_Learning\main\datasets\cardiomegaly\cardiomegaly",
        "expect_type": "xray",
        "expect_disease": "Cardiomegaly",
    },
]

results = []
for i, test in enumerate(TESTS, 1):
    folder = test["path"]
    imgs = sorted([f for f in os.listdir(folder) if f.lower().endswith(('.jpg','.jpeg','.png'))])
    img_path = os.path.join(folder, imgs[5])  # Pick 6th image to avoid edge cases

    print(f"\n{'='*70}")
    print(f"TEST {i}/10: {test['label']}")
    print(f"File: {imgs[5]}")
    print(f"Expected: type={test['expect_type']}, disease contains '{test['expect_disease']}'")
    print(f"{'='*70}")

    try:
        with open(img_path, "rb") as f:
            resp = requests.post(f"{BACKEND}/api/predict",files={"file": ("test.jpg", f, "image/jpeg")}, timeout=30)
        if resp.status_code != 200:
            print(f"  ❌ HTTP ERROR {resp.status_code}: {resp.text[:200]}")
            results.append(("FAIL", test["label"], f"HTTP {resp.status_code}"))
            continue

        d = resp.json()
        detected_type = d['image_type']['detected']
        type_conf = d['image_type']['confidence']
        top_disease = d['top_prediction']['disease']
        top_prob = d['top_prediction']['probability']

        # Check image type
        type_ok = detected_type == test['expect_type']
        # Check disease (partial match)
        disease_ok = test['expect_disease'].lower() in top_disease.lower()

        type_icon = "✅" if type_ok else "❌"
        disease_icon = "✅" if disease_ok else "❌"

        print(f"  {type_icon} Image Type: {detected_type} ({type_conf*100:.1f}%) [expected: {test['expect_type']}]")
        print(f"  {disease_icon} Top Disease: {top_disease} ({top_prob*100:.1f}%) [expected contains: {test['expect_disease']}]")

        # Show all predictions
        print(f"  All predictions:")
        for p in d['predictions']:
            bar = '#' * int(p['probability'] * 20)
            print(f"    {p['probability']*100:5.1f}% {p['disease']:25s} {bar}")

        status = "PASS" if (type_ok and disease_ok) else "FAIL"
        results.append((status, test["label"], f"type={detected_type}, top={top_disease} ({top_prob*100:.0f}%)"))

    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        results.append(("FAIL", test["label"], str(e)))

# Summary
print(f"\n{'='*70}")
print(f"SUMMARY: {sum(1 for r in results if r[0]=='PASS')}/{len(results)} tests passed")
print(f"{'='*70}")
for status, label, detail in results:
    icon = "✅" if status == "PASS" else "❌"
    print(f"  {icon} {label:35s} → {detail}")
