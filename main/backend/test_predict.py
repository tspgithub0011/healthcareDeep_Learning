"""Test with a cardiomegaly normal image to verify the fix."""
import requests

img_path = r"d:\healthcareDeep_Learning\main\datasets\cardiomegaly\normal\1.png"

resp = requests.post(
    "http://localhost:8000/api/predict",
    files={"file": ("1.png", open(img_path, "rb"), "image/png")},
)
data = resp.json()

print(f"Image type: {data['image_type']['detected']} ({data['image_type']['confidence']:.0%})")
print()
print(f"{'Model':>15} | {'Disease':>22} | {'Prob':>6} | Status")
print("-" * 65)
for p in data["predictions"]:
    m = p.get("model", "N/A")
    prob_pct = f"{p['probability']:.0%}"
    print(f"{m:>15} | {p['disease']:>22} | {prob_pct:>6} | {p['status']}")
print()
print(f"Top: {data['top_prediction']['disease']} ({data['top_prediction']['probability']:.0%})")
