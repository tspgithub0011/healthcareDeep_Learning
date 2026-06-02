# Skin Lesion Classification — Abbreviation Guide

This contains the extra information on skin lession class.. like disease explanation and stuff
If you are a medical student then read it... might be useful for you....

This document explains all **7 skin lesion classes** used in our dataset and model. The dataset is based on the **HAM10000** (Human Against Machine with 10000 training images) — a large collection of dermoscopic images of common pigmented skin lesions.

---

## Class Overview

| Abbreviation | Full Name | Type | Risk Level |
|:---:|---|---|---|
| **akiec** | Actinic Keratosis | Pre-cancerous | ⚠️ Medium–High |
| **bcc** | Basal Cell Carcinoma | Malignant (Cancer) | 🔴 High |
| **bkl** | Benign Keratosis | Benign (Non-cancerous) | 🟢 Low |
| **df** | Dermatofibroma | Benign (Non-cancerous) | 🟢 Low |
| **mel** | Melanoma | Malignant (Cancer) | 🔴 Very High |
| **nv** | Melanocytic Nevus | Benign (Mole) | 🟢 Low |
| **vasc** | Vascular Lesion | Benign (Blood vessel) | 🟢 Low |

---

## Detailed Descriptions

### 1. `akiec` — Actinic Keratosis (Intraepithelial Carcinoma)

- **What it is:** A rough, scaly patch on the skin caused by years of sun exposure. Also known as **solar keratosis**.
- **Appearance:** Flat to slightly raised, rough or scaly patches. Usually pink, red, or brown. Commonly found on face, ears, scalp, neck, forearms, and backs of hands.
- **Risk Level:** ⚠️ **Pre-cancerous** — If left untreated, about 5–10% of actinic keratoses can progress into **squamous cell carcinoma** (a type of skin cancer).
- **Who it affects:** Most common in fair-skinned individuals over age 40 with significant sun exposure.
- **Treatment:** Cryotherapy (freezing), topical creams, photodynamic therapy, or surgical removal.

---

### 2. `bcc` — Basal Cell Carcinoma

- **What it is:** The **most common type of skin cancer**. It arises from the basal cells in the lowest layer of the epidermis.
- **Appearance:** Pearly or waxy bumps, flat flesh-colored or brown scar-like lesions, or bleeding/scabbing sores that heal and return. Often has visible blood vessels.
- **Risk Level:** 🔴 **Malignant (Cancer)** — While it rarely metastasizes (spreads to other organs), it can cause significant local tissue destruction if untreated.
- **Who it affects:** Most common skin cancer worldwide. Strongly linked to UV exposure. More frequent in fair-skinned people.
- **Treatment:** Surgical excision, Mohs surgery, radiation therapy, or topical treatments for superficial cases.

---

### 3. `bkl` — Benign Keratosis

- **What it is:** A group of **non-cancerous skin growths** that includes seborrheic keratoses, solar lentigines (age/liver spots), and lichen planus-like keratoses.
- **Appearance:** Waxy, slightly elevated, brown to black growths that look like they're "stuck on" the skin. Can also appear as flat brown spots.
- **Risk Level:** 🟢 **Benign (Harmless)** — These are non-cancerous and don't become cancerous. However, they can sometimes be confused with melanoma visually.
- **Who it affects:** Very common in adults over 50. Almost everyone develops at least one seborrheic keratosis during their lifetime.
- **Treatment:** Usually no treatment needed. Can be removed for cosmetic reasons via cryotherapy or curettage.

---

### 4. `df` — Dermatofibroma

- **What it is:** A common, **harmless skin growth** (benign fibrous histiocytoma). It's a small, firm bump made of fibrous tissue in the dermis (deeper skin layer).
- **Appearance:** Small (usually < 1 cm), firm, raised bumps. Color ranges from pink to light brown to dark brown. Characteristically dimples inward when squeezed ("dimple sign").
- **Risk Level:** 🟢 **Benign (Harmless)** — Completely benign and does not transform into cancer.
- **Who it affects:** More common in women. Often appears on the legs. May develop after minor injuries like insect bites or shaving nicks.
- **Treatment:** No treatment necessary. Can be surgically excised if bothersome or for cosmetic reasons.

---

### 5. `mel` — Melanoma

- **What it is:** The **most dangerous form of skin cancer**. It develops from melanocytes — the cells that produce melanin (skin pigment).
- **Appearance:** Typically an irregularly shaped mole with uneven borders, multiple colors (brown, black, red, white, blue), and a diameter > 6mm. Follow the **ABCDE rule**: Asymmetry, Border irregularity, Color variation, Diameter > 6mm, Evolving size/shape/color.
- **Risk Level:** 🔴 **Very High (Life-threatening)** — Melanoma is responsible for the majority of skin cancer deaths. It can metastasize rapidly to lymph nodes, lungs, liver, brain, and bones.
- **Who it affects:** Can occur at any age. Risk factors include UV exposure, fair skin, many moles, family history, and prior sunburns.
- **Treatment:** Early surgical excision is critical. Advanced stages may require immunotherapy, targeted therapy, chemotherapy, or radiation.
- **Survival:** 5-year survival rate is **99% if caught early** (Stage I), but drops significantly with late-stage diagnosis.

---

### 6. `nv` — Melanocytic Nevus (Mole)

- **What it is:** A common **benign mole** made of clusters of melanocytes. This is the most common skin lesion — nearly every adult has some.
- **Appearance:** Small, round or oval spots that are evenly colored (tan, brown, or black). Usually have well-defined borders and are uniform in shape.
- **Risk Level:** 🟢 **Benign (Harmless)** — The vast majority of moles are completely harmless. However, having many moles (>50) increases melanoma risk, and existing moles can rarely transform into melanoma.
- **Who it affects:** Universal — almost everyone has moles. They appear during childhood and adolescence and may fade with age.
- **Treatment:** No treatment needed. Should be monitored for changes (ABCDE criteria) that might indicate melanoma.

---

### 7. `vasc` — Vascular Lesion

- **What it is:** Abnormalities of blood vessels in the skin. This category includes **angiomas** (cherry angiomas), **angiokeratomas**, **pyogenic granulomas**, and **hemorrhages**.
- **Appearance:** Red, purple, or blue spots or bumps on the skin. Cherry angiomas are small, bright red dots. Angiokeratomas are dark red to black, rough-surfaced papules.
- **Risk Level:** 🟢 **Benign (Harmless)** — Vascular lesions are non-cancerous. Cherry angiomas are extremely common and harmless.
- **Who it affects:** Cherry angiomas increase with age (very common after 30). Pyogenic granulomas can occur at any age, often after minor trauma.
- **Treatment:** Usually cosmetic treatment only. Options include laser therapy, electrocautery, or cryotherapy.

---

## Dataset Statistics

| Class | Abbreviation | Image Count | % of Dataset |
|---|:---:|:---:|:---:|
| Melanocytic Nevus | `nv` | 6,705 | 67.0% |
| Melanoma | `mel` | 1,113 | 11.1% |
| Benign Keratosis | `bkl` | 1,099 | 11.0% |
| Basal Cell Carcinoma | `bcc` | 514 | 5.1% |
| Actinic Keratosis | `akiec` | 327 | 3.3% |
| Vascular Lesion | `vasc` | 142 | 1.4% |
| Dermatofibroma | `df` | 115 | 1.1% |
| **Total** | | **10,015** | **100%** |

> **Note:** The dataset is heavily imbalanced — `nv` (moles) makes up 67% of all images. The training pipeline uses **class-weighted loss** to compensate for this imbalance and ensure the model learns to recognize rare but dangerous classes like `mel` (melanoma) and `bcc` (basal cell carcinoma).

---

## Risk Classification Summary

### 🔴 Malignant (Requires Immediate Medical Attention)
- **mel** (Melanoma) — Most dangerous skin cancer
- **bcc** (Basal Cell Carcinoma) — Most common skin cancer

### ⚠️ Pre-cancerous (Requires Monitoring/Treatment)
- **akiec** (Actinic Keratosis) — Can progress to squamous cell carcinoma

### 🟢 Benign (Non-cancerous)
- **bkl** (Benign Keratosis) — Harmless age spots/growths
- **nv** (Melanocytic Nevus) — Common moles
- **df** (Dermatofibroma) — Harmless fibrous bumps
- **vasc** (Vascular Lesion) — Harmless blood vessel growths

---

## Clinical Importance

Early detection of **melanoma (`mel`)** is the primary clinical goal of this model. Melanoma accounts for only ~1% of skin cancers but causes the **majority of skin cancer deaths**. The 5-year survival rate drops from 99% (Stage I) to 27% (Stage IV), making early AI-assisted screening potentially life-saving.

The model also helps identify **basal cell carcinoma (`bcc`)** and **actinic keratosis (`akiec`)**, both of which benefit from early detection and treatment to prevent progression and tissue damage.
