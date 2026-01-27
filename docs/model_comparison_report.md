# Model Comparison Report (Hybrid vs AE vs IF vs OCSVM)

This report compares four models using the **same hash‑clean holdout bench**:
Hybrid, Autoencoder (AE), Isolation Forest (IF), and One‑Class SVM (OCSVM).

---

## 1) Evaluation protocol (systematic)

Holdout construction (hash‑based, no leakage):
- LandFall: **all 6 samples** (held out).
- general_mal: **50 samples** (held out, random seed=42).
- benign: **100 DNG files only** (held out, random seed=42).

Training set excludes all holdout hashes:
- benign: 5773
- general_mal: 45
- LandFall: 0 (all held out)

Holdout artifacts:
- `outputs/bench_holdout_manifest.json`
- `outputs/bench_holdout_list.txt`
- `outputs/bench_holdout_device_list_relative.txt`

Holdout feature NPZs:
- Hybrid: `outputs/hybrid_features_holdout.npz`
- AE/IF/OCSVM: `outputs/anomaly_features_holdout.npz`

---

## 2) Models and feature sets (what’s different)

| Model | Training labels | Feature set | Notes |
| --- | --- | --- | --- |
| Hybrid (logistic regression + rules) | benign + general_mal | full byte histogram + structural + rule flags | Supervised, interpretable |
| AE (dense autoencoder) | benign only | full byte histogram + structural + entropy | Unsupervised, higher‑dim |
| Isolation Forest | benign only | `struct_bytes` (compact) | Lower‑dim for stability |
| One‑Class SVM | benign only | `struct_bytes` (compact) | RBF kernel, sensitive to distribution |

**Why `struct_bytes` for IF/OCSVM?**  
OCSVM can be unstable in high‑dim spaces; compact structural + byte summary stats capture
format layout while avoiding a 500+ dimensional histogram.

---

## 3) Holdout metrics (DNG‑only benign)

Thresholds:
- Hybrid: 0.45 (val‑optimized)
- AE: 6.232 (p99 val‑benign)
- IF: 0.686 (p99 val‑benign)
- OCSVM: 0.0959 (p99 val‑benign)

Holdout results (from `outputs/holdout_eval_models.json`,
`outputs/iforest_metrics_struct_bytes.json`, `outputs/ocsvm_metrics_struct_bytes.json`):

| Model | Benign FPR | general_mal recall | LandFall recall |
| --- | --- | --- | --- |
| Hybrid | 0/100 = **0.00** | 49/50 = **0.98** | 6/6 = **1.00** |
| AE (p99) | 0/100 = **0.00** | 47/50 = **0.94** | 6/6 = **1.00** |
| IF (p99) | 1/100 = **0.01** | 12/50 = **0.24** | 6/6 = **1.00** |
| OCSVM (p99) | 1/100 = **0.01** | 50/50 = **1.00** | 6/6 = **1.00** |

---

## 4) Critical critique of performance differences

### Hybrid
Why it performs well:
- Uses **supervised labels** and **rule flags**, so it can learn a clean boundary for known malware.
- Captures both **byte fingerprints** and **structural DNG anomalies**.

Limitations:
- Requires labeled malware; performance depends on how representative general_mal is.
- Rule flags can be brittle if attackers adjust opcode patterns.

### Autoencoder (AE)
Why it performs well:
- Learns **benign structure** and flags deviations without needing malware labels.
- Full byte histogram + structure + entropy captures broad deviations.

Limitations:
- Threshold is **sensitive** to benign distribution shift (different cameras/scanners).
- High‑dim input can amplify outliers; calibration is critical.

### Isolation Forest
Why it underperforms here:
- The benign and malware distributions overlap in the compact feature space.
- IF prefers isolating extreme outliers; many general_mal samples are not extreme enough.

Limitations:
- Low recall at p99 despite low FPR; not strong enough as a standalone detector here.

### One‑Class SVM
Why it performs extremely well here:
- The **benign DNG manifold** is tight; OCSVM learns a compact boundary.
- general_mal and LandFall fall far outside that boundary in this holdout.

Limitations:
- **Highly sensitive** to domain shift; new benign sources can inflate FPR.
- RBF boundary may **overfit** with small or homogeneous benign data.

---

## 5) Interpretation (what this means)

- **OCSVM looks best on this DNG‑only bench**, but it is also the most sensitive to
  distribution shift. If benign DNGs expand to scanners, lab DNGs, or rare encoders,
  FPR can rise sharply.
- **Hybrid is the safest “general” detector**: strong recall on general_mal and
  robust rule triggers for DNG opcode anomalies.
- **AE is a solid benign‑only fallback**, but thresholds must be calibrated per
  benign distribution. It is safer than IF and less brittle than OCSVM.
- **IF is the weakest baseline** here; it does not separate general_mal well in
  the compact feature space.

---

## 6) On‑device note (AE thresholding)

On device, AE uses a **manual threshold of 30** to control DNG false positives.
This reduces recall on general_mal compared to p99:

- DNG holdout (device): general_mal recall **0.66**, benign FPR **0**
- DNG holdout (p99, Linux): general_mal recall **0.94**, benign FPR **0**

This is the expected trade‑off when tightening the threshold to reduce FPR.

---

## 7) Limitations

- Only **6 LandFall samples**; recall estimates are unstable.
- **Benign bench is DNG‑only**; does not represent classic TIFF.
- Feature sets differ across models (Hybrid/AE use full hist; IF/OCSVM use compact features).

---

## 8) Recommendations

- Keep **Hybrid** as the primary detector for general malware.
- Use **AE or OCSVM** as a benign‑only anomaly layer when labels are missing.
- Re‑evaluate OCSVM with **broader benign sources** and multiple seeds.
- Add a **TIFF‑only benign bench** if you plan to scan classic TIFF at scale.
