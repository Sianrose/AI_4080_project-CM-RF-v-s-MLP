# Project context – Sign Language MNIST (PROJ_MIST)

**Use this file when you start a new chat.** Open a **new Cursor chat tab**, open this file (or reference it), and say: “Read PROJECT_CONTEXT.md and [your question or task].” That way the new chat has full project context without re-explaining everything.

---

## What we’re building

- **Translation engine**: Input = static image of a hand sign (A–Z). Output = predicted letter.
- **Two approaches**: (1) Classical ML (Random Forest) – “Control”. (2) Neural network (MLP) – “Experiment”. Deploy the winner as an app.
- **Dataset**: [Sign Language MNIST on Kaggle](https://www.kaggle.com/datasets/datamunge/sign-language-mnist) (datamunge).
- **Constraint**: No pre-trained models.

---

## Dataset (everyone must use the same)

- **Format**: CSV with `label` + `pixel1` … `pixel784` (one row = one 28×28 grayscale image).
- **Classes**: 24 letters (A–Z **excluding J and Z**). Labels **0–23** map to letters in order: A, B, C, D, E, F, G, H, I, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y.
- **Files**: `sign_mnist_train.csv`, `sign_mnist_test.csv` (~27.5k train, ~7k test).
- **Where**: On Kaggle the dataset is already available. Locally, put CSVs in `data/` (see README for download).

---

## Repo structure

```
PROJ_MIST/
  data/                    # CSVs (gitignored); see README to download
  src/
    data_loader.py         # Load CSVs, reshape, normalize, train/test
    labels.py              # 24 letters, label_to_letter(i)
  classical/               # Week 1
    train_rf.py            # Train Random Forest, save model + confusion matrix
    evaluate_classical.py  # Load model, evaluate, confusion matrix
  mlp/                     # Week 2
    model.py               # SignLanguageMLP + SignLanguageMLP_v2 (BatchNorm)
    train_mlp.py           # Training loop with W&B logging
  app/                     # Week 3
    app.py                 # Streamlit app: upload image → predicted letter + confidence
    best_mlp_model.pt      # Best MLP weights (download from Kaggle)
  notebooks/               # Kaggle notebooks (CM_MIST_sian.ipynb, MLP_MIST_sian.ipynb)
  requirements.txt
  README.md
  PROJECT_CONTEXT.md       # This file
```

---

## 3-week plan (short)

| Week | Goal | Deliverable |
|------|------|-------------|
| **1** | Classical baseline (Random Forest) | Confusion matrix for classical model |
| **2** | MLP in PyTorch/TensorFlow, W&B | W&B dashboard: MLP loss curves vs classical |
| **3** | Deployment | Streamlit app: upload PNG → predicted letter + confidence |

---

## Where we run things

- **Laptop**: 16 GB RAM, Intel Iris Xe, Core i7 12th gen. OK for **Week 1** (Random Forest) and **Week 3** (Streamlit). Fine for inference.
- **Week 2 (MLP training)**: Prefer **Kaggle Notebooks** (free GPU, dataset already there) or Google Colab. Download trained model and use it locally in Week 3.

---

## Team

- 3 people, **beginners** to ML and notebooks.
- **Week 1 roles**: Person A = data loader + labels. Person B = classical model + confusion matrix. Person C = evaluate + README.
- Rotate roles in Week 2 and 3 (see plan in `.cursor/plans/` or repo README).

---

## Technical choices so far

- **Classical model**: Random Forest (sklearn). Seed = 42 everywhere.
- **MLP framework**: PyTorch (chosen for Week 2).
- **Experiment tracking**: Weights & Biases (wandb).
- **App**: Streamlit, local. Preprocess upload to 28×28 grayscale, same normalization as training.

---

## Week 1 results (completed)

- **Baseline RF** (100 trees, depth=20): **82.19%** test accuracy.
- **Best tuned RF** (200 trees, depth=20): **83.27%** test accuracy.
- Deeper trees / unlimited depth made almost no difference (~83.2%).
- Saved model: `model_rf.joblib`. Confusion matrices and comparison chart saved.
- Dataset note: labels 0–24 in CSV; label 24 filtered out (keep 0–23 = 24 classes). After filtering: 26,337 train, 6,840 test.
- Kaggle dataset path: `/kaggle/input/datasets/datamunge/sign-language-mnist/`

---

## Key files

### Week 1
- `src/labels.py` – Letter list and `label_to_letter(i)`.
- `src/data_loader.py` – `get_data()` returning X_train, y_train, X_test, y_test.
- `classical/train_rf.py` – Train RF, save model + confusion matrix.
- `classical/evaluate_classical.py` – Load model, test set accuracy, confusion matrix.
- `notebooks/CM_MIST_sian.ipynb` – Week 1 Kaggle notebook (RF + tuning).

### Week 2
- `mlp/model.py` – Configurable MLP definition (PyTorch): `SignLanguageMLP` (no BN) and `SignLanguageMLP_v2` (with BatchNorm).
- `mlp/train_mlp.py` – Training loop with W&B logging.
- `notebooks/MLP_MIST_sian.ipynb` – Week 2 Kaggle notebook (MLP experiments + W&B).

### Week 3
- `app/app.py` – Streamlit app: upload PNG/JPG → predicted letter + confidence.
- `app/best_mlp_model.pt` – Best MLP weights (download from Kaggle Output tab and place here).

---

## Week 3 (deployment)

- **App**: `streamlit run app/app.py` → opens at `http://localhost:8501`.
- **Model file**: Download `best_mlp_model.pt` from the Kaggle notebook output and put it in `app/`.
- The app loads `SignLanguageMLP_v2` using the config saved inside the `.pt` file (hidden_sizes, activation, dropout).
- Preprocessing: uploaded image → grayscale → resize 28×28 → normalize /255 → flatten to 784.
- Displays: predicted letter, confidence %, top-5 predictions with progress bars, original vs preprocessed image.

---

## How to start a new chat with this context

1. Open a **new chat** in Cursor.
2. Say something like: “Read PROJECT_CONTEXT.md and [what you want to do].”
3. Or: “We’re on Week 1 / Week 2 / Week 3 of PROJ_MIST; [your question or task].”

This file is the single place that summarizes the project, dataset, structure, and choices for handoff.
