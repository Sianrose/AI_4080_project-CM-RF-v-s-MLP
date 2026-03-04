---
name: Sign Language MNIST Plan
overview: A 3-week plan to build a sign-language letter classifier (Classical ML vs MLP), track experiments with W&B, and deploy the best model in a Streamlit app, with work split for three complete beginners and a focus on clean, versioned code over notebook-only hacks.
todos: []
isProject: false
---

# Sign Language MNIST – 3-Week Plan for 3 Beginners

## Dataset (everyone should understand)

- **Source**: [Kaggle – Sign Language MNIST (datamunge)](https://www.kaggle.com/datasets/datamunge/sign-language-mnist)
- **Format**: CSV with `label` + `pixel1`…`pixel784` (one row = one 28×28 grayscale image).
- **Classes**: 24 letters (A–Z **excluding J and Z**). Labels 0–23 map to letters (e.g. 0→A, 1→B, …, 9→J skipped in mapping, so you’ll keep a fixed `index → letter` list).
- **Size**: ~27.5k train, ~7k test. Download as `sign_mnist_train.csv` and `sign_mnist_test.csv`.

**Important**: Store the CSVs in a folder like `data/` and add `data/` to `.gitignore` if you don’t want to commit large files; document in README how to download from Kaggle (and optionally use Kaggle API or manual download).

---

## Repo structure (keep it clean from day one)

Use a **single repo** with version control (Git), not a single messy notebook.

```
PROJ_MIST/
  data/                 # CSVs here (gitignored), README says how to download
  src/
    data_loader.py      # load CSVs, reshape to (N, 28, 28), train/test split, normalize
    labels.py           # LABELS = ['A','B',...,'Y'] (24 letters), label_to_letter(i)
  classical/            # Week 1
    train_svm.py        # or train_rf.py
    evaluate_classical.py
  mlp/                  # Week 2
    model.py            # MLP definition (PyTorch or TensorFlow)
    train_mlp.py        # training loop, W&B logging
    sweep_config.yaml   # optional W&B sweep for “varying architectures”
  app/                  # Week 3
    app.py              # Streamlit: upload PNG → predict → letter + confidence
  notebooks/            # optional: EDA, quick tests (not the source of truth)
  requirements.txt
  README.md
```

- **Source of truth**: scripts in `src/`, `classical/`, `mlp/`, `app/`. Notebooks are for exploration only.
- **Reproducibility**: Same random seed everywhere; `requirements.txt` with fixed versions (e.g. `scikit-learn==1.3.x`, `torch==2.x`, `streamlit`, `wandb`, `pandas`, `numpy`, `Pillow`).

---

## Week 1 – Baseline: Classical model (SVM or Random Forest)

**Goal**: One classical model (either SVM or Random Forest), trained on the same data pipeline, and a **confusion matrix** as the deliverable.

- **Data**: Use `data_loader.py` to load CSVs, reshape to (N, 28, 28), optionally flatten to (N, 784) for classical, normalize (e.g. /255), and expose `X_train`, `y_train`, `X_test`, `y_test`.
- **Model**: Pick **one** of:
  - **Random Forest** (e.g. `sklearn.ensemble.RandomForestClassifier`) – often easier for beginners (fewer hyperparameters, no scaling required if you use tree-based).
  - **SVM** (e.g. `sklearn.svm.SVC`) – needs scaling (e.g. StandardScaler); more sensitive to hyperparameters.
- **Deliverable**: Script that trains the model, runs on test set, and saves a **confusion matrix** (e.g. `sklearn.metrics.confusion_matrix` + `ConfusionMatrixDisplay`; save as image and/or log to W&B in Week 2 for comparison). Optionally also save the trained model (e.g. `joblib`) for later use in the app.

**Suggested split (3 people)**:

- **Person A (Data owner)**: Implements `src/data_loader.py` and `src/labels.py`, documents how to download data in README. Ensures everyone uses the same train/test and label mapping.
- **Person B (Classical trainer)**: Implements `classical/train_svm.py` or `train_rf.py`, trains the model, and produces the confusion matrix (script + saved figure).
- **Person C (Evaluation / docs)**: Writes `classical/evaluate_classical.py` (load model, run on test set, print accuracy and save confusion matrix), and starts README (project description, how to run, how to get data).

By end of Week 1 you have: one classical model, one confusion matrix, and a shared data pipeline everyone will reuse.

---

## Week 2 – MLP and Weights & Biases

**Goal**: Multi-layer perceptron (MLP) in **PyTorch or TensorFlow** (no pre-trained models). Run experiments with different architectures (e.g. number of hidden layers, units, activation functions). **Track everything in Weights & Biases**. Deliverable: **W&B dashboard** with loss curves for the MLP and, if possible, a comparison with the classical model (e.g. classical test accuracy as a horizontal line, or a separate “run” that logs classical metrics).

- **Model**: MLP only (no CNNs required by the brief): input 784 → hidden layers (e.g. 256, 128) → 24 outputs. Vary:
  - Number of hidden layers (e.g. 1 vs 2 vs 3).
  - Hidden sizes (e.g. 128 vs 256 vs 512).
  - Activation (ReLU, tanh, etc.).
- **Training**: Same data from `data_loader.py` (reshape to (N, 784), normalize). Use the same train/test split as Week 1.
- **W&B**: 
  - `wandb.init(project="sign-language-mnist", config={...})` with hyperparameters.
  - Log `loss` and `accuracy` (and optionally `val_loss`/`val_accuracy`) each epoch.
  - Optionally define a W&B Sweep to try several architectures; or run a few configs manually and log each as a run.
  - For “neural net vs classical”: either log the classical test accuracy as a metric in one W&B run (e.g. “classical_accuracy”) or add a note in the report/dashboard comparing the best MLP run to the classical baseline.
- **Deliverable**: Dashboard in W&B showing multiple MLP runs (loss curves, accuracy) and a clear comparison to the classical model (accuracy and/or confusion matrix). Save the **best MLP** (e.g. `mlp/best_model.pt` or `.keras`) for Week 3.

**Suggested split (3 people)**:

- **Person A**: Implements `mlp/model.py` (configurable MLP: layers, activations) and integrates it into a training script that uses `data_loader` and logs to W&B.
- **Person B**: Implements `mlp/train_mlp.py` (training loop, optimizer, loss, metrics, checkpointing best model). Runs 3–5 architecture variants and ensures all are logged to W&B.
- **Person C**: Sets up W&B project and (optional) sweep config; documents how to run experiments and where to find the dashboard; adds classical model metrics to W&B (e.g. one run that logs classical accuracy) so the dashboard shows “MLP vs classical”.

By end of Week 2 you have: W&B dashboard with MLP loss curves and comparison to classical, plus one saved “winner” model for deployment.

---

## Week 3 – Deployment (Streamlit app)

**Goal**: A **local dashboard** (e.g. Streamlit) where a user can **upload a PNG** of a hand sign and the app displays the **predicted letter** and **confidence score**. “Winner” = best of classical vs MLP (by validation/test accuracy); deploy that one.

- **Preprocessing**: Resize/crop uploaded image to 28×28, grayscale, same normalization as in training (e.g. /255). Handle different input sizes (e.g. Pillow resize to 28×28).
- **Model**: Load the chosen model (classical joblib or MLP state dict / Keras model) in `app/app.py`. Run inference and get class index + probability (for MLP use softmax; for sklearn use `predict_proba`).
- **UI**: 
  - File uploader for PNG (and optionally JPG).
  - “Predict” (or auto on upload): show predicted letter (use `labels.label_to_letter`) and confidence (e.g. max probability).
  - Optional: show a simple confusion-matrix or “runner-up” letters.
- **Deliverable**: Live demo: run `streamlit run app/app.py`, upload an image, show prediction and confidence. Prepare a short presentation (what you built, classical vs MLP results from W&B, how to run the app).

**Suggested split (3 people)**:

- **Person A**: Implements image preprocessing in the app (resize to 28×28, grayscale, normalize) and integrates the **classical** model path (load joblib, predict, predict_proba).
- **Person B**: Integrates the **MLP** model path (load PyTorch/TF model, inference, softmax probabilities) and writes the logic to choose “winner” (e.g. config or env: `USE_MLP=true`) and call the right predictor.
- **Person C**: Builds the Streamlit UI (upload, button, display letter + confidence), writes README run instructions and presentation outline/slides.

By end of Week 3 you have: one deployable app, one live demo, and a clear story (baseline → MLP experiments → deployed winner).

---

## Dependency overview

- **Week 1**: `pandas`, `numpy`, `scikit-learn`, `matplotlib` (confusion matrix), `joblib` (save model).
- **Week 2**: `torch` or `tensorflow`, `wandb`, same data stack.
- **Week 3**: `streamlit`, `Pillow`, same model-loading as in Week 1/2.

Single `requirements.txt` can list all; use a virtual environment from the start.

---

## Tips for complete beginners

1. **One shared repo, small commits**: Push often; avoid one big dump at the end.
2. **Reuse one data pipeline**: `data_loader.py` and `labels.py` are used by classical, MLP, and app. Change data only in one place.
3. **Fixed random seed**: e.g. `np.random.seed(42)`, `torch.manual_seed(42)`, so results are reproducible.
4. **No pre-trained models**: Your MLP is from scratch (no transfer learning); classical is sklearn only.
5. **“Clean pipeline” beats “highest accuracy”**: A readable, runnable, versioned repo with a working app and W&B dashboard will score better than a single notebook with a slightly higher accuracy.

---

## Where to run (no local GPU / 16 GB RAM)

Your laptop (Intel Core i7 12th gen, 16 GB RAM, Intel Iris Xe) is **enough for Week 1 and Week 3**; use **free cloud** for the heavy part (Week 2 MLP training) so you don’t hit RAM or CPU limits.

### What runs where


| Part                             | Your laptop (16 GB) | Why                                                                                        |
| -------------------------------- | ------------------- | ------------------------------------------------------------------------------------------ |
| **Week 1** – Random Forest / SVM | Yes                 | ~27k × 784 fits in RAM; sklearn is CPU-only and fine. RF is light; SVM may take 10–30 min. |
| **Week 2** – MLP training        | Prefer cloud        | Many epochs on 27k samples; GPU speeds this up a lot and avoids RAM spikes.                |
| **Week 3** – Streamlit app       | Yes                 | Only inference (one 28×28 image); almost no compute.                                       |


### Recommended platforms (all free tiers)

1. **Kaggle Notebooks** (best fit for you)
  - **Why**: The Sign Language MNIST dataset is already on Kaggle; no upload.
  - **Free**: GPU (P100) ~30 hours/week, 16 GB RAM, 20 GB disk.
  - **Use**: Create a notebook, add the dataset (“Sign Language MNIST” by datamunge), write or paste your `train_mlp.py` logic (or run scripts with `!python mlp/train_mlp.py`). Train there, then **download** the saved model (e.g. `best_model.pt`) and put it in your repo for Week 3.
  - **Tip**: Keep your repo as the source of truth; copy code into Kaggle for runs, then copy back any changes.
2. **Google Colab**
  - **Free**: GPU (T4) available, ~12 GB RAM.
  - **Use**: Upload the two CSVs to Google Drive (or use Colab’s “Files” or mount Drive), then run your data loader and `train_mlp.py` in a notebook. Download the trained model for Week 3.
  - **Tip**: `pip install wandb` in the first cell; W&B works from Colab.
3. **Local CPU fallback**
  - If you can’t use cloud: use a **small MLP** (e.g. 1–2 hidden layers, 128–256 units) and **small batch size** (e.g. 64). Training will be slow (maybe 1–2 hours) but 16 GB RAM is usually enough. PyTorch/TensorFlow on CPU will work.

### Workflow that fits your setup

- **Week 1**: Run classical training and evaluation **on your laptop** (same for all 3).
- **Week 2**: Do MLP training in **Kaggle Notebooks** (or Colab). Sync code via Git; only run training in the cloud; download `best_model.pt` (or `.keras`) and commit it (or store in Drive and document in README) for Week 3.
- **Week 3**: Run Streamlit **locally**; it only loads the saved model and runs inference—no GPU or big RAM needed.

### If you use Kaggle/Colab

- Install deps in the first cell: `pip install wandb` (and anything else not preinstalled).
- Set the same random seed as in your repo so results are reproducible.
- At the end of training, download the model file (Kaggle: “Output” → download; Colab: “Files” or Drive). Put it in `mlp/best_model.pt` (or your chosen path) so `app/app.py` can load it.

---

## Summary: who does what (rotating roles)


| Week | Person A                               | Person B                           | Person C                             |
| ---- | -------------------------------------- | ---------------------------------- | ------------------------------------ |
| 1    | Data loader + labels                   | Classical model + confusion matrix | Evaluate + README                    |
| 2    | MLP model + W&B integration            | Training loop + experiments        | W&B dashboard + classical comparison |
| 3    | Image preprocessing + classical in app | MLP in app + “winner” logic        | Streamlit UI + demo/presentation     |


You can rotate A/B/C each week so everyone touches data, training, and deployment.