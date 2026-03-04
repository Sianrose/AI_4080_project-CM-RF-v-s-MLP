# Deploy with Streamlit Community Cloud

1. **Push your app to GitHub**  
   Make sure the repo is pushed and includes:
   - `app/app.py` (main app)
   - `app/best_mlp_model.pt` (trained model; required for the app)
   - `mlp/model.py`, `src/labels.py`, `src/__init__.py`
   - Root `requirements.txt`

2. **Go to [share.streamlit.io](https://share.streamlit.io)**  
   Sign in with your GitHub account.

3. **New app**
   - Click **New app**
   - **Repository**: `YOUR_USERNAME/PROJ_MIST` (or your repo name)
   - **Branch**: `main`
   - **Main file path**: `app/app.py`
   - Leave **Advanced settings** as default (it uses the repo root and `requirements.txt`)

4. **Deploy**  
   Click **Deploy**. The first run can take a few minutes (installing PyTorch etc.). When it’s done, you’ll get a public URL like `https://your-app-name.streamlit.app`.

**Note:** If the model file is large and you don’t want it in the repo, you can later switch to loading it from a URL or Streamlit’s secrets; for a first deploy, committing `app/best_mlp_model.pt` is simplest.
