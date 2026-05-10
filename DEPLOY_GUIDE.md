# 🚀 Deploying Pro Stock Suite on Render (Free Tier)

Your app has **two separate services** — deploy them in this order:

---

## Step 1 — Prepare your GitHub repo

Put all these files in one GitHub repo (public or private):

```
your-repo/
├── backend_fastapi_4.py        ← your backend (no changes needed)
├── front_streamlit_render.py   ← modified frontend (reads BACKEND_URL env var)
├── requirements_backend.txt
├── requirements_frontend.txt
└── render.yaml
```

Push to GitHub.

---

## Step 2 — Deploy the BACKEND first

1. Go to **render.com** → Sign up / Log in (free)
2. Click **New +** → **Web Service**
3. Connect your GitHub repo
4. Fill in:
   - **Name**: `stock-backend` (or anything you like)
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements_backend.txt`
   - **Start Command**: `uvicorn backend_fastapi_4:app --host 0.0.0.0 --port $PORT`
   - **Plan**: Free
5. Under **Environment Variables**, add:
   - `PYTHON_VERSION` = `3.11.0`
6. Click **Create Web Service**

⏳ First build takes **5–10 minutes** (downloading ML libraries).

Once deployed, Render gives you a URL like:
`https://stock-backend-xxxx.onrender.com`

✅ **Copy this URL — you need it for Step 3.**

---

## Step 3 — Deploy the FRONTEND

1. Click **New +** → **Web Service** again
2. Connect the **same GitHub repo**
3. Fill in:
   - **Name**: `stock-frontend`
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements_frontend.txt`
   - **Start Command**:
     ```
     streamlit run front_streamlit_render.py --server.port $PORT --server.address 0.0.0.0 --server.headless true
     ```
   - **Plan**: Free
4. Under **Environment Variables**, add:
   - `PYTHON_VERSION` = `3.11.0`
   - `BACKEND_URL` = `https://stock-backend-xxxx.onrender.com`
     ← paste the URL you copied from Step 2
5. Click **Create Web Service**

⏳ Build takes ~1–2 minutes (only Streamlit + Plotly).

Your frontend URL will be: `https://stock-frontend-xxxx.onrender.com`

---

## ⚠️ Free Tier Limitations to Know

| Thing              | What happens on free tier                              |
|--------------------|--------------------------------------------------------|
| **RAM**            | 512 MB — enough for most models except TensorFlow      |
| **LSTM models**    | Disabled (TF not installed) — all other 8 models work |
| **Sleep after 15 min** | First request after idle takes ~30 s to wake up   |
| **CPU**            | 0.1 CPU — forecasts may take 30–90 s                  |
| **News Sentiment** | Works but DistilBERT (~260 MB) downloads on first use |

---

## 🧪 Test it locally first

```bash
# Terminal 1 — start backend
pip install -r requirements_backend.txt
uvicorn backend_fastapi_4:app --reload

# Terminal 2 — start frontend
pip install -r requirements_frontend.txt
streamlit run front_streamlit_render.py
# BACKEND_URL not set → defaults to http://127.0.0.1:8000  ✓
```

---

## 🆙 Upgrade tip (if you want LSTM models)

Switch to Render's **Starter plan** ($7/mo) for 1 GB RAM, then add this env var on the **backend** service:

```
INSTALL_TENSORFLOW = true
```

And add to `requirements_backend.txt`:
```
tensorflow-cpu>=2.16.0
```

---

## Folder structure summary

```
your-repo/
├── backend_fastapi_4.py          no changes
├── front_streamlit_render.py     API URL now from env var
├── requirements_backend.txt      all ML libs (no TensorFlow)
├── requirements_frontend.txt     streamlit + plotly only
└── render.yaml                   optional: auto-deploy both at once
```
