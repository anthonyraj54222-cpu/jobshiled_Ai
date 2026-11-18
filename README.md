# 🛡️ JobShield AI — Multi-Document Job Fraud Detection & User History

A machine learning application that analyzes job postings from **PDFs (`.pdf`), Images via OCR (`.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`, `.tiff`), Microsoft Word documents (`.docx`, `.doc`, `.rtf`, `.odt`), Plain Text (`.txt`, `.md`), Emails (`.eml`), Web pages (`.html`), Spreadsheets (`.csv`, `.xlsx`)**, or direct text input, predicts whether the posting is **Real or Fake** using a **4-Branch Hybrid Ensemble Model**, and records only your actual scans in a clean **User Scan History**.

**Dataset:** [Real / Fake Job Posting Prediction (EMSCAD)](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction) — 17,880 job postings (17,014 Real, 866 Fake).

## 1. Key Highlights

| Feature | Description |
| :--- | :--- |
| **Single Unified Webpage (`index.html`)** | Running `python app_server.py` (or `python member4_app.py`) in the VS Code terminal opens the exact same clean webpage (`index.html`) that opens when clicking **Go Live** |
| **Multi-Format Document & Image Support** | Accepts PDF, Image screenshots, Word (`.docx`/`.rtf`), `.txt`, `.eml`, `.csv`, `.xlsx`, `.html`, or pasted text |
| **Clean User-Only History** | Starts empty (`0` records) and logs only the postings and files you actually scan, with one-click view and delete controls |
| **Upgraded 4-Branch Hybrid Ensemble** | Combines 75,000 Sublinear TF-IDF Unigrams+Bigrams, Structured & Pure-Text NBSVM Logistic Regression, Sparse TF-IDF Cosine k-NN, and a 16-Signal Domain Risk Calibrator |

## 2. Model Evaluation Results (on 3,576 Unseen Test Postings)

| Metric | Original Baseline (Random Forest) | Upgraded 4-Branch Hybrid Ensemble |
| :--- | :---: | :---: |
| **Overall Accuracy** | 98.13% | **99.33%** |
| **ROC-AUC Score** | 96.40% | **99.63%** |
| **Fake Class Precision** | 93.44% | **96.27%** |
| **Fake Class Recall (Catch Rate)** | 65.90% (114 / 173) | **89.60% (155 / 173)** |
| **Fake Class F1-Score** | 77.29% | **92.81%** |

### Confusion Matrix (3,576 Unseen Test Postings)

| Actual \ Predicted | Predicted Real (0) | Predicted Fake (1) |
| :--- | :---: | :---: |
| **Actually Real (3,403)** | **3,397 (True Negatives)** | 6 (False Positives) |
| **Actually Fake (173)** | 18 (False Negatives) | **155 (True Positives)** |

## 3. Project Files

| File | Description |
| :--- | :--- |
| `index.html` | Unified web frontend for scanning text/files, viewing your scan history, and inspecting model benchmarks |
| `app_server.py` | Built-in Python web server that serves `index.html` at `http://127.0.0.1:8000`, connects `jobshield_history.db`, and opens your browser automatically |
| `member4_app.py` | Entry-point alias that launches `app_server.py` so both commands open the exact same webpage |
| `jobshield_model.py` | Core 4-Branch Hybrid Ensemble & 16-signal feature extraction engine |
| `member3_train_model.py` | Model training and evaluation script (`fraud_model.pkl`, `tfidf_vectorizer.pkl`, `model_metrics.json`) |
| `document_parser.py` | Multi-format document & OCR parser (`.pdf`, `.png`/`.jpg`, `.docx`, `.txt`, `.eml`, `.csv`, `.xlsx`) |
| `database.py` | SQLite database manager for `jobshield_history.db` |
| `sample_documents/` | Sample test files (`.pdf`, `.png`, `.docx`, `.txt`) |

## 4. How to Run in VS Code

1. Open the **`jobshieldAI`** folder in **VS Code** and open the terminal (**`Ctrl + ` `**).
2. Run the web application from the terminal (automatically opens **`http://127.0.0.1:8000`** with the exact same webpage as **Go Live**):
   ```powershell
   python app_server.py
   ```
   *(Or click **Go Live** in the bottom-right status bar of VS Code to open `index.html` directly).*
3. Optional — Re-train the model from scratch:
   ```powershell
   python member3_train_model.py
   ```