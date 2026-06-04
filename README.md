# 💬 SentiAnalyze — Analisis Sentimen Bahasa Indonesia

Aplikasi analisis sentimen teks Bahasa Indonesia menggunakan **Logistic Regression + TF-IDF**.

> **Author:** Nadya Angelie Lislie (270231680)

---

## 📊 Hasil Training

| Metric | Value |
|--------|-------|
| 🎯 Accuracy | **99.88%** |
| 📚 Training Data | 3,312 samples |
| 🧪 Test Data | 829 samples |
| 🏷️ Classes | Positif 😊 / Negatif 😞 |

---

## 📁 Struktur Proyek

```
sentiment_lr/
├── app.py                        # Flask Backend API
├── streamlit_app.py              # Streamlit Frontend UI
├── requirements.txt              # Dependensi Python
├── dataset.csv                   # Dataset (4141 ulasan Indonesia)
├── README.md
├── static/
│   └── training_results.png      # Plot hasil training
└── model_cache/
    ├── lr_model.pkl              # Model Logistic Regression
    ├── tfidf_vectorizer.pkl      # TF-IDF Vectorizer
    └── training_history.json     # Metrics & history
```

---

## ⚙️ Pipeline NLP

1. Lowercase
2. Hapus URL & mention
3. Hapus karakter non-alfabet
4. Hapus stopwords Bahasa Indonesia
5. TF-IDF Vectorization (5000 fitur, unigram + bigram)
6. Logistic Regression

---

## 🚀 Cara Menjalankan Lokal

```bash
pip install -r requirements.txt

# Terminal 1 — Flask API
python app.py

# Terminal 2 — Streamlit UI
streamlit run streamlit_app.py
```

---

## 🌐 Deploy ke Streamlit Cloud

1. Push ke GitHub
2. Buka [share.streamlit.io](https://share.streamlit.io)
3. Main file: `streamlit_app.py`
4. Deploy ✅

---

## 🔌 API Endpoints

| Method | Endpoint | Deskripsi |
|--------|----------|-----------|
| GET | `/` | Info API |
| GET | `/health` | Status model |
| GET | `/stats` | Metrik model |
| POST | `/predict` | Prediksi 1 teks |
| POST | `/predict/batch` | Prediksi banyak teks |
| POST | `/train` | Retrain model |

---

## 🛠️ Tech Stack

- **Backend:** Flask + scikit-learn
- **Frontend:** Streamlit (Dark Theme)
- **Model:** Logistic Regression (C=1.0, solver=lbfgs)
- **NLP:** TF-IDF + n-gram(1,2)
- **Bahasa:** Python 3.9+
