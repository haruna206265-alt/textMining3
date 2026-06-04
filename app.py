"""
Sentiment Analysis API - Logistic Regression
Analisis Sentimen Bahasa Indonesia
Author: Nadya Angelie Lislie (270231680)
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib, re, os, json, sys
import pandas as pd

app = Flask(__name__)
CORS(app)

MODEL_PATH  = os.path.join("model_cache", "lr_model.pkl")
TFIDF_PATH  = os.path.join("model_cache", "tfidf_vectorizer.pkl")
HISTORY_PATH = os.path.join("model_cache", "training_history.json")

model = None
tfidf = None
training_history = {}

def load_models():
    global model, tfidf, training_history
    try:
        model = joblib.load(MODEL_PATH)
        tfidf = joblib.load(TFIDF_PATH)
        with open(HISTORY_PATH) as f:
            training_history = json.load(f)
        print("✅ Model berhasil dimuat.")
    except Exception as e:
        print(f"⚠️ {e}")

STOPWORDS_ID = {
    'yang','dan','di','ke','dari','ini','itu','dengan','untuk','pada',
    'ada','tidak','sudah','juga','saya','kami','kita','mereka','dia',
    'ia','anda','karena','namun','tapi','atau','jika','bisa','lebih',
    'sangat','akan','seperti','telah','masih','hanya','semua','banyak',
    'setelah','saat','ketika','selain','dalam','oleh','atas','antara',
    'hingga','sehingga','supaya','bahwa','lagi','pun','nya','apa','mau',
    'maka','serta','yaitu','yakni','apabila','sekali','terhadap','tentang',
    'sebuah','setiap','dapat','harus','boleh','walaupun','meski','begitu',
    'lalu','kemudian','sebelum','saja'
}

def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+|@\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = [w for w in text.split() if w not in STOPWORDS_ID and len(w) > 2]
    return ' '.join(words)

@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "name": "Sentiment Analysis API",
        "author": "Nadya Angelie Lislie (270231680)",
        "version": "2.0.0",
        "model": "Logistic Regression + TF-IDF",
        "endpoints": ["/predict", "/predict/batch", "/health", "/stats", "/train"]
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "model_loaded": model is not None})

@app.route("/stats", methods=["GET"])
def stats():
    if not training_history:
        return jsonify({"error": "Model belum dilatih"}), 400
    return jsonify(training_history)

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model belum dimuat"}), 503
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Field 'text' wajib diisi"}), 400
    text_clean = preprocess(data["text"])
    if not text_clean.strip():
        return jsonify({"error": "Teks tidak valid"}), 400
    vec = tfidf.transform([text_clean])
    pred = int(model.predict(vec)[0])
    proba = model.predict_proba(vec)[0]
    label = "Positif" if pred == 1 else "Negatif"
    return jsonify({"success": True, "result": {
        "text_original": data["text"],
        "text_preprocessed": text_clean,
        "prediction": label.lower(),
        "prediction_label": f"{label} {'😊' if pred == 1 else '😞'}",
        "confidence": {
            "negatif": round(float(proba[0]) * 100, 2),
            "positif": round(float(proba[1]) * 100, 2)
        }
    }})

@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    if model is None:
        return jsonify({"error": "Model belum dimuat"}), 503
    data = request.get_json()
    if not data or "texts" not in data:
        return jsonify({"error": "Field 'texts' wajib diisi"}), 400
    results = []
    for text in data["texts"][:50]:
        text_clean = preprocess(text)
        if not text_clean.strip():
            results.append({"text": text, "error": "Invalid"}); continue
        vec = tfidf.transform([text_clean])
        pred = int(model.predict(vec)[0])
        proba = model.predict_proba(vec)[0]
        label = "Positif" if pred == 1 else "Negatif"
        results.append({"text": text, "prediction": label.lower(),
            "prediction_label": f"{label} {'😊' if pred==1 else '😞'}",
            "confidence_positif": round(float(proba[1])*100, 2),
            "confidence_negatif": round(float(proba[0])*100, 2)})
    return jsonify({"success": True, "results": results, "count": len(results)})

@app.route("/train", methods=["POST"])
def train():
    global model, tfidf, training_history
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score
        df = pd.read_csv("dataset.csv")
        df['text_clean'] = df['text'].apply(preprocess)
        X, y = df['text_clean'], df['label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        tfidf_new = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
        X_train_vec = tfidf_new.fit_transform(X_train)
        lr_new = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
        lr_new.fit(X_train_vec, y_train)
        acc = accuracy_score(y_test, lr_new.predict(tfidf_new.transform(X_test)))
        joblib.dump(lr_new, MODEL_PATH); joblib.dump(tfidf_new, TFIDF_PATH)
        model, tfidf = lr_new, tfidf_new
        training_history = {"accuracy": float(acc), "train_size": len(X_train), "test_size": len(X_test)}
        with open(HISTORY_PATH, 'w') as f: json.dump(training_history, f)
        return jsonify({"success": True, "accuracy": round(acc*100, 2)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    load_models()
    app.run(debug=True, port=5000)
