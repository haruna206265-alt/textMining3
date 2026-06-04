"""
Streamlit UI - Sentiment Analysis Logistic Regression
Analisis Sentimen Bahasa Indonesia
Author: Nadya Angelie Lislie (270231680)
"""

import streamlit as st
import requests, json, os, sys, time, re
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib

st.set_page_config(
    page_title="SentiAnalyze — Nadya Angelie Lislie",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.stApp { background: linear-gradient(160deg, #0a0a1a 0%, #0d1b2a 50%, #0a1628 100%); }
[data-testid="stSidebar"] {
    background: rgba(8,10,24,0.97) !important;
    border-right: 1px solid #3b82f6;
}
.card {
    background: rgba(15,23,42,0.9);
    border: 1px solid rgba(59,130,246,0.3);
    border-radius: 14px;
    padding: 22px;
    margin: 10px 0;
}
.result-pos {
    background: linear-gradient(135deg, rgba(16,185,129,0.15), rgba(5,150,105,0.08));
    border: 2px solid #10b981;
    border-radius: 16px; padding: 28px; text-align: center;
}
.result-neg {
    background: linear-gradient(135deg, rgba(239,68,68,0.15), rgba(185,28,28,0.08));
    border: 2px solid #ef4444;
    border-radius: 16px; padding: 28px; text-align: center;
}
.stButton > button {
    background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important; font-weight: 600 !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #1d4ed8, #1e40af) !important;
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(37,99,235,0.4) !important;
}
.stTextArea textarea {
    background: rgba(15,23,42,0.95) !important;
    border: 1px solid rgba(59,130,246,0.4) !important;
    border-radius: 10px !important; color: #e2e8f0 !important;
}
.author-tag {
    position: fixed; bottom: 14px; right: 14px;
    background: rgba(37,99,235,0.85); color: white;
    padding: 6px 14px; border-radius: 20px;
    font-size: 11px; font-weight: 600; z-index: 999;
}
footer, #MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Flask launcher ──────────────────────────────
FLASK_URL = "http://localhost:5000"
_started = False

def start_flask():
    global _started
    if _started: return
    _started = True
    import subprocess
    subprocess.Popen([sys.executable, "app.py"],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(2)

def api(endpoint, method="GET", payload=None):
    try:
        start_flask()
        url = FLASK_URL + endpoint
        r = requests.post(url, json=payload, timeout=30) if method == "POST" \
            else requests.get(url, timeout=10)
        return r.json()
    except Exception as e:
        return {"error": str(e)}

# ── Sidebar ──────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:20px 0 8px;'>
        <div style='font-size:44px;'>💬</div>
        <h2 style='color:#93c5fd; margin:6px 0 2px;'>SentiAnalyze</h2>
        <p style='color:#64748b; font-size:12px; margin:0;'>Analisis Sentimen Bahasa Indonesia</p>
        <div style='background:rgba(37,99,235,0.15); border-radius:8px; padding:8px; margin:10px 0;'>
            <p style='color:#bfdbfe; font-size:11px; margin:0; font-weight:600;'>👩‍💻 Nadya Angelie Lislie</p>
            <p style='color:#64748b; font-size:11px; margin:2px 0;'>NIM: 270231680</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    menu = st.radio("Menu", [
        "🔍 Prediksi Teks",
        "📋 Batch Analysis",
        "📊 Model Performance",
        "ℹ️ Tentang"
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("""
    <div style='background:rgba(37,99,235,0.1); border-radius:10px; padding:12px;'>
        <p style='color:#93c5fd; font-size:12px; font-weight:700; margin:0 0 6px;'>🔧 Model Info</p>
        <p style='color:#94a3b8; font-size:11px; margin:2px 0;'>📦 Logistic Regression</p>
        <p style='color:#94a3b8; font-size:11px; margin:2px 0;'>📊 TF-IDF (5000 fitur)</p>
        <p style='color:#94a3b8; font-size:11px; margin:2px 0;'>🔡 n-gram (1,2)</p>
        <p style='color:#94a3b8; font-size:11px; margin:2px 0;'>📚 4141 data</p>
        <p style='color:#3b82f6; font-size:11px; margin:6px 0 0; font-weight:600;'>🎯 Accuracy: 99.88%</p>
    </div>
    """, unsafe_allow_html=True)

# ── PREDIKSI TEKS ────────────────────────────────
if "Prediksi" in menu:
    st.markdown("""
    <div style='text-align:center; padding:28px 0 16px;'>
        <h1 style='color:#93c5fd; font-size:2.4em; margin-bottom:6px;'>💬 SentiAnalyze</h1>
        <p style='color:#64748b; font-size:1em;'>Analisis Sentimen Teks Bahasa Indonesia · Logistic Regression</p>
        <div style='display:inline-block; background:rgba(37,99,235,0.15); border-radius:20px; padding:3px 14px; margin-top:6px;'>
            <span style='color:#93c5fd; font-size:12px;'>✨ Accuracy 99.88% · 4141 Data</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### 💬 Input Teks")
        user_text = st.text_area("Teks", placeholder="Masukkan teks ulasan bahasa Indonesia...",
                                  height=130, label_visibility="collapsed")
        ec1, ec2 = st.columns(2)
        with ec1:
            if st.button("✅ Contoh Positif", use_container_width=True):
                st.session_state["ex"] = "Produk sangat bagus dan berkualitas, pelayanan ramah pengiriman cepat sekali!"
        with ec2:
            if st.button("❌ Contoh Negatif", use_container_width=True):
                st.session_state["ex"] = "Barang tidak sesuai gambar, kualitas buruk dan pengiriman sangat lama mengecewakan!"
        if "ex" in st.session_state:
            user_text = st.session_state.pop("ex"); st.rerun()
        go = st.button("🔍 Analisis Sekarang", use_container_width=True, type="primary")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### 📊 Statistik Model")
        s = api("/stats")
        if "accuracy" in s:
            st.metric("🎯 Accuracy", f"{round(s['accuracy']*100,2)}%")
            st.metric("📚 Data Train", f"{s['train_size']:,}")
            st.metric("🧪 Data Test", f"{s['test_size']:,}")
        else:
            st.info("Memuat statistik...")
        st.markdown('</div>', unsafe_allow_html=True)

    if go and user_text.strip():
        with st.spinner("🔍 Menganalisis sentimen..."):
            res = api("/predict", "POST", {"text": user_text})
        if "result" in res:
            r = res["result"]
            pred = r["prediction"]
            cp = r["confidence"]["positif"]
            cn = r["confidence"]["negatif"]
            st.markdown("<br>", unsafe_allow_html=True)
            if pred == "positif":
                st.markdown(f"""<div class='result-pos'>
                    <div style='font-size:56px;'>😊</div>
                    <h2 style='color:#10b981; margin:4px 0;'>SENTIMEN POSITIF</h2>
                    <p style='color:#a7f3d0; font-size:1.1em;'>Confidence: <b>{cp:.1f}%</b></p>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class='result-neg'>
                    <div style='font-size:56px;'>😞</div>
                    <h2 style='color:#ef4444; margin:4px 0;'>SENTIMEN NEGATIF</h2>
                    <p style='color:#fca5a5; font-size:1.1em;'>Confidence: <b>{cn:.1f}%</b></p>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(7, 2))
            fig.patch.set_facecolor('none'); ax.set_facecolor('none')
            bars = ax.barh(['😞 Negatif','😊 Positif'], [cn, cp],
                            color=['#ef4444','#10b981'], height=0.45)
            for b, v in zip(bars, [cn, cp]):
                ax.text(min(v+1,93), b.get_y()+b.get_height()/2, f'{v:.1f}%',
                        va='center', color='white', fontweight='bold', fontsize=12)
            ax.set_xlim(0,100)
            ax.tick_params(colors='white', labelsize=11)
            for sp in ax.spines.values(): sp.set_visible(False)
            plt.tight_layout()
            st.pyplot(fig, transparent=True); plt.close()

            with st.expander("🔎 Detail Preprocessing"):
                st.code(f"Original : {r['text_original']}\nCleaned  : {r['text_preprocessed']}", language="text")
    elif go:
        st.warning("⚠️ Masukkan teks terlebih dahulu!")

# ── BATCH ────────────────────────────────────────
elif "Batch" in menu:
    st.markdown("### 📋 Analisis Batch Teks")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    batch_input = st.text_area("Teks (satu per baris, maks 50):", height=180,
        placeholder="Produk bagus dan berkualitas!\nPengiriman sangat lambat dan mengecewakan\nBarang sesuai deskripsi")
    if st.button("🔍 Analisis Semua", use_container_width=True):
        texts = [t.strip() for t in batch_input.strip().split('\n') if t.strip()]
        if texts:
            with st.spinner(f"Menganalisis {len(texts)} teks..."):
                res = api("/predict/batch", "POST", {"texts": texts})
            if "results" in res:
                df_r = pd.DataFrame(res["results"])
                pos = (df_r['prediction']=='positif').sum()
                neg = (df_r['prediction']=='negatif').sum()
                c1,c2,c3 = st.columns(3)
                c1.metric("📝 Total", len(texts))
                c2.metric("✅ Positif", pos)
                c3.metric("❌ Negatif", neg)
                st.dataframe(df_r[['text','prediction_label','confidence_positif','confidence_negatif']],
                             use_container_width=True, height=280)
                fig2, ax2 = plt.subplots(figsize=(4,4))
                fig2.patch.set_facecolor('none'); ax2.set_facecolor('none')
                ax2.pie([pos,neg], labels=['Positif 😊','Negatif 😞'],
                        colors=['#10b981','#ef4444'], autopct='%1.1f%%',
                        textprops={'color':'white','fontsize':12}, startangle=90)
                ax2.set_title('Distribusi Sentimen', color='white', fontsize=12)
                st.pyplot(fig2, transparent=True); plt.close()
        else:
            st.warning("Masukkan minimal 1 teks!")
    st.markdown('</div>', unsafe_allow_html=True)

# ── PERFORMANCE ──────────────────────────────────
elif "Performance" in menu:
    st.markdown("### 📊 Model Performance")
    s = api("/stats")
    if "accuracy" in s:
        c1,c2,c3 = st.columns(3)
        c1.metric("🎯 Accuracy", f"{round(s['accuracy']*100,2)}%")
        c2.metric("📚 Train Size", f"{s['train_size']:,}")
        c3.metric("🧪 Test Size", f"{s['test_size']:,}")
        if s.get("report"):
            rep = s["report"]
            st.markdown("#### 📋 Classification Report")
            st.dataframe(pd.DataFrame({
                "Kelas":     ["😊 Positif","😞 Negatif","📊 Macro Avg"],
                "Precision": [f"{rep.get('Positif',{}).get('precision',0):.4f}",
                              f"{rep.get('Negatif',{}).get('precision',0):.4f}",
                              f"{rep.get('macro avg',{}).get('precision',0):.4f}"],
                "Recall":    [f"{rep.get('Positif',{}).get('recall',0):.4f}",
                              f"{rep.get('Negatif',{}).get('recall',0):.4f}",
                              f"{rep.get('macro avg',{}).get('recall',0):.4f}"],
                "F1-Score":  [f"{rep.get('Positif',{}).get('f1-score',0):.4f}",
                              f"{rep.get('Negatif',{}).get('f1-score',0):.4f}",
                              f"{rep.get('macro avg',{}).get('f1-score',0):.4f}"],
            }), use_container_width=True, hide_index=True)
    if os.path.exists("static/training_results.png"):
        st.image("static/training_results.png", caption="Confusion Matrix & Metrics", use_container_width=True)

# ── TENTANG ──────────────────────────────────────
elif "Tentang" in menu:
    st.markdown("### ℹ️ Tentang Proyek")
    st.markdown("""
    <div class="card">
        <h3 style='color:#93c5fd;'>💬 SentiAnalyze — Analisis Sentimen Bahasa Indonesia</h3>
        <p style='color:#94a3b8; line-height:1.8;'>
        Aplikasi analisis sentimen teks Bahasa Indonesia menggunakan model
        <b style='color:#93c5fd;'>Logistic Regression</b> dengan fitur
        <b style='color:#93c5fd;'>TF-IDF</b> (unigram + bigram).
        Dilatih pada 4141 ulasan e-commerce dan restoran berbahasa Indonesia.
        </p>
        <hr style='border-color:#1e293b;'>
        <table style='width:100%; color:#cbd5e1; border-collapse:collapse;'>
            <tr style='border-bottom:1px solid #1e293b;'><td style='padding:8px 0; color:#93c5fd;'><b>👩‍💻 Author</b></td><td>Nadya Angelie Lislie</td></tr>
            <tr style='border-bottom:1px solid #1e293b;'><td style='padding:8px 0; color:#93c5fd;'><b>🎓 NIM</b></td><td>270231680</td></tr>
            <tr style='border-bottom:1px solid #1e293b;'><td style='padding:8px 0; color:#93c5fd;'><b>📊 Dataset</b></td><td>4141 teks Bahasa Indonesia (Positif / Negatif)</td></tr>
            <tr style='border-bottom:1px solid #1e293b;'><td style='padding:8px 0; color:#93c5fd;'><b>🎯 Accuracy</b></td><td>99.88%</td></tr>
            <tr style='border-bottom:1px solid #1e293b;'><td style='padding:8px 0; color:#93c5fd;'><b>🔧 Model</b></td><td>Logistic Regression (C=1.0, solver=lbfgs)</td></tr>
            <tr style='border-bottom:1px solid #1e293b;'><td style='padding:8px 0; color:#93c5fd;'><b>📦 Features</b></td><td>TF-IDF max_features=5000, ngram_range=(1,2)</td></tr>
            <tr><td style='padding:8px 0; color:#93c5fd;'><b>⚙️ Stack</b></td><td>Flask + Streamlit + scikit-learn</td></tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="author-tag">👩‍💻 Nadya Angelie Lislie · 270231680</div>', unsafe_allow_html=True)
