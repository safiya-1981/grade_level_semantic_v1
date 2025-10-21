import streamlit as st
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
import re
from docx import Document
import PyPDF2
import matplotlib.pyplot as plt

# =================== CONFIG ===================
st.set_page_config(
    page_title="Adabiy Matn Tavsiyachi — 5–9-sinflar",
    page_icon="📘",
    layout="wide"
)

# 🌈 Milliy gradient fon (bayroq ranglari)
page_bg = """
<style>
[data-testid="stAppViewContainer"] > .main {
    background: linear-gradient(180deg, #0099CC 0%, #ffffff 50%, #009933 100%);
    color: #000000;
}
[data-testid="stSidebar"] {
    background-color: rgba(240,240,240,0.9);
}
h2, h3, h4 { color: #004E7C !important; }
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# 🇺🇿 Sarlavha — faqat bayroq (gerb yo‘q)
flag_url = "https://upload.wikimedia.org/wikipedia/commons/8/84/Flag_of_Uzbekistan.svg"
st.markdown(
    f"""
    <div style="display:flex; align-items:center; justify-content:center; margin-top:-25px; margin-bottom:10px;">
        <h2 style="text-align:center; color:#004E7C; margin-right:20px;">
            📘 Adabiy Matn Tavsiyachi — 5–9-sinflar
        </h2>
        <img src="{flag_url}" alt="Bayroq" height="60" style="display:block;">
    </div>
    <div style="height:4px; background:linear-gradient(to right,#0099CC,#ffffff,#009933); margin-bottom:20px;"></div>
    """,
    unsafe_allow_html=True
)

st.caption("🇺🇿 O‘quvchilarga mos adabiy matn darajasini aniqlovchi mashina o‘rganish modeli (E5 / SBERT + Logistic Regression)")

# =================== YO‘LLAR ===================
HERE = Path(__file__).resolve().parent
MODELS_DIR = (HERE.parent / "models").resolve()
ENCODER_DIR = (MODELS_DIR / "semantic_encoder").resolve()
CLASSIFIER_PATH = (MODELS_DIR / "semantic_classifier.pkl").resolve()

# =================== KIRILL → LOTIN ===================
def kiril_to_latin(text: str) -> str:
    pairs = {
        "А":"A","а":"a","Б":"B","б":"b","В":"V","в":"v","Г":"G","г":"g","Д":"D","д":"d",
        "Е":"E","е":"e","Ё":"Yo","ё":"yo","Ж":"J","ж":"j","З":"Z","з":"z","И":"I","и":"i",
        "Й":"Y","й":"y","К":"K","к":"k","Л":"L","л":"l","М":"M","м":"m","Н":"N","н":"n",
        "О":"O","о":"o","П":"P","п":"p","Р":"R","р":"r","С":"S","с":"s","Т":"T","т":"t",
        "У":"U","у":"u","Ф":"F","ф":"f","Х":"X","х":"x","Ц":"S","ц":"s","Ч":"Ch","ч":"ch",
        "Ш":"Sh","ш":"sh","Щ":"Sh","щ":"sh","Ъ":"","ъ":"","Ы":"I","ы":"i","Ь":"","ь":"",
        "Э":"E","э":"e","Ю":"Yu","ю":"yu","Я":"Ya","я":"ya","Қ":"Q","қ":"q","Ғ":"G‘","ғ":"g‘",
        "Ҳ":"H","ҳ":"h","Ў":"O‘","ў":"o‘"
    }
    return "".join(pairs.get(ch, ch) for ch in text)

# =================== MODEL YUKLASH ===================
@st.cache_resource(show_spinner=False)
def load_models():
    """
    Cloud-da barqaror ishlashi uchun: agar lokal encoder papkasi bor va bo'sh bo'lmasa — shundan,
    aks holda HuggingFace'dan 'intfloat/multilingual-e5-base' yuklanadi.
    """
    HF_MODEL = "intfloat/multilingual-e5-base"

    encoder_path_ok = ENCODER_DIR.exists() and any(ENCODER_DIR.iterdir())
    if encoder_path_ok:
        enc = SentenceTransformer(ENCODER_DIR.as_posix(), device="cpu")
    else:
        enc = SentenceTransformer(HF_MODEL, device="cpu")

    if not CLASSIFIER_PATH.exists():
        st.error("❌ `models/semantic_classifier.pkl` topilmadi. Uni repoga joylang va ilovani qayta ishga tushiring.")
        st.stop()

    clf = joblib.load(CLASSIFIER_PATH)
    return enc, clf

encoder, clf = load_models()

# =================== FUNKSIYALAR ===================
def read_file(uploaded_file) -> str:
    ext = uploaded_file.name.lower().split(".")[-1]
    text = ""
    try:
        if ext == "txt":
            raw = uploaded_file.read()
            for enc_try in ("utf-8-sig", "utf-8", "cp1251", "cp1252"):
                try:
                    text = raw.decode(enc_try)
                    break
                except Exception:
                    continue
            if not text:
                text = raw.decode("utf-8", errors="ignore")
        elif ext == "docx":
            doc = Document(uploaded_file)
            text = "\n".join(p.text for p in doc.paragraphs)
        elif ext == "pdf":
            reader = PyPDF2.PdfReader(uploaded_file)
            for page in reader.pages:
                text += page.extract_text() or ""
        else:
            st.error("❌ Faqat .txt, .docx yoki .pdf fayllar qo‘llaniladi.")
    except Exception as e:
        st.error(f"Faylni o‘qishda xato: {e}")
    return text

def predict_text(text: str):
    # E5 formatida "query: ..." prefiksi
    inp = f"query: {text.strip()}"
    emb = encoder.encode([inp], normalize_embeddings=True)
    probs = clf.predict_proba(emb)[0]
    classes = clf.classes_
    idx = int(np.argmax(probs))
    pred = classes[idx]
    conf = float(probs[idx])
    details = sorted(zip(classes, probs), key=lambda x: int(x[0]))
    return pred, conf, details

# =================== SIDEBAR ===================
with st.sidebar:
    st.header("⚙️ Rejim va sozlamalar")
    mode = st.radio("Baholash usuli:", ["Matn yozish", "Fayl yuklash"])
    kiril_auto = st.checkbox("Kirilni avtomatik lotinga o‘girish", value=True)
    st.markdown("---")
    st.write("👩‍🏫 O‘qituvchi uchun:")
    st.text_input("PIN", type="password", key="pin")

    # 🔁 Yangi tekshiruv tugmasi — Streamlit 1.50 uchun
    if st.button("🔁 Yangi tekshiruv / Tozalash"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

# =================== ASOSIY QISM ===================
st.subheader("✏️ Matn kiriting yoki yuklang:")

text = ""
if mode == "Matn yozish":
    text = st.text_area("Matn:", height=200)
elif mode == "Fayl yuklash":
    uploaded = st.file_uploader("Faylni tanlang (.txt, .docx, .pdf)", type=["txt", "docx", "pdf"])
    if uploaded:
        text = read_file(uploaded)

# Kirilni avtomatik lotinga o‘girish
if text and kiril_auto and re.search(r"[А-Яа-яЁёЎўҚқҒғҲҳ]", text):
    text = kiril_to_latin(text)
    st.info("🔤 Kirill matn lotinga o‘girildi.")

st.markdown("---")

# =================== PREDIKTSIYA NATIJASI ===================
if st.button("✅ Baholash", type="primary"):
    if not text or not text.strip():
        st.warning("Iltimos, matn kiriting yoki fayl yuklang.")
    else:
        # Juda uzun matnlar uchun xavfsiz kesish
        if len(text) > 12000:
            text = text[:12000]
            st.caption("ℹ️ Juda uzun matn qisqartirildi (12,000 belgi).")
        with st.spinner("Semantik tahlil qilinmoqda..."):
            pred, conf, details = predict_text(text)

        st.success(f"**Eng mos sinf:** {pred}-sinf | **Ishonch:** {conf:.2f}")

        st.subheader("📊 Sinf bo‘yicha ehtimollar")
        labels = [f"{int(c)}-sinf" for c, _ in details]
        probs = [float(p) for _, p in details]

        fig, ax = plt.subplots(figsize=(5, 2.8))
        bars = ax.bar(labels, probs, color="#0099CC")
        for b in bars:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
                    f"{b.get_height():.2f}", ha='center', fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Ehtimol")
        ax.set_xlabel("Sinf")
        ax.set_title("Sinf bo‘yicha ehtimollar grafigi", fontsize=10)
        plt.tight_layout()
        st.pyplot(fig)

st.markdown("<hr>", unsafe_allow_html=True)
st.subheader("💬 Fikr-mulohaza")
st.text_area("Dastur haqida fikringiz (xatolar, takliflar)...")

# === Mualliflik va loyiha ma'lumotlari ===
st.markdown(
    """
    <div style='text-align:center; color:gray; margin-top:30px;'>
    <p><b>📘 Dastur muallifi:</b> Sattorova Sapura Beknazarovna</p>
    <p>Al-Beruniy nomidagi Urganch davlat universiteti<br>
    “Kompyuter ilmlari va Sun’iy intellekt texnologiyalari” kafedrasi o‘qituvchisi</p>
    <p>© 2025 — Adabiy Matn Tavsiyachi | Neuro-Symbolic Learning Lab 🇺🇿</p>
    </div>
    """,
    unsafe_allow_html=True
)
