import os
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer

# Fayl yo‘llari
HERE = os.path.dirname(__file__)
DATASET_PATH = os.path.join(HERE, "..", "data", "dataset.csv")
MODELS_DIR = os.path.join(HERE, "..", "models")
ENCODER_DIR = os.path.join(MODELS_DIR, "semantic_encoder")   # ixtiyoriy: local saqlash uchun
CLASSIFIER_PATH = os.path.join(MODELS_DIR, "semantic_classifier.pkl")

def main():
    print(f"📥 CSV yuklanmoqda: {os.path.abspath(DATASET_PATH)}")
    df = pd.read_csv(DATASET_PATH)

    # --- Tozalash: bo'sh/Nan matnlarni olib tashlaymiz
    df["text"] = df["text"].astype(str).fillna("").str.strip()
    df = df[df["text"] != ""].copy()
    df["grade"] = df["grade"].astype(int)

    texts = df["text"].tolist()
    labels = df["grade"].tolist()
    print(f"🧾 Namuna soni: {len(texts)} | Sinflar: {sorted(set(labels))}")

    # --- 1) E5 encoder (app bilan mos)
    print("🔭 E5-base model yuklanmoqda... (birinchi marta ~200MB yuklanadi)")
    model_name = "intfloat/multilingual-e5-base"
    encoder = SentenceTransformer(model_name)  # device=CPU default; GPU bo'lsa avtodetektsiya qiladi

    # --- 2) Embeddinglar (E5 formati: 'query: ...')
    print("🧮 Embeddinglar hisoblanmoqda (E5 formati, 'query:' prefiksi)...")
    texts_prefixed = [f"query: {t}" for t in texts]
    X = encoder.encode(
        texts_prefixed,
        show_progress_bar=True,
        batch_size=32,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    # --- 3) Logistic Regression (ko‘p sinf)
    print("🤖 Logistic Regression o‘rgatilmoqda...")
    clf = LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        solver="saga",        # katta o'lchamli xususiyatlar uchun yaxshi
        multi_class="multinomial",
        n_jobs=-1,
        C=4.0,
        random_state=42,
    )
    clf.fit(X, labels)

    # --- 4) Saqlash
    print("💾 Model saqlanmoqda...")
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Encoder'ni lokalga saqlash ixtiyoriy (GitHub'ga yuklamaymiz — og'ir):
    try:
        encoder.save(ENCODER_DIR)
        print(f"   ↳ Encoder saqlandi: {os.path.abspath(ENCODER_DIR)}")
    except Exception as e:
        print(f"   ⚠️ Encoder saqlanmadi (ixtiyoriy): {e}")

    joblib.dump(clf, CLASSIFIER_PATH)
    print(f"   ↳ Classifier saqlandi: {os.path.abspath(CLASSIFIER_PATH)}")

    print("✅ Tayyor!")

if __name__ == "__main__":
    main()
