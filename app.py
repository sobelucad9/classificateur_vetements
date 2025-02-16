import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import matplotlib.pyplot as plt
from PIL import Image

# Charger le modèle entraîné
MODEL_PATH = "classification_vetements_model.h5"
model = load_model(MODEL_PATH)

# Classes du modèle
class_labels = ["dress", "hat", "longsleeve", "outwear", "pants", "shirts", "shoes", "shorts", "skirt", "t-shirt"]

# 📌 Fonction de prétraitement
def preprocess_image(image):
    image = image.resize((128, 128))  # Redimensionner pour MobileNetV2
    image = image.convert("RGB")  # Assurer 3 canaux (RVB)
    img_array = np.array(image) / 255.0  # Normalisation [0,1]
    img_array = img_array.reshape(1, 128, 128, 3)  # Adapter aux dimensions attendues
    return img_array

# 📌 Fonction de prédiction
def predict_image(image):
    img_array = preprocess_image(image)  # Prétraitement
    pred = model.predict(img_array)  # Prédiction
    predicted_class = class_labels[np.argmax(pred)]  # Classe prédite

    # Résultats sous forme de DataFrame
    results_df = pd.DataFrame({'Classe': class_labels, 'Probabilité': pred.flatten()}).sort_values(by='Probabilité', ascending=False)

    return predicted_class, results_df

# Interface Streamlit
st.set_page_config(page_title="👕🧢 Classificateur de Vêtements", layout="centered")

st.title("👕🧢 Classificateur de Vêtements")
st.write("""
Téléchargez une image pour la classer.

**Catégories disponibles :**
- 👗 Dress
- 🧢 Hat
- 👕 Longsleeve
- 🧥 Outwear
- 👖 Pants
- 👔 Shirts
- 👟 Shoes
- 🩳 Shorts
- 👚 Skirt
- 👕 T-shirt
""")


uploaded_file = st.file_uploader("Choisissez une image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Charger l'image
    image = Image.open(uploaded_file)
    st.image(image, caption="Image téléchargée", use_container_width=True)

    # Prédiction
    predicted_class, results_df = predict_image(image)

    # Affichage du résultat
    st.subheader(f"🛍️ Classe prédite : **{predicted_class}**")

    # Afficher les probabilités sous forme de tableau
    st.write("### 🔍 Résultat détaillé de la classification :")
    st.dataframe(results_df.style.format({"Probabilité": "{:.2f}"}))
