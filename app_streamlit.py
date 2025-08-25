import streamlit as st
import pandas as pd
import requests
from io import StringIO

# Nettoyer le cache
st.cache_data.clear()

# Config page
st.set_page_config(page_title="Billet Predictor", page_icon="💸", layout="wide")

# HEADER
st.markdown("""
<style>
h1 {color: #2E86C1; font-size: 3rem;}
</style>
""", unsafe_allow_html=True)

st.title("💸 Billet Predictor")
st.markdown("""
Bienvenue !  
Upload un fichier CSV contenant les billets à analyser et obtenez :
- ✅ Prédiction billet authentique ou non  
- 📊 Probabilité associée
""")

# SECTION UPLOAD
st.markdown("---")
st.subheader("📂 Upload du fichier CSV")
uploaded_file = st.file_uploader("Choisissez un fichier CSV", type="csv")

if uploaded_file is not None:
    st.success(f"✅ Fichier chargé : **{uploaded_file.name}**")
    st.info("Cliquez sur le bouton ci-dessous pour lancer la prédiction.")

    if st.button("🔍 Lancer la prédiction"):
        with st.spinner("Envoi du fichier à l'API et génération des prédictions..."):
            try:
                # Envoi du fichier à FastAPI
                response = requests.post(
                    "http://127.0.0.1:8000/predict-file/",
                    files={"file": (uploaded_file.name, uploaded_file, "text/csv")}
                )

                if response.status_code == 200:
                    csv_text = response.content.decode("utf-8")
                    df_result = pd.read_csv(StringIO(csv_text), sep=';')

                    st.success("🎉 Prédictions générées !")

                    # Metrics
                    col1, col2 = st.columns(2)
                    col1.metric("Billets authentiques", int(df_result['prediction'].sum()))
                    col2.metric("Billets non authentiques", int(len(df_result) - df_result['prediction'].sum()))

                    # Tableau
                    st.dataframe(df_result.style.highlight_max(subset=['proba'], color='lightgreen'))

                    # Téléchargement
                    st.download_button(
                        label="⬇️ Télécharger les résultats CSV",
                        data=csv_text,
                        file_name="predictions.csv",
                        mime="text/csv"
                    )
                else:
                    st.error(f"❌ Erreur API : {response.status_code}")
                    st.text(response.text)
            except Exception as e:
                st.error(f"❌ Impossible de contacter l'API : {e}")

# FOOTER
st.markdown("---")
st.markdown("""
Développé par **Younes HACHAMI** | Modèle Random Forest | 💡 ML pour la détection de billets
""")
