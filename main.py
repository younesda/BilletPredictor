from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import pandas as pd
import io

from utils import load_model, preprocess_data

# Charger le modèle une fois au démarrage
model = load_model("rf_model_060825.sav")

# Créer l'app FastAPI
app = FastAPI(title="Billet Prediction API")

# Servir le fichier HTML (interface frontend)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_root():
    return FileResponse("static/index.html")

@app.post("/predict-file/")
async def predict_file(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))  # Support pour CSV avec ; ou ,

        # Prétraitement (imputation + sélection des colonnes)
        df_processed = preprocess_data(df.copy(), model)

        # Prédictions
        predictions = model.predict(df_processed)
        probabilities = model.predict_proba(df_processed)[:, 1]  # Proba que Y=1

        # Ajouter les résultats au dataframe original
        df["prediction"] = predictions
        df["proba"] = probabilities.round(4)  # Probabilités arrondies à 4 décimales

        # Convertir en CSV pour retour
        output = io.StringIO()
        df.to_csv(output, sep=';', index=False)
        output.seek(0)

        return StreamingResponse(
            output,
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=predictions.csv"}
        )

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
