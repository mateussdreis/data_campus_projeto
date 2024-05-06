from fastapi import FastAPI # type: ignore
from pydantic import BaseModel # type: ignore
import numpy as np
import joblib
from typing import List
#Instanciando o fastapi
app = FastAPI()

class HouseFeatures(BaseModel):
    gender: float
    ssc_p:float
    hsc_p:float
    hsc_s:float
    degree_p:float
    degree_t:float
    workex:float
    specialisation:float

@app.post('/prever_colocacao/')

async def predict(features: List[HouseFeatures]):
    # Carregando o modelo
    model = joblib.load("modelo_finalizado_KMN.pkl")

    # Extraindo valores dos objetos do tipo BaseModel
    data = [list(vars(feature).values()) for feature in features]

    # Convertendo para matriz numpy para a previsão
    data_np = np.array(data)
    # Fazendo a previsão
    predictions = model.predict(data_np)

    # Transformando o resultado em um formato adequado para retorno no FastAPI
    return {"predictions": predictions.tolist()}