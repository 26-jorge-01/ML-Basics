from fastapi import FastAPI
from pydantic import BaseModel
import pickle

# Crear una instancia de FastAPI
app = FastAPI()

# Cargar el modelo previamente entrenado (suponiendo que lo guardamos en un archivo)
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Definir el esquema de los datos de entrada
class InputData(BaseModel):
    age: int
    sex: int
    bmi: float
    children: int
    smoker: int
    region: int

# Crear un endpoint para hacer predicciones
@app.post("/predict/")
def predict(data: InputData):
    features = [[data.age, data.sex, data.bmi, data.children,
                 data.smoker, data.region]]
    prediction = model.predict(features)
    return {"prediction": prediction[0]}

# Ejecutar el servidor con uvicorn:
# uvicorn app:app --reload
