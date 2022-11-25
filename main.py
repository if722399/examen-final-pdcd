from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import pickle
import re

def check_for_digits(string): 
    
    digit_regex = r'\d+' # busca todos los dígitos que aparezcan uno o más veces (juntos)
    digits = re.findall(digit_regex,string)
    return len(digits)

def get_predictors(string):
    p3 = lambda x: 1 if '"' in x else 0
    p5 = lambda x: 1 if '!' in (x) else 0
    return [[len(string), len(string.split(' ')), p3(string),check_for_digits(string), p5(string) ]]



app = FastAPI(
    title='API para generar predicciones en base una descripción de proyecto',
)

class input(BaseModel):
    description: str


# Indicación para que se ejecute en cuanto inicie la api
@app.on_event("startup")
def load_model():
    global Model  # Función para que sea global la variable
    with open(r"./Model/text_model.pickle", "rb") as f:
        Model = pickle.load(f)


@app.get("/")
def home():
    return{"Desc": "Everything is OK!"}


@app.get("/api/v1/classify")
def classify_project(model: input):
    prediction = round(Model.predict_proba(get_predictors(model.description))[0][1])
    decition = {0:'Not Funded',1:'Funded'}

    return {"Decition": decition.get(prediction,0),
            "Desc": "Predicción hecha correctamente"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="localhost", port=8000, reload=True)
