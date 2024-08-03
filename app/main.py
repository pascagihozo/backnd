from fastapi import FastAPI
from pydantic import BaseModel
import onnxruntime as ort
import pandas as pd
import joblib
from fastapi.middleware import cors
import numpy as np

class PredictionRequest(BaseModel):
    Location_Type: str
    Environment: str
    Age: int
    Gender: str
    Hearing_Protection_Used: str
    Hearing_Sensitivity: str
    Health_Issues: str
    Noise_Level_dB: int
    Duration_Minutes: int

class PredictionResponse(BaseModel):
    prediction: str

app = FastAPI()

model = None
preprocessor = None

app.add_middleware(
      cors.CORSMiddleware,
      allow_origins=["*"],
      allow_credentials=True,
      allow_methods=["*"],
      allow_headers=["*"]
)
async def load_model():
    global ort_session
    global preprocessor
    ort_session = ort.InferenceSession("./app/nihl.onnx")
    preprocessor = joblib.load('./app/preprocessor.pkl')

app.add_event_handler("startup", load_model)

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(input: PredictionRequest):
    df = pd.DataFrame([input.dict()])
    processed_df = preprocessor.transform(df)
    input_name = ort_session.get_inputs()[0].name
    prediction = ort_session.run(None, {input_name: processed_df.astype(np.float32)})[0][0]
    return PredictionResponse(prediction="Harmful" if prediction > 0.5 else "Harmless")