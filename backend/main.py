# from fastapi import FastAPI, HTTPException
from fastapi import FastAPI
from pydantic import BaseModel
# import torch

# from backend.inference_utils import load_model, run_inference

# -- FastAPI app --
app = FastAPI()

# -- Input Schema --
class InferenceRequest(BaseModel):
    model_name: str
    spectrum: list[float]

@app.get("/")
def root():
    return {"message": "Polymer Aging Inference API is online"}

@app.post("/infer")
def infer(request: InferenceRequest):
    return{
        "prediction": "Stubbed Output",
        "class_index": 0,
        "logits": [0.0, 1.0],
        "class_labels": ["Stub", "Output"],
    }
# def infer(request: InferenceRequest):
#     try:
#         model = load_model(request.model_name)
#         result = run_inference(model, request.spectrum)
#         return result
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e)) from e