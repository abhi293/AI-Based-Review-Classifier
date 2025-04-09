from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class InputData(BaseModel):
    text: str

@app.post("/predict/")
def predict(data: InputData):
    # Load model and process the input
    result = {"prediction": "positive"}  # Replace with actual model output
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
