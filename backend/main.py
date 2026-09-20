from fastapi import FastAPI

app = FastAPI(title="FinSight AI")

@app.get("/")
def root():
    return {"message": "FinSight AI API is running"}
