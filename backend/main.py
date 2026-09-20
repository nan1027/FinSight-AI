from fastapi import FastAPI

from backend.api.v1.router import router as v1_router
from backend.config import get_settings

settings = get_settings()

app = FastAPI(title=settings.APP_NAME)
app.include_router(v1_router, prefix=settings.API_V1_PREFIX)


@app.get("/")
def root():
    return {"message": "FinSight AI API is running"}
