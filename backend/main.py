from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from backend.api.v1.router import router as v1_router
from backend.config import get_settings

settings = get_settings()
project_root = Path(__file__).resolve().parents[1]
frontend_dist = project_root / "frontend" / "dist"

app = FastAPI(title=settings.APP_NAME)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in settings.CORS_ORIGINS.split(",") if origin.strip()],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)
app.include_router(v1_router, prefix=settings.API_V1_PREFIX)


def _frontend_file(path: str | None = None):
    index_file = frontend_dist / "index.html"
    if not index_file.is_file():
        return None
    if path:
        candidate = (frontend_dist / path).resolve()
        if frontend_dist.resolve() not in candidate.parents:
            raise HTTPException(status_code=404, detail="Not found")
        if candidate.is_file():
            return FileResponse(candidate)
    return FileResponse(index_file)


@app.get("/", include_in_schema=False)
def root():
    response = _frontend_file()
    return response if response is not None else {"message": "FinSight AI API is running"}


@app.get("/{frontend_path:path}", include_in_schema=False)
def frontend_route(frontend_path: str):
    if frontend_path == settings.API_V1_PREFIX.lstrip("/") or frontend_path.startswith(
        f"{settings.API_V1_PREFIX.lstrip('/')}/"
    ):
        raise HTTPException(status_code=404, detail="Not found")
    response = _frontend_file(frontend_path)
    if response is None:
        raise HTTPException(status_code=404, detail="Not found")
    return response
