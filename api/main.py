from fastapi import FastAPI
from api.routes.dbscan import router as dbscan_router

app = FastAPI(title="ML Algorithms API")

app.include_router(dbscan_router)

@app.get("/")
def root():
    return {"status": "API is running"}
