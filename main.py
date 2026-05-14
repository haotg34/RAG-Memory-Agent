from fastapi import FastAPI, Depends, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session
import os
import shutil
import time

from database.session import engine, Base, get_db
from services.chat_service import chat
from services.model_service import list_models
from modules.rag.indexer import index_document

app = FastAPI(title="AGI Assistant")

os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.on_event("startup")
def _startup_create_tables():
    for _ in range(30):
        try:
            Base.metadata.create_all(bind=engine)
            return
        except Exception:
            time.sleep(1)


class ChatRequest(BaseModel):
    user_id: str
    message: str
    session_id: str | None = None
    llm_provider: str | None = None
    llm_model: str | None = None


@app.get("/")
async def root():
    return FileResponse("static/index.html")


@app.post("/api/chat")
async def chat_endpoint(
    request: ChatRequest,
    db: Session = Depends(get_db),
):
    try:
        return chat(
            user_id=request.user_id,
            message=request.message,
            session_id=request.session_id,
            llm_provider=request.llm_provider,
            llm_model=request.llm_model,
            db=db,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/models")
async def models_endpoint(provider: str):
    try:
        return {"provider": provider, "models": list_models(provider)}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/upload")
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    try:
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        chunk_count = index_document(temp_path, db)

        os.remove(temp_path)
        return {"message": f"文档上传成功，共索引 {chunk_count} 个分块"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
