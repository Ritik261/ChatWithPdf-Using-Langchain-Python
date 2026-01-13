import os
from fastapi import FastAPI, UploadFile, File
from backend.schemas import ChatRequest, ChatResponse
from backend.rag import build_rag_chain

UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="Chat with PDF (Gemini)")

rag_chain = None

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global rag_chain

    file_path = f"{UPLOAD_DIR}/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())

    rag_chain = build_rag_chain(file_path)
    return {"message": "PDF uploaded and indexed successfully"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not rag_chain:
        return ChatResponse(answer="Please upload a PDF first.")

    response = rag_chain.invoke(request.question)
    return ChatResponse(answer=response.content)
