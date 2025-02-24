from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import torch
import logging

from app.models import ChatbotModel
from app.data import prepare_input

# Configure production-level logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("chatbot")

app = FastAPI(
    title="Production Customer Support Chatbot",
    description="A production-grade chatbot using Transformer-based attention with large-scale FAQ data.",
    version="1.0.0"
)

# Load the model (in production, load pre-trained weights)
model = ChatbotModel()
model.eval()

class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    query: str
    retrieved_context: str
    response: str
    retrieval_accuracy: float  # Simulated measure of retrieval accuracy

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query text is required.")
    
    try:
        input_tensor, retrieved_context, accuracy = prepare_input(query)
    except Exception as e:
        logger.error("Data preparation failed: %s", str(e))
        raise HTTPException(status_code=500, detail="Data preparation failed.")
    
    try:
        with torch.no_grad():
            logits = model(input_tensor)
    except Exception as e:
        logger.error("Model inference error: %s", str(e))
        raise HTTPException(status_code=500, detail="Model inference failed.")
    
    # Simulated response: convert logits into dummy token words
    predicted_tokens = torch.argmax(logits, dim=-1).squeeze(0).tolist()
    response_words = ["word" + str(token) for token in predicted_tokens]
    response_text = " ".join(response_words)
    
    return ChatResponse(
        query=query,
        retrieved_context=retrieved_context,
        response=response_text,
        retrieval_accuracy=accuracy
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
