from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import torch
import logging
from prometheus_fastapi_instrumentator import Instrumentator


# Import Hugging Face libraries
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.data import prepare_input  # (we'll update this in Step 2/3)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("chatbot")

app = FastAPI(
    title="Production Customer Support Chatbot",
    description="A chatbot using pre-trained models and advanced retrieval.",
    version="1.0.0"
)

Instrumentator().instrument(app).expose(app)


# Load the pre-trained conversational model
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
model.eval()

class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    query: str
    retrieved_context: str
    response: str
    retrieval_accuracy: float

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query text is required.")
    
    try:
        # Prepare input (we will improve this in Steps 2 and 3)
        input_tensor, retrieved_context, accuracy = prepare_input(query)
    except Exception as e:
        logger.error("Data preparation failed: %s", str(e))
        raise HTTPException(status_code=500, detail="Data preparation failed.")
    
    try:
        # Combine the query with the retrieved context
        # Here we create an input string for the generation model.
        input_text = query + " " + retrieved_context + "\nResponse:"
        input_ids = tokenizer.encode(input_text, return_tensors="pt")
        # Generate a response (tweak generation parameters as needed)
        chat_history_ids = model.generate(
            input_ids,
            max_length=input_ids.shape[-1] + 50,  # Allow up to 50 new tokens
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,         # Enable sampling for more varied responses
            top_k=50,               # Limit the sampling pool
            top_p=0.95,             # Use nucleus sampling
            no_repeat_ngram_size=2  # Avoid repetitive phrases
        )
        generated_text = tokenizer.decode(
            chat_history_ids[0][input_ids.shape[-1]:],
            skip_special_tokens=True
        )
    except Exception as e:
        logger.error("Model inference error: %s", str(e))
        raise HTTPException(status_code=500, detail="Model inference failed.")
    
    return ChatResponse(
        query=query,
        retrieved_context=retrieved_context,
        response=generated_text,
        retrieval_accuracy=accuracy
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
