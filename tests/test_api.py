from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_chat_endpoint():
    payload = {"query": "I need help with my delayed order."}
    response = client.post("/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    # Check that the response contains the expected keys
    assert "retrieved_context" in data
    assert "response" in data
    # Simulated retrieval accuracy should be between 0 and 1
    assert 0 <= data.get("retrieval_accuracy", 0) <= 1

if __name__ == "__main__":
    test_chat_endpoint()
