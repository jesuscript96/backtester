from fastapi.testclient import TestClient
from backend.main import app
import traceback

client = TestClient(app)

print("--- Testing /api/datasets ---")
try:
    response = client.get("/api/datasets")
    print("Status:", response.status_code)
    print("Body:", response.text)
except Exception as e:
    traceback.print_exc()

print("\n--- Testing /api/strategies ---")
try:
    response = client.get("/api/strategies")
    print("Status:", response.status_code)
    print("Body:", response.text)
except Exception as e:
    traceback.print_exc()
