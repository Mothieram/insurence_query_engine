import requests

# Simple text query
response = requests.post(
    "http://localhost:8000/analyze/query",
    json="Does my policy cover roof leaks?"
)
print(response.json())

# Structured query
structured_query = {
    "query": "claim submission",
    "context": {
        "policy_type": "auto",
        "claim_amount": 5000
    }
}
response = requests.post(
    "http://localhost:8000/analyze/query",
    json=structured_query
)
print(response.json()) 