import random

import requests

url = "http://localhost:8000/predict"
headers = {"Content-Type": "application/json"}

# If you use API_KEY, uncomment:
# headers["X-API-Key"] = "your-secret-key"

for i in range(500):
    payload = {
        "sepal_length": round(random.uniform(4.3, 7.9), 1),
        "sepal_width": round(random.uniform(2.0, 4.4), 1),
        "petal_length": round(random.uniform(1.0, 6.9), 1),
        "petal_width": round(random.uniform(0.1, 2.5), 1),
    }

    response = requests.post(url, json=payload, headers=headers)
    print(i + 1, response.status_code, response.json())
