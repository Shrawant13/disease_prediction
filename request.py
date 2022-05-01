import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'Life Expectance':25.3})

print(r.json())