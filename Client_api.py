import requests

url = "http://127.0.0.1:5000/api/items"
# params = {"category": "food", "order": "desc"}
# params = {"category": "Action::RUNNING", "api_key_dp": "dp-testapikey"}

API_KEY = "dp-testapikeyken"
params = {"category": "Action::RUNNING", "Authorization": API_KEY}

# response = requests.get(url, params=params)
response = requests.post(url, params=params)

print(response.status_code)
print(response.text)