import requests

# url = "http://あなたのサーバーのURL/protected_resource"

# url = "http://127.0.0.1:5000/api/items"
# url = "http://DemoAppServer-dev-elb-265816914.ap-northeast-1.elb.amazonaws.com/api/items" # ELBに接続するDNS名
url = "http://dev-elb.demoappserver.com/api/items" # ELBからドメイン名でアクセス(HTTPだが、HTTPS接続になる)
url = "https://dev-elb.demoappserver.com/api/items" # ELBからドメイン名でアクセス(HTTPS)

# header version.
# API_KEY = "dp-testapikeyken"
API_KEY = "dp-testapikeyhori"

"""
# headerのみ version.
"""
# headers = {
#     # 'Authorization': 'Bearer your_api_key'
#     "category": "Action::RUNNING", "Authorization": API_KEY
# }
# response = requests.get(url, headers=headers)

# # arg version.
# # params = {"category": "Action::RUNNING", "Authorization": API_KEY}
# # response = requests.get(url, params=params)
# # response = requests.post(url, params=params)

# print(response.status_code)
# print(response.text)


"""
# headerとparams version.
"""
headers = {
    "Authorization": API_KEY,
    "User-Agent": "DemoApplication/1.0"
}
params = {
    "query": "python",
    "page": 1,
    "category": "Action::RUNNING",
}
response = requests.get(url, headers=headers, params=params, timeout=10)

if response.status_code == 200:
    # print(response.json())  # JSON形式のレスポンスを解析
    # pass
    print(response.text)
else:
    print("エラーが発生しました:", response.status_code)
    print(response.text)
# print(response.status_code)
# print(response.text)