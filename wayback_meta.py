import requests, urllib.parse, sys
url = 'https://www.ssga.com/library-content/products/fund-data/etfs/us/holdings-daily-us-en-spy-20231013.xlsx'
api = f'https://archive.org/wayback/available?url={urllib.parse.quote(url, safe="")}&timestamp=20231015'
print('API', api)
resp = requests.get(api, timeout=30)
print('Status', resp.status_code)
print(resp.text) 