import requests, urllib.parse, sys
url = 'https://www.spdrs.com/library-content/public/documents/etfs/us/fund-data/daily-us-en-us-spy-20110401.xls'
api = f'https://archive.org/wayback/available?url={urllib.parse.quote(url, safe="")}&timestamp=20110401'
print('API', api)
resp = requests.get(api, timeout=30)
print('Status', resp.status_code)
print(resp.text) 