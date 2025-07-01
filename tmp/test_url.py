import requests, sys, textwrap
url='https://duckduckgo.com/html/?q=facebook+cusip'
print('fetching…')
html=requests.get(url, headers={"User-Agent":"Mozilla/5.0"}, timeout=10).text
print('len', len(html))
print(html[:1000]) 