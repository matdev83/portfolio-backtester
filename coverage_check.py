import pandas as pd, requests, re

def main():
    url='https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    tables=pd.read_html(url)
    current=list(tables[0].iloc[:,0])
    changes=list(tables[1].iloc[:,1].dropna())
    symbols=set([s for s in current+changes if isinstance(s,str) and s.isalpha() and len(s)<=5])
    print('Total symbols', len(symbols))
    seed=pd.read_csv('data/cusip_tickers_seed.csv', header=None, names=['cusip','ticker','bbg','name','country','flag'])
    missing=[s for s in symbols if s not in set(seed['ticker'])]
    print('Missing', len(missing))
    print('Sample missing', missing[:20])

if __name__=='__main__':
    main() 