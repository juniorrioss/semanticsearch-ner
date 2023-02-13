import requests
import pandas as pd
import numpy as np



bertgle_endpoint="http://0.0.0.0:3000/predict"

header_auth = {
    "Authorization": "Bearer 545d1bb660522eae90980fe152f1c82215bb337b54d3257df321d6418f003f29"
}


df_query=pd.read_csv('data/query_testdata.csv').rename(columns={'Query':'query'})
df_query.drop(df_query[df_query['query'].isnull()].index.tolist(), axis="rows", inplace=True)
df_query.reset_index(drop=True, inplace=True)
vec_first=np.zeros(len(df_query))
vec_five=np.zeros(len(df_query))

for i in df_query.index.tolist():
    query=df_query.loc[i, 'query'].replace('\r','').split('\n')[0]
    if query=='':
        query=df_query.loc[i, 'query'].replace('\r','').split('\n')[1]
    if query=='':
        continue    
    
    print(query)
    print(i)
    
    df_query.loc[i, 'query']=query
    json_inf = {'query':query}
    results = requests.post(bertgle_endpoint, json=json_inf, headers=header_auth).json()
    # results = requests.post(bertgle_endpoint, json=json_inf, headers=header_auth).json()
    r_temp=[results[i]['title'] for i in range(len(results))]
    # esse ai é pra adicionar se achou no primeiro macho
    if df_query.loc[i, 'Title'] in r_temp[0]:
        vec_first[i]=1
    
    else: 
        # muda o -1 até onde tu quiser pegar
        if df_query.loc[i, 'Title'] in r_temp[1:-1]:
            vec_five[i]=1

print(f'Primeiros: {vec_first.sum()}')
print(f'Quintos: {vec_five.sum()}')

    
    
        





a=1
