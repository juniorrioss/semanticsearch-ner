import requests
import pandas as pd
from tqdm.auto import tqdm

bertgle_endpoint = "http://localhost:3000/search"
header_auth = {
    "Authorization": "Bearer 545d1bb660522eae90980fe152f1c82215bb337b54d3257df321d6418f003f29"
}


df = pd.read_csv("data/query_testdata.csv")
b = pd.read_parquet("data/corpus_embeddings/corpus_test_filtered.parquet")
rewards = []
for i in tqdm(range(len(df))):

    # query = "o maior erro de Einstein"
    sample = df.iloc[i]
    query = sample["Query"]
    label = sample["Title"]

    json_inf = {"query": query}
    predicts = requests.post(
        bertgle_endpoint, json=json_inf, headers=header_auth
    ).json()

    # Metrics
    all_titles = [i["title"] for i in predicts]
    reward = 0
    # IS TOP 1?
    if label == all_titles[0]:
        reward = 1
    # IS TOP 5?
    elif label in all_titles[:5]:
        reward = 0.5
    # IS TOP 10?
    elif label in all_titles[:10]:
        reward = 0.25

    rewards.append(reward)

df["Score"] = rewards

print("Top1: ", (df["Score"] == 1).sum())
print("Top5: ", (df["Score"] == 0.5).sum())
print("Top10: ", (df["Score"] == 0.25).sum())
print("Erro: ", (df["Score"] == 0).sum())
print("Done!")
