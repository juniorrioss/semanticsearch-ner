from datasets import load_dataset
from glob import glob
import torch
import numpy as np
from prepare import EmbeddingsPipeline, getnerpipe
import pandas as pd

path_dataset = "data/corpus_embeddings/corpusss.parquet"  # glob("data/texts/ciencia/*")
dataset = load_dataset("parquet", data_files=[path_dataset])["train"]
corpus_embeddings = np.array(dataset["embeddings"])

reference_dataset = pd.read_parquet("data/corpus_embeddings/reference_table.parquet")


# retriever_name = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
retriever_name = "neuralmind/bert-base-portuguese-cased"

ner_path = "runs/overfited_ner/checkpoint-816"

retrivier_pipe = EmbeddingsPipeline(retriever_name)
ner_pipe = getnerpipe(ner_path)


query = "A psicanálise de Freud é considerada ciência?"
query_ents = ner_pipe(query)
query_ents = np.array([i["word"] for i in query_ents])
query_emb = retrivier_pipe(query)

scores = np.matmul(query_emb, corpus_embeddings.T)
# Cos with torch
# scores = torch.cosine_similarity(torch.from_numpy(query_emb), torch.from_numpy(corpus_embeddings))

top_indexes = scores.argsort()[-10:][::-1]
filtered_scores = scores[top_indexes]

filtered_data = dataset[top_indexes]

final_rank = {}

video_ids = np.array(filtered_data["video_id"])

new_scores = {}
for video_id in np.unique(filtered_data["video_id"]):
    video_ents = reference_dataset[reference_dataset["video_id"] == video_id][
        "entities"
    ].values[0]
    ents_intersection = np.isin(query_ents, video_ents)
    count_intersection = ents_intersection.sum()

    for block_idx in top_indexes[video_ids == video_id]:
        final_rank[block_idx] = scores[block_idx] * float(f"1.{count_intersection}")


# Sorting dict by values
final_rank = dict(sorted(final_rank.items(), key=lambda x: x[1], reverse=True))


## Get link and time
links_result = []
for block_idx in final_rank.keys():
    block_idx = int(block_idx)
    video = dataset[block_idx]["video_id"]
    link = reference_dataset.query(f"video_id == {video}")["link"].values[0]

    start = round(dataset[block_idx]["time_start"])
    youtube_video_id = link.split("=")[-1]
    timed_link = f"https://youtu.be/{youtube_video_id}?t={start}"
    links_result.append(timed_link)
print("a")
