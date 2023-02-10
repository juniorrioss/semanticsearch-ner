from datasets import load_dataset
import numpy as np
from .pipe_modeling import EmbeddingsPipeline, getnerpipe
import pandas as pd
from nltk.stem.porter import PorterStemmer


class BERTgle:
    def __init__(self, retriever_path, ner_path, corpus_path, reference_path):
        self.ner_path = ner_path
        self.retriever_path = retriever_path
        self.corpus_path = corpus_path
        self.reference_path = reference_path
        self.stemmer = PorterStemmer()
        self.build_models()
        self.load_data()

    def build_models(self):
        self.embedder_pipe = EmbeddingsPipeline(self.retriever_path)
        self.ner_pipe = getnerpipe(self.ner_path)

    def load_data(self):
        self.reference_dataset = pd.read_parquet(self.reference_path)
        self.corpus_table = load_dataset("parquet", data_files=[self.corpus_path])[
            "train"
        ]
        self.corpus_embeddings = np.array(self.corpus_table["embeddings"])

    def reranker(self, scores, query_ents, topk=10):

        top_indexes = scores.argsort()[-topk:][::-1]
        filtered_data = self.corpus_table[top_indexes]

        final_rank = {}

        video_ids = np.array(filtered_data["video_id"])

        for video_id in np.unique(filtered_data["video_id"]):
            video_ents = self.reference_dataset.query(f"video_id == {video_id}")[
                "entities"
            ].values[0]
            ents_intersection = np.isin(query_ents, video_ents)
            count_intersection = ents_intersection.sum()

            for block_idx in top_indexes[video_ids == video_id]:
                multiplier = count_intersection / 20 + 1.0
                final_rank[block_idx] = scores[block_idx] * multiplier

        # Sorting dict by values
        ordered_rank = dict(
            sorted(final_rank.items(), key=lambda x: x[1], reverse=True)
        )

        return ordered_rank

    def generate_response(self, reranked):
        ## Get link and time
        links_result = []

        for block_idx in reranked.keys():
            block_idx = int(block_idx)
            video = self.corpus_table[block_idx]["video_id"]
            link = self.reference_dataset.query(f"video_id == {video}")["link"].values[
                0
            ]

            start = round(self.corpus_table[block_idx]["time_start"])
            youtube_video_id = link.split("=")[-1]
            timed_link = f"https://youtu.be/{youtube_video_id}?t={start}"
            text = self.corpus_table[block_idx]["text"]
            title = self.corpus_table[block_idx]["title"]

            links_result.append(
                {
                    "title": title,
                    "link": timed_link,
                    "text": text,
                    "score": reranked[block_idx],
                }
            )

            # texts.append(text)

        return links_result

    def search(self, query="A psicanálise de Freud é considerada ciência?"):

        # Retriever predict
        query_emb = self.embedder_pipe(query)

        # Ner predict
        query_ents = self.ner_pipe(query)
        query_ents = np.array([self.stemmer.stem(i["word"]) for i in query_ents])

        # Dot Product for score
        scores = np.matmul(query_emb, self.corpus_embeddings.T)
        ## Cosine Similarity with torch
        # scores = torch.cosine_similarity(torch.from_numpy(query_emb), torch.from_numpy(corpus_embeddings))

        reranked = self.reranker(scores, query_ents, topk=10)

        links = self.generate_response(reranked)

        return links


if __name__ == "__main__":

    path_dataset = "data/corpus_embeddings/corpus_multiqa.parquet"
    reference_path = "data/corpus_embeddings/reference_table.parquet"
    # retriever_path = "neuralmind/bert-base-portuguese-cased"
    retriever_path = "sentence-transformers/multi-qa-mpnet-base-dot-v1"

    ner_path = "runs/overfited_ner/checkpoint-816"

    print("[ INFO ] Loading BERTgle")

    # Instantiate BERTgle
    searcher = BERTgle(retriever_path, ner_path, path_dataset, reference_path)

    print("[ INFO ] Searching query ...")

    query = "maior erro do Einstein"
    # Do a search with a custom query
    links = searcher.search(query=query)

    print("[ INFO ] Best Match")
    print("Text: \n", links[0]["text"])
    print("Link: \n", links[0]["link"])

    # retriever_name = "sentence-transformers/multi-qa-mpnet-base-dot-v1"

    print("Done!")
