from fastapi import FastAPI
from typing import List
from pydantic import BaseModel

from semantic_search.inference import BERTgle


path_dataset = "data/corpus_embeddings/corpus_multiqa.parquet"
reference_path = "data/corpus_embeddings/reference_table.parquet"
retriever_path = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
ner_path = "runs/ner_honesto_wd02/checkpoint-588"


# Instantiate BERTgle
searcher = BERTgle(retriever_path, ner_path, path_dataset, reference_path)

app = FastAPI(
    title="BERTgle API",
    description="API do modelo BERTgle - Busca Semântica e NER. Projeto Final da Disciplina de NLP - PPGCC UFG",
)


class PredictRequest(BaseModel):
    query: str


class Prediction(BaseModel):
    title: str
    link: str
    text: str
    score: float


@app.post("/predict", response_model=List[Prediction])
async def predict(request: PredictRequest):
    query = request.query
    links = searcher.search(query=query)
    predictions = [
        Prediction(
            title=link["title"],
            link=link["link"],
            text=link["text"],
            score=link["score"],
        )
        for link in links
    ]
    return predictions


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=3000)
