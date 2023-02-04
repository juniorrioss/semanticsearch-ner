from transformers import BertModel, AutoTokenizer, pipeline
import pandas as pd
import torch


class EmbeddingsPipeline:
    """A simple custom transformers pipeline to get only CLS Bert pooler embeddings
    Inference only with one text
    """

    def __init__(
        self, model_name="neuralmind/bert-base-portuguese-cased", device="cpu"
    ):
        self.model_name = model_name
        self.device = device
        self.model, self.tokenizer = self.init_retriever(model_name)

    def init_retriever(self, model_name):
        return BertModel.from_pretrained(model_name).to(
            self.device
        ), AutoTokenizer.from_pretrained(model_name)

    def cls_pooling(self, model_output):
        return model_output.last_hidden_state[:, 0]

    def __call__(self, text):
        encoded_input = self.tokenizer(
            text, padding=True, truncation=True, return_tensors="pt", max_length=512
        ).to(self.device)

        with torch.no_grad():
            model_output = self.model(**encoded_input, return_dict=True)

        # Perform pooling
        embeddings = self.cls_pooling(model_output)
        return embeddings.detach().cpu().numpy()[0]


def getnerpipe(model_path, device="cpu"):
    ner = pipeline(
        "ner",
        model=model_path,
        aggregation_strategy="max",
        device=device,
    )
    ner.tokenizer.model_max_length = 512
    return ner


if __name__ == "__main__":
    import torch
    import numpy as np
    from glob import glob
    from tqdm import tqdm

    path_dataset = glob("data/texts/ciencia/*")
    link_reference = pd.read_parquet("data/links/ciencia.parquet")

    # model_name = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
    model_name = "neuralmind/bert-base-portuguese-cased"

    retrivier_pipe = EmbeddingsPipeline(model_name=model_name, device="cuda:0")
    ner_path = "runs/overfited_ner/checkpoint-816"

    ner_pipe = getnerpipe(ner_path, device="cuda:0")

    data_blocks = {
        "text": [],
        "embeddings": [],
        "video_id": [],
        "title": [],
        "entities": [],
        "time_start": [],
        "block_id": [],
    }
    data_videos = {"video_id": [], "title": [], "entities": [], "link": []}
    for idx, path in tqdm(enumerate(path_dataset), total=len(path_dataset)):
        texts = []
        embeddings = []
        entities_labels = []
        times_starts = []

        df = pd.read_parquet(path)
        df["len_text"] = df["text"].apply(lambda x: len(x.split()))
        accumulative_sum = df["len_text"].cumsum()
        step = round(len(accumulative_sum) / (accumulative_sum.max() / 320))

        splits = np.arange(0, len(df) + step, step).tolist()
        for i in range(len(splits) - 1):
            minor_split = df["text"][splits[i] : splits[i + 1]]
            if len(minor_split) < 5:
                continue
            minor_text = " ".join(minor_split)

            result = retrivier_pipe(minor_text)
            embeddings.append(result)

            # ner pipeline
            entity = ner_pipe(minor_text)
            entity_label = list(set([(i["word"], i["entity_group"]) for i in entity]))
            entities_labels.append(entity_label)

            # when block text starts
            time_start = df["start"][splits[i]]
            times_starts.append(time_start)

            # texts
            texts.append(minor_text)

        video_ids = [idx] * len(embeddings)
        block_ids = np.arange(0, len(embeddings)).tolist()
        title = (
            path.split("/")[-1]
            .replace(".parquet", "")
            .replace("？", "?")
            .replace("⧸", "/")
            # .translate(str.maketrans("", "", string.punctuation))
        )

        titles = [title] * len(embeddings)

        video_link = link_reference[link_reference["title"] == title]["link"].values[0]

        # Text Block section
        data_blocks["text"].extend(texts)
        data_blocks["video_id"].extend(video_ids)
        data_blocks["block_id"].extend(block_ids)
        data_blocks["embeddings"].extend(embeddings)
        data_blocks["title"].extend(titles)
        data_blocks["entities"].extend(entities_labels)
        data_blocks["time_start"].extend(times_starts)

        # Video Section
        data_videos["title"].append(title)
        data_videos["video_id"].append(idx)
        unique_entities = [
            ent[0] for video_ents in entities_labels for ent in video_ents
        ]
        data_videos["entities"].append(unique_entities)
        data_videos["link"].append(video_link)

    corpus_embedding = pd.DataFrame.from_dict(data_blocks)
    reference_table = pd.DataFrame.from_dict(data_videos)

    corpus_embedding.to_parquet("data/corpus_embeddings/corpus.parquet")
    reference_table.to_parquet("data/corpus_embeddings/reference_table.parquet")
