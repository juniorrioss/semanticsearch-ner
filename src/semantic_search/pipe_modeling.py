from transformers import AutoModel, AutoTokenizer, pipeline
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
        return AutoModel.from_pretrained(model_name).to(
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
