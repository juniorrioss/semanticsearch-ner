import pandas as pd
import json
from transformers import pipeline
from tqdm.auto import tqdm


nlp = pipeline("ner", model="model", aggregation_strategy="first", device=0)
nlp.tokenizer.model_max_length = 512

entity_to_preannot = ["PER", "ORG", "LOC", "CEL", "INST", "TIME"]
threshold_ent = 0.25


def preannotate(texts):
    json_data = []
    for text in tqdm(texts):
        ents = nlp(text)

        results = []
        predictions = []
        for i, ent in enumerate(ents):
            if (
                ent["entity_group"] in entity_to_preannot
                and ent["score"] > threshold_ent
            ):
                results.append(
                    {
                        # "id": i,
                        "from_name": "label",
                        "to_name": "text",
                        "type": "labels",
                        "value": {
                            "start": ent["start"],
                            "end": ent["end"],
                            "text": ent["word"],
                            "score": float(round(ent["score"], 2)),
                            "labels": [ent["entity_group"]],
                        },
                    }
                )
        predictions.append({"model_version": "one", "result": results})
        json_data.append(
            {
                "data": {
                    "text": text,
                },
                "predictions": predictions,
            }
        )

    return json_data


if __name__ == "__main__":
    import numpy as np
    from glob import glob
    import json

    folder_path = "data/texts/ciencia"
    texts = []

    # Load texts and split them into block of 320 words
    for path in glob(folder_path + "/*.parquet"):
        df = pd.read_parquet(path)
        text = " ".join(df["text"])
        split_text = text.split()
        if len(split_text) > 320:
            splits = np.arange(0, len(split_text), 320).tolist() + [len(split_text)]
            for i in range(len(splits) - 1):
                texts.append(" ".join(split_text[splits[i] : splits[i + 1]]))

    # Pre annotate texts
    json_data = preannotate(texts)

    # Save data
    with open("preanot_multinerd-025.json", "w", encoding="utf8") as f:
        json.dump(json_data, f, ensure_ascii=False)
