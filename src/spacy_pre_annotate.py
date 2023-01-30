import spacy
import pandas as pd
import json

nlp = spacy.load("pt_core_news_lg")


def preannotate(texts):
    json_data = []
    for text in texts:
        doc = nlp(text)
        ents = doc.ents
        results = []
        predictions = []
        for i, ent in enumerate(ents):
            results.append(
                {
                    # "id": i,
                    "from_name": "label",
                    "to_name": "text",
                    "type": "labels",
                    "value": {
                        "start": ent.start_char,
                        "end": ent.end_char,
                        "text": ent.text,
                        "labels": [ent.label_],
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

    folder_path = "texts/ciencia"
    texts = []

    for path in glob(folder_path + "/*.parquet"):
        df = pd.read_parquet(path)
        text = " ".join(df["text"])
        split_text = text.split()
        if len(split_text) > 384:
            splits = np.arange(0, len(split_text), 384).tolist() + [len(split_text)]
            for i in range(len(splits) - 1):
                texts.append(" ".join(split_text[splits[i] : splits[i + 1]]))

    json_data = preannotate(texts)

    with open("pre_annot.json", "w", encoding="utf8") as f:
        json.dump(json_data, f, ensure_ascii=False)

# print([(w.text, w.pos_, w.ent_iob_, w.ent_type_) for w in doc])
