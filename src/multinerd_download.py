from datasets import load_dataset


def df2json(df, filename):
    texts = []
    from tqdm.auto import tqdm

    for i in tqdm(range(len(df))):
        text_dict = {"tokens": df[i]["tokens"], "tags": df[i]["filtered_ner"]}
        texts.append(text_dict)

    import json

    with open(f"{filename}.json", "w", encoding="utf8") as file:
        for text in texts:
            json.dump(text, file, ensure_ascii=False)
            file.write("\n")


label2id = {
    "O": 0,
    "B-PER": 1,
    "I-PER": 2,
    "B-LOC": 3,
    "I-LOC": 4,
    "B-ORG": 5,
    "I-ORG": 6,
    "B-ANIM": 7,
    "I-ANIM": 8,
    "B-BIO": 9,
    "I-BIO": 10,
    "B-CEL": 11,
    "I-CEL": 12,
    "B-DIS": 13,
    "I-DIS": 14,
    "B-EVE": 15,
    "I-EVE": 16,
    "B-FOOD": 17,
    "I-FOOD": 18,
    "B-INST": 19,
    "I-INST": 20,
    "B-MEDIA": 21,
    "I-MEDIA": 22,
    "B-PLANT": 23,
    "I-PLANT": 24,
    "B-MYTH": 25,
    "I-MYTH": 26,
    "B-TIME": 27,
    "I-TIME": 28,
    "B-VEHI": 29,
    "I-VEHI": 30,
    "B-SUPER": 31,
    "I-SUPER": 32,
    "B-PHY": 33,
    "I-PHY": 34,
}
# BIO, PHY, SUPER, MYTH

id2label = {v: k for k, v in label2id.items()}
data = load_dataset("tner/multinerd", "pt")
df = data["test"]
df = df.map(lambda x: {"ner_tags": [id2label[i] for i in x["tags"]]})
remove_ents = ["BIO", "PHY", "SUPER", "MYTH"]
df = df.map(
    lambda x: {
        "filtered_ner": [i if i[2:] not in remove_ents else "O" for i in x["ner_tags"]]
    }
)

dataset = df.train_test_split(0.2, seed=10)
df2json(dataset["train"], "data/ner/pt-multinerd-train")
df2json(dataset["test"], "data/ner/pt-multinerd-dev")
# BIO, PHY, SUPER, MYTH


# from collections import Counter
# import numpy as np

# tags = np.array([id2label[tag][2:] for tags in df["tags"] for tag in tags if id2label[tag][0] == "B"])

# labels = dict(Counter(tags).most_common())
print("Done!")
