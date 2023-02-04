import pandas as pd
import json


def conll2pandas(path: str, sep=" "):
    """Convert conll file to pandas dataframe

    Args:
        path (str): filename (eg. dataset.conll)

    Returns:
        pandas.DataFrame: pandas DataFrame with text and tags cols
    """
    with open(path, "r", encoding="UTF8") as f:
        texts = []
        labels = []

        words = []
        tags = []
        for line in f.readlines():
            if line.startswith("-DOCSTART-"):
                continue
            line_list = line.split(sep)
            if line_list[0] != "\n":
                words.append(line_list[0])
                tags.append(line_list[-1][:-1])
            else:
                texts.append(words.copy())
                labels.append(tags.copy())
                words.clear()
                tags.clear()

    df = pd.DataFrame()
    df["text"] = texts
    df["tags"] = labels

    return df


def pandas2json(df, fname: str):
    """Convert pandas to json file

    Args:
        df (pd.DataFrame): Dataframe Object
        fname (str): file name
    """

    texts = []
    for i in range(len(df)):
        text_dict = {"text": df["text"].iloc[i], "tags": df["tags"].iloc[i]}
        texts.append(text_dict)

    with open(fname, "w", encoding="utf8") as file:
        for text in texts:
            json.dump(text, file, ensure_ascii=False)
            file.write("\n")


if __name__ == "__main__":
    print("[ INFO ] Loading Conll Dataset")
    conll_data = conll2pandas("data/ner/first_ner.conll")
    print("[ INFO ] Conll Loaded")
    print("[ INFO ] Save Json Dataset")
    pandas2json(conll_data, "data/ner/first_ner.json")
    print("[ INFO ] Completed!")
