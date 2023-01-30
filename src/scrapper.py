from bs4 import BeautifulSoup
import pandas as pd


def generate_df_videos(file_path, save_file):
    with open(file_path, "r") as f:
        html = f.read()

    soup = BeautifulSoup(html, "html.parser")

    c = soup.find(id="contents")
    a_tags = c.find_all(id="video-title-link")

    titles = [t.attrs["title"] for t in a_tags]
    links = [a.attrs["href"] for a in a_tags]

    df = pd.DataFrame({"title": titles, "link": links})
    df.to_parquet(f"{save_file}.parquet")


def generate_df_playlist(file_path, save_file):
    import numpy as np

    with open(file_path, "r") as f:
        html = f.read()

    soup = BeautifulSoup(html, "html.parser")

    a_tags = soup.find_all(
        class_="yt-simple-endpoint style-scope ytd-playlist-video-renderer"
    )

    titles = [t.attrs["title"] for t in a_tags]
    links = [a.attrs["href"] for a in a_tags]

    df = pd.DataFrame({"title": titles, "link": links})
    df.to_parquet(f"{save_file}.parquet")


# generate_df_playlist(
#     file_path="htmls/socios_cortes.html", save_file="data/links/socios"
# )
generate_df_videos(file_path="data/htmls/ciencia.html", save_file="data/links/ciencia")
