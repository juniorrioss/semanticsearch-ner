import yt_dlp


def download_audios(urls, save_folder="audios"):
    import os

    os.makedirs(save_folder, exist_ok=True)
    ydl_opts = {
        "format": "m4a/bestaudio/best",
        "outtmpl": {"default": f"{save_folder}/%(title)s.%(ext)s"},
        "ignoreerrors": True,
        # ℹ️ See help(yt_dlp.postprocessor) for a list of available Postprocessors and their arguments
        # "postprocessors": [
        #     {  # Extract audio using ffmpeg
        #         "key": "FFmpegExtractAudio",
        #         "preferredcodec": "m4a",
        #     }
        # ],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        error_code = ydl.download(urls)


if __name__ == "__main__":
    import pandas as pd

    df = pd.read_parquet("data/links/ciencia.parquet")

    # TRAIN  - 500
    save_folder = "audios/ciencia"
    links = df["link"][:500].tolist()
    download_audios(links, save_folder=save_folder)

    # TEST - 160
    save_folder = "audios/test"
    links = df["link"][500:].tolist()
    download_audios(links, save_folder=save_folder)

    print("Done!")
