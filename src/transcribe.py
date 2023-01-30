import whisper
import pandas as pd
import os
from tqdm.auto import tqdm


def transcribe_files(files, save_folder="texts"):
    print("[ INFO ] Loading model")

    model = whisper.load_model("large-v2", device="cuda")

    print("[ INFO ] Transcribe initialized")

    os.makedirs(save_folder, exist_ok=True)

    for path in tqdm(files):
        result = model.transcribe(path)

        df = pd.DataFrame(result["segments"])
        filename = path.split("/")[-1].replace(".m4a", ".parquet")
        df.to_parquet(save_folder + "/" + filename)

    print("[ INFO ] Done!")


if __name__ == "__main__":
    from glob import glob

    folder_path = "audios/ciencia"
    files = glob(folder_path + "/*.m4a")
    save_folder = "data/texts/ciencia"
    already_saved = os.listdir(save_folder)
    if len(already_saved) > 0:
        files = files[len(already_saved) :]
    transcribe_files(files, save_folder=save_folder)
