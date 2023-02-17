## NLP Project - Semantic NER + audio transcription

(![image](https://user-images.githubusercontent.com/58753373/219784293-38addfa6-7efa-4612-bafa-186f9183e939.png)

#### 1 - Download audio from videos
```bash
  $ python src/download_audio.py
```


#### 2.1 - Transcribe audio to text with Whisper-LargeV2
```bash
  $ python src/transcribe.py
```


#### 2.2 - Download MultiNerd and train
```
  $ python src/multinerd_download.py

  $ bash bashs/run_ner_multinerd.sh
```


#### 3 - NER Pre annotate to labelstudio
```bash
  $ python src/hf_pre_annotate.py
```

#### 4 - Split into Kfold in annotated data and train
```bash
  $ python src/ner/kfold_split.py
  
  $ bash bashs/run_ner.sh
  # or init a hpo experiment
  #$ bash bashs/hpo_search.sh
```

### 5 - Prepare Corpus and load api for inference
```bash
  $ python src/semantic_search/prepare_corpus.py
  
  $ python src/api.py
```
