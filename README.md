# CaNNopy
Simple ML research framework/shell that focuses on repeatability and supports flexible and isolated diversity of tasks. 

### What it does
- Unified CLI (`./data`, `./train`, `./eval`, `./test`) routes through `src/app/entrypoint.py` to load layered YAML configs, build Hugging Face `TrainingArguments` plus custom `DataArguments`/`ModelArguments`, and dispatch to task modules.
- Supports dataset creation/downloading (ESG, multilingual keyword match, Slavic NER, EURLEX) and sequence/token model training and evaluation; writes outputs under `result/` with logs in `log/`.
- Uses PyTorch/Transformers with config-driven runs for reproducibility.

### Project structure
- `src/app/`: CLI machinery (arg parsing, config stacking, logging, discovery utilities).
- `src/data/`, `src/train/`, `src/eval/`, `src/test/`: command-specific task modules discovered by the entrypoint.
- `conf/`: layered YAML configs; data source docs and defaults live here.
- `data/`, `result/`, `log/`, `tmp/`: runtime assets and artifacts (keep large files out of git).
- `train/`, `eval/`: notebooks/scripts for experiments; adjust configs via `-c` to stack overrides.
- `requirements.txt`, `pyproject.toml`: dependencies and script entrypoints.

```shell
python -m venv .venv && source .venv/bin/activate
# or
python3 -m venv .venv && source .venv/bin/activate
# or
python3.13 -m venv .venv && source .venv/bin/activate
# as of this time not all libs support 3.14
```

```shell
pip install -U pip setuptools
```

```shell
pip install -r requirements.txt
```

Oneliner to reinitialize:
```shell

rm -Rf .venv && python3.13 -m venv .venv && source .venv/bin/activate && pip install -U pip setuptools && pip install -r requirements.txt
```

Set environment variables
```shell
set -a; source .env; set +a
```

## 1. Dataset mining / creation tasks

Prerequisites:
- Access to archive servers.
- The `CPTM_SPASS` environment variable is needed in `.env` file. 

### 1.1 ESG Slovene News Dataset
Create environmental, social, and governance dataset from the Slovene news data source:
```shell
./data create esg
```

### 1.2 Multilingual Keyword Match Slovene News Dataset
Create a dataset from a multilingual keywords matching data source:
```shell
./data create ml-kw-match
```

## 2. Multilingual Slavic NER task

### 2.1 Dataset preparation

Download and prepare Slavic NER dataset:
```shell
./data download ner
./data prepare ner
./data split ner
./data analyze ner
```

### 2.2 Training and evaluation
```shell
# train the google-bert/bert-base-multilingual-cased
./train token ner -c bert-mc.yaml
# train the FacebookAI/xlm-roberta-base
./train token ner -c xlmr.yaml
# train the jhu-clsp/mmBERT-base
./train token ner -c mm-bert.yaml
# not implemented yet
./train token ner -c gemma3-270m.yaml
# not implemented yet
./train token ner -c gemma3-1b-pt.yaml
# not implemented yet
./train token ner -c qwen3-1.7b.yaml
```
Now we can also run evaluation:
```shell
# evaluate the trained model
./eval token ner -c xlmr.yaml
./eval token ner -c mm-bert.yaml
```


### 2.3 Submit to [Slobench](https://slobench.cjvt.si/)

```shell
# download sample and test dataset to be annotated
./data download ner-slobench

# annotate the data
./eval token ner-slobench -c mm-bert.yaml
./eval token ner-slobench -c xlmr.yaml
```

## 3. Multilingual Slavic Retrieval task
Note: work in progress

### 3.1 Evaluate Machine Translation methods on Slobench dataset
```shell
# download sample and test dataset to be translated
./data download mt-slobench
```

```shell
# translate the data to the Slovenian language (see conf/data/translate for other languages)
./data translate mt-slobench -c sl.yaml -c gpt-oss-120b.yaml                    #  <-- remote Groq GPT OSS API model
./data translate mt-slobench -c sl.yaml -c gpt-5-mini.yaml                      #  <-- remote OpenAI API model
./data translate mt-slobench -c sl.yaml -c google-translate.yaml                #  <-- remote Google Translate API model
./data translate mt-slobench -c sl.yaml -c ollama-eurollm-9b-it.yaml            #  <-- local ollama (16GB VRAM GPU needed)
./data translate mt-slobench -c sl.yaml -c ollama-translategemma-27b.yaml       #  <-- local ollama (32GB VRAM GPU needed)
./data translate mt-slobench -c sl.yaml -c ollama-gams-it-dpo-trans-9b.yaml     #  <-- local ollama (16GB VRAM GPU needed)
./data translate mt-slobench -c sl.yaml -c ollama-gams-it-dpo-trans-9b-f16.yaml #  <-- local ollama (32GB VRAM GPU needed)
./data translate mt-slobench -c sl.yaml -c ollama-gams-sft-trans-9b.yaml        #  <-- local ollama (16GB VRAM GPU needed)
./data translate mt-slobench -c sl.yaml -c ollama-gams-sft-trans-9b-f16.yaml    #  <-- local ollama (32GB VRAM GPU needed)
./data translate mt-slobench -c sl.yaml -c seamless-m4t                         #  <-- local model (16GB VRAM GPU needed)
./data translate mt-slobench -c sl.yaml -c tiny-aya-water.yaml                  #  <-- local model (16GB VRAM GPU needed)
./data translate mt-slobench -c sl.yaml -c eurollm-9b-it.yaml                   #  <-- local model (16GB VRAM GPU needed)
./data translate mt-slobench -c sl.yaml -c translategemma-12b-it.yaml           #  <-- local model (32GB VRAM GPU needed)
```

### 3.2 Evaluate translation on BGE-M3 dataset

Download BGE-M3 dataset (beware it's size is `~24GB`):
```shell
# download and extract the BGE-M3 dataset
./data download bge-m3-ds
# copy English language source documents
./data prepare bge-m3-ds
```

Execute stratified sampling of the dataset to reduce the size:
```shell
./data sample bge-m3-ds
```

Translate the BGE-M3 sampled dataset:
```shell
# translate the data to the Slovenian language (see conf/data/translate for other languages)
./data translate bge-m3-ds-sampled -c sl.yaml -c gpt-oss-120b.yaml              #  <-- remote Groq API model
./data translate bge-m3-ds-sampled -c sl.yaml -c gpt-5-mini.yaml                #  <-- remote OpenAI API model
./data translate bge-m3-ds-sampled -c sl.yaml -c seamless-m4t                   #  <-- local model (16GB VRAM GPU needed)
./data translate bge-m3-ds-sampled -c sl.yaml -c ollama-eurollm-9b-it.yaml      #  <-- local ollama (16GB VRAM GPU needed)
./data translate bge-m3-ds-sampled -c sl.yaml -c ollama-translategemma-27b.yaml #  <-- local ollama (32GB VRAM GPU needed)
```

### 3.3 Evaluate translation on BGE-M3 dataset

```shell
# translate the data to the Slovenian language (see conf/data/translate for other languages)
./data translate bge-m3-ds -c sl.yaml -c gpt-oss-120b.yaml
```

## 4. Extreme Multilingual Multilabel Text Classification  

### Dataset downloading

Download EURLEX57K dataset:
```shell
./data download eurlex
```

Download NewsMon dataset (due to a license, you need a password to decrypt the archive):
```shell
./data download newsmon
```

## 5. News Stories

Download NewsMon dataset (due to a license, you need a password to decrypt the archive):
```shell
./data download newsmon
```

Embed the newsmon dataset to a ada_002 or BGE-M3 embeddings:
```shell
./data embed newsmon -c stories.yaml -c oai-ada_002.yaml
# or other models (see conf/data/embed for other embedding models)
./data embed newsmon -c stories.yaml -c bge-m3.yaml
./data embed newsmon -c stories.yaml -c oai-txt_ebd_3s.yaml
./data embed newsmon -c stories.yaml -c qwen3-ebd06.yaml
./data embed newsmon -c stories.yaml -c jina-ebd-v3.yaml
```

Now we can cluster the dataset with Louvain communities algorithm:
```shell
./data cluster newsmon -c stories.yaml -c oai-ada_002.yaml
./data cluster newsmon -c stories.yaml -c bge-m3.yaml
./data cluster newsmon -c stories.yaml -c oai-txt_ebd_3s.yaml
./data cluster newsmon -c stories.yaml -c qwen3-ebd06.yaml
./data cluster newsmon -c stories.yaml -c jina-ebd-v3.yaml
```

## 100. TODO :D
```shell

./data prepare newsmon
./data prepare eurlex
./data resample newsmon -c sl.yaml
./data resample newsmon -c sr.yaml
./data resample newsmon
./data sample newsmon -c hard_neg.yaml
./data split newsmon -c sl.yaml
./data split eurlex -c sl.yaml
./data analyze newsmon -c sl.yaml  
./data analyze newsmon -c sr.yaml  
./data analyze newsmon  
./data analyze eurlex
./train seqence newsmon -c xlmr.yaml
./train seqence newsmon -c mm-bert.yaml
./train seqence eurlex -c xlmr.yaml
./train seqence eurlex -c mm-bert.yaml
./train hard_neg newsmon -c bge-m3.yaml
./train hard_neg newsmon -c m-gte.yaml
./train hard_neg newsmon -c emb-gemma3.yaml
./eval seqence newsmon -c xlmr.yaml
./eval seqence newsmon -c mm-bert.yaml
./eval token ner -c xlmr.yaml
./eval token ner -c m-bert.yaml
./eval token ner -c mm-bert.yaml
./eval token ner -c gemma3-200m.yaml
./eval token ner -c gemma3-1b.yaml
```
