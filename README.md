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
- `pyproject.toml`: uv-managed dependencies, package metadata, and script entrypoints.

```shell
uv sync
```

Oneliner to reinitialize:
```shell

rm -Rf .venv && uv sync
```

This project targets Python 3.13 and 3.14. Use `uv python pin python3.13`
or `uv python pin python3.14` before syncing if you need to switch the
interpreter used by `.venv`.

Set environment variables
```shell
set -a; source .env; set +a
```

The default environment uses Torch 2.11, CUDA 13, and FlashAttention-2 via
custom third-party Linux wheels for Python 3.13 and 3.14.

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

### 1.3 IPTC-14 Slovene News Dataset
Create a dataset from Slovene news articles tagged with fourteen selected IPTC categories:
```shell
./data create iptc-14
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
./train token ner -c bert-mc
# train the FacebookAI/xlm-roberta-base
./train token ner -c xlmr
# train the jhu-clsp/mmBERT-base
./train token ner -c mm-bert
# not implemented yet
./train token ner -c gemma3-270m
# not implemented yet
./train token ner -c gemma3-1b-pt
# not implemented yet
./train token ner -c qwen3-1.7b
```
Now we can also run evaluation:
```shell
# evaluate the trained model
./eval token ner -c xlmr
./eval token ner -c mm-bert
```


### 2.3 SDTJ Paper Experiments

```shell
# run the 20-configuration Multi-8 learning-rate/dropout sweep
./sdjt-multi8-sweep.sh 2611

./data resample ner-sdjt -s data.split.seed=2611 -s data.sampling.seed=2611
./data analyze ner-sdjt -s data.split.seed=2611 -s data.sampling.seed=2611
./train token ner-sdjt -c mm-bert -s data.attributes.run_name=mono-bg -s train.seed=2611
./train token ner-sdjt -c mm-bert -s data.attributes.run_name=mono-cs -s train.seed=2611
./train token ner-sdjt -c mm-bert -s data.attributes.run_name=mono-hr -s train.seed=2611
./train token ner-sdjt -c mm-bert -s data.attributes.run_name=mono-pl -s train.seed=2611
./train token ner-sdjt -c mm-bert -s data.attributes.run_name=mono-ru -s train.seed=2611
./train token ner-sdjt -c mm-bert -s data.attributes.run_name=mono-sl -s train.seed=2611
./train token ner-sdjt -c mm-bert -s data.attributes.run_name=mono-sr -s train.seed=2611
./train token ner-sdjt -c mm-bert -s data.attributes.run_name=mono-uk -s train.seed=2611
./train token ner-sdjt -c mm-bert -s data.attributes.run_name=multi8 -s train.seed=2611
./train token ner-sdjt -c mm-bert -s data.attributes.run_name=multi12 -s train.seed=2611
 
./train token ner-sdjt -c mm-bert -s data.attributes.run_name=mono-sl-p10 -s train.seed=2611
./train token ner-sdjt -c mm-bert -s data.attributes.run_name=mono-sl-p25 -s train.seed=2611
./train token ner-sdjt -c mm-bert -s data.attributes.run_name=mono-sl-p50 -s train.seed=2611
./train token ner-sdjt -c mm-bert -s data.attributes.run_name=multi8-sl-p10 -s train.seed=2611
./train token ner-sdjt -c mm-bert -s data.attributes.run_name=multi8-sl-p25 -s train.seed=2611
./train token ner-sdjt -c mm-bert -s data.attributes.run_name=multi8-sl-p50 -s train.seed=2611
./train token ner-sdjt -c mm-bert -s data.attributes.run_name=multi12-sl-p10 -s train.seed=2611
./train token ner-sdjt -c mm-bert -s data.attributes.run_name=multi12-sl-p25 -s train.seed=2611
./train token ner-sdjt -c mm-bert -s data.attributes.run_name=multi12-sl-p50 -s train.seed=2611

./train token ner-sdjt -c mm-bert -s data.attributes.run_name=mono-sr-p10 -s train.seed=2611
./train token ner-sdjt -c mm-bert -s data.attributes.run_name=mono-sr-p25 -s train.seed=2611
./train token ner-sdjt -c mm-bert -s data.attributes.run_name=mono-sr-p50 -s train.seed=2611
./train token ner-sdjt -c mm-bert -s data.attributes.run_name=multi8-sr-p10 -s train.seed=2611
./train token ner-sdjt -c mm-bert -s data.attributes.run_name=multi8-sr-p25 -s train.seed=2611
./train token ner-sdjt -c mm-bert -s data.attributes.run_name=multi8-sr-p50 -s train.seed=2611
./train token ner-sdjt -c mm-bert -s data.attributes.run_name=multi12-sr-p10 -s train.seed=2611
./train token ner-sdjt -c mm-bert -s data.attributes.run_name=multi12-sr-p25 -s train.seed=2611
./train token ner-sdjt -c mm-bert -s data.attributes.run_name=multi12-sr-p50 -s train.seed=2611

./train token ner-sdjt -c mm-bert -s data.attributes.run_name=multi8-full-bg -s train.seed=2611
./train token ner-sdjt -c mm-bert -s data.attributes.run_name=multi8-full-cs -s train.seed=2611
./train token ner-sdjt -c mm-bert -s data.attributes.run_name=multi8-full-hr -s train.seed=2611
./train token ner-sdjt -c mm-bert -s data.attributes.run_name=multi8-full-pl -s train.seed=2611
./train token ner-sdjt -c mm-bert -s data.attributes.run_name=multi8-full-ru -s train.seed=2611
./train token ner-sdjt -c mm-bert -s data.attributes.run_name=multi8-full-sl -s train.seed=2611
./train token ner-sdjt -c mm-bert -s data.attributes.run_name=multi8-full-sr -s train.seed=2611
./train token ner-sdjt -c mm-bert -s data.attributes.run_name=multi8-full-uk -s train.seed=2611

./train token ner-sdjt -c mm-bert -s data.attributes.run_name=pretrain-multi7-full-bg -s train.seed=2611
./train token ner-sdjt -c mm-bert -s data.attributes.run_name=pretrain-multi7-full-cs -s train.seed=2611
./train token ner-sdjt -c mm-bert -s data.attributes.run_name=pretrain-multi7-full-hr -s train.seed=2611
./train token ner-sdjt -c mm-bert -s data.attributes.run_name=pretrain-multi7-full-pl -s train.seed=2611
./train token ner-sdjt -c mm-bert -s data.attributes.run_name=pretrain-multi7-full-ru -s train.seed=2611
./train token ner-sdjt -c mm-bert -s data.attributes.run_name=pretrain-multi7-full-sl -s train.seed=2611
./train token ner-sdjt -c mm-bert -s data.attributes.run_name=pretrain-multi7-full-sr -s train.seed=2611
./train token ner-sdjt -c mm-bert -s data.attributes.run_name=pretrain-multi7-full-uk -s train.seed=2611

./train token ner-sdjt -c mm-bert -s data.attributes.run_name=full-multi8 -s train.seed=2611
./train token ner-sdjt -c mm-bert -s data.attributes.run_name=full-multi12 -s train.seed=2611
./train token ner-sdjt -c mm-bert -s data.attributes.run_name=full-multi12-capaux -s train.seed=2611

# token-matched Croatian source-quality ablation; all three select and test on L8 minus Croatian
./train token ner-sdjt -c mm-bert -s data.attributes.run_name=multi7-no-hr -s train.seed=2611
./train token ner-sdjt -c mm-bert -s data.attributes.run_name=multi7-plus-hr500k -s train.seed=2611
./train token ner-sdjt -c mm-bert -s data.attributes.run_name=multi7-plus-hr-wikiann -s train.seed=2611

# delete all splits and repeat for each seed

./eval token ner-sdjt -c mm-bert
./data analyze ner-sdjt results -c mm-bert
```

### 2.4 Submit to [Slobench](https://slobench.cjvt.si/)

```shell
# download sample and test dataset to be annotated
./data download ner-slobench

# annotate the data
./eval token ner-slobench -c mm-bert
./eval token ner-slobench -c xlmr
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
./data translate mt-slobench -c sl -c gpt-oss-120b                    #  <-- remote Groq GPT OSS API model
./data translate mt-slobench -c sl -c gpt-5-mini                      #  <-- remote OpenAI API model
./data translate mt-slobench -c sl -c google-translate                #  <-- remote Google Translate API model
./data translate mt-slobench -c sl -c ollama-eurollm-9b-it            #  <-- local ollama (16GB VRAM GPU needed)
./data translate mt-slobench -c sl -c ollama-translategemma-27b       #  <-- local ollama (32GB VRAM GPU needed)
./data translate mt-slobench -c sl -c ollama-gams-it-dpo-trans-9b     #  <-- local ollama (16GB VRAM GPU needed)
./data translate mt-slobench -c sl -c ollama-gams-it-dpo-trans-9b-f16 #  <-- local ollama (32GB VRAM GPU needed)
./data translate mt-slobench -c sl -c ollama-gams-sft-trans-9b        #  <-- local ollama (16GB VRAM GPU needed)
./data translate mt-slobench -c sl -c ollama-gams-sft-trans-9b-f16    #  <-- local ollama (32GB VRAM GPU needed)
./data translate mt-slobench -c sl -c seamless-m4t                    #  <-- local model (16GB VRAM GPU needed)
./data translate mt-slobench -c sl -c tiny-aya-water                  #  <-- local model (16GB VRAM GPU needed)
./data translate mt-slobench -c sl -c eurollm-9b-it                   #  <-- local model (16GB VRAM GPU needed)
./data translate mt-slobench -c sl -c translategemma-12b-it           #  <-- local model (32GB VRAM GPU needed)
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
./data translate bge-m3-ds-sampled -c sl -c gpt-oss-120b              #  <-- remote Groq API model
./data translate bge-m3-ds-sampled -c sl -c gpt-5-mini                #  <-- remote OpenAI API model
./data translate bge-m3-ds-sampled -c sl -c seamless-m4t              #  <-- local model (16GB VRAM GPU needed)
./data translate bge-m3-ds-sampled -c sl -c ollama-eurollm-9b-it      #  <-- local ollama (16GB VRAM GPU needed)
./data translate bge-m3-ds-sampled -c sl -c ollama-translategemma-27b #  <-- local ollama (32GB VRAM GPU needed)
```

### 3.3 Evaluate translation on BGE-M3 dataset

```shell
# translate the data to the Slovenian language (see conf/data/translate for other languages)
./data translate bge-m3-ds -c sl -c gpt-oss-120b
```

## 4. Extreme Multilingual Multilabel Text Classification  

### Dataset preparation

Prepare EURLEX57K dataset:
```shell
./data download eurlex
./data prepare eurlex
./data embed eurlex -c bge-m3
# remove duplicates from the dataset (that's why we need to embed)
./data resample eurlex dedup -c bge-m3
./data split eurlex
./data analyze eurlex
# repeat embedding step with deduplicated data
./data embed eurlex -c bge-m3
./data sample eurlex hard_neg -c bge-m3
```

Prepare Slovene NewsMon dataset:
note: (due to a license, you need a password to decrypt the archive):

```shell
./data download newsmon
./data prepare newsmon -c sl
./data embed newsmon -c sl -c bge-m3
# remove duplicates from the dataset (that's why we need to embed)
./data resample newsmon dedup -c sl -c bge-m3
./data split newsmon -c sl
./data analyze newsmon -c sl
# repeat embedding step with deduplicated data
./data embed newsmon -c sl -c bge-m3
./data sample newsmon hard_neg -c sl -c bge-m3
```

Prepare Serbian NewsMon_sr dataset:
```shell
./data download newsmon
./data prepare newsmon -c sr
./data embed newsmon -c sr -c bge-m3
# remove duplicates from the dataset (that's why we need to embed)
./data resample newsmon dedup -c sr -c bge-m3
./data split newsmon -c sr
./data analyze newsmon -c sr
# repeat embedding step with deduplicated data
./data embed newsmon -c sr -c bge-m3
./data sample newsmon hard_neg -c sr -c bge-m3
```

Prepare Macedonian NewsMon_mk dataset:
```shell
./data download newsmon
./data prepare newsmon -c mk
./data embed newsmon -c mk -c bge-m3
# remove duplicates from the dataset (that's why we need to embed)
./data resample newsmon dedup -c mk -c bge-m3
./data split newsmon -c mk
./data analyze newsmon -c mk
# repeat embedding step with deduplicated data
./data embed newsmon -c mk -c bge-m3
./data sample newsmon hard_neg -c mk -c bge-m3
```

## 5. News Stories

Download NewsMon dataset (due to a license, you need a password to decrypt the archive):
```shell
./data download newsmon
./data prepare newsmon -c stories
```

Embed the newsmon dataset using the specified embedding model and settings:  
(see conf/task/embed, conf/model)

The Q8_0 GGUF variants use llama-cpp-python directly and download the selected
GGUF file from Hugging Face on first use.

```shell
./data embed newsmon -c stories -c oai-ada_002
./data embed newsmon -c stories -c oai-txt_ebd_3s
./data embed newsmon -c stories -c bge-m3
./data embed newsmon -c stories -c alib-gte-mmbert
./data embed newsmon -c stories -c qwen3-ebd-0.6b
./data embed newsmon -c stories -c qwen3-ebd-4b
./data embed newsmon -c stories -c qwen3-ebd-4b-256
./data embed newsmon -c stories -c qwen3-ebd-4b-128
./data embed newsmon -c stories -c qwen3-ebd-4b-64
./data embed newsmon -c stories -c qwen3-ebd-8b-q8_0-256
./data embed newsmon -c stories -c qwen3-ebd-8b-q8_0-128
./data embed newsmon -c stories -c qwen3-ebd-8b-q8_0-64
./data embed newsmon -c stories -c jina-ebd-v5-txts
./data embed newsmon -c stories -c jina-ebd-v5-txts-256
./data embed newsmon -c stories -c f2llm-v2-0.6b
./data embed newsmon -c stories -c ml-ebd-0.6b
./data embed newsmon -c stories -c ml-ebd-0.6b-256
./data embed newsmon -c stories -c arctic-ebd-l-v2
./data embed newsmon -c stories -c arctic-ebd-l-v2-256
```

We select a desired clustering model at specific threshold ("gold standard") and fit others to best match the selected model:
```shell
./data cluster newsmon fit -c stories -c oai-ada_002
./data cluster newsmon fit -c stories -c oai-txt_ebd_3s
./data cluster newsmon fit -c stories -c bge-m3
./data cluster newsmon fit -c stories -c alib-gte-mmbert
./data cluster newsmon fit -c stories -c qwen3-ebd-0.6b
./data cluster newsmon fit -c stories -c qwen3-ebd-4b
./data cluster newsmon fit -c stories -c qwen3-ebd-4b-256
./data cluster newsmon fit -c stories -c qwen3-ebd-4b-128
./data cluster newsmon fit -c stories -c qwen3-ebd-4b-64
./data cluster newsmon fit -c stories -c qwen3-ebd-8b-q8_0-256
./data cluster newsmon fit -c stories -c qwen3-ebd-8b-q8_0-128
./data cluster newsmon fit -c stories -c qwen3-ebd-8b-q8_0-64
./data cluster newsmon fit -c stories -c jina-ebd-v5-txts
./data cluster newsmon fit -c stories -c jina-ebd-v5-txts-256
./data cluster newsmon fit -c stories -c f2llm-v2-0.6b
./data cluster newsmon fit -c stories -c ml-ebd-0.6b
./data cluster newsmon fit -c stories -c ml-ebd-0.6b-256
./data cluster newsmon fit -c stories -c arctic-ebd-l-v2
./data cluster newsmon fit -c stories -c arctic-ebd-l-v2-256
```

Now, that we have adjusted the individual thresholds, we can cluster the dataset with Louvain communities algorithm:   
(see conf/task/cluster)

```shell
./data cluster newsmon -c stories -c oai-ada_002
./data cluster newsmon -c stories -c oai-txt_ebd_3s
./data cluster newsmon -c stories -c bge-m3
./data cluster newsmon -c stories -c alib-gte-mmbert
./data cluster newsmon -c stories -c qwen3-ebd-0.6b
./data cluster newsmon -c stories -c qwen3-ebd-4b
./data cluster newsmon -c stories -c qwen3-ebd-4b-256
./data cluster newsmon -c stories -c qwen3-ebd-4b-128
./data cluster newsmon -c stories -c qwen3-ebd-4b-64
./data cluster newsmon -c stories -c qwen3-ebd-8b-q8_0-256
./data cluster newsmon -c stories -c qwen3-ebd-8b-q8_0-128
./data cluster newsmon -c stories -c qwen3-ebd-8b-q8_0-64
./data cluster newsmon -c stories -c jina-ebd-v5-txts
./data cluster newsmon -c stories -c jina-ebd-v5-txts-256
./data cluster newsmon -c stories -c f2llm-v2-0.6b
./data cluster newsmon -c stories -c ml-ebd-0.6b
./data cluster newsmon -c stories -c ml-ebd-0.6b-256
./data cluster newsmon -c stories -c arctic-ebd-l-v2
./data cluster newsmon -c stories -c arctic-ebd-l-v2-256
```

## 100. TODO :D
```shell
./data sample newsmon -c sl -c bge-m3 -c hard_neg
./data sample newsmon -c sr -c bge-m3 -c hard_neg
./data sample newsmon -c mk -c bge-m3 -c hard_neg


./data sample newsmon -c hard_neg
./train seqence newsmon -c xlmr
./train seqence newsmon -c mm-bert
./train seqence eurlex -c xlmr
./train seqence eurlex -c mm-bert
./train hard_neg newsmon -c bge-m3
./train hard_neg newsmon -c m-gte
./train hard_neg newsmon -c emb-gemma3
./eval seqence newsmon -c xlmr
./eval seqence newsmon -c mm-bert
./eval token ner -c xlmr
./eval token ner -c m-bert
./eval token ner -c mm-bert
./eval token ner -c gemma3-200m
./eval token ner -c gemma3-1b
```
