# CaNNopy
Simple ML research framework that focuses on repeatability and supports flexible diversity of tasks. 

## What it does
- Unified CLI (`./data`, `./train`, `./eval`, `./test`) routes through `src/app/entrypoint.py` to load layered YAML configs, build Hugging Face `TrainingArguments` plus custom `DataArguments`/`ModelArguments`, and dispatch to task modules.
- Supports dataset creation/downloading (ESG, multilingual keyword match, Slavic NER, EURLEX) and sequence/token model training and evaluation; writes outputs under `result/` with logs in `log/`.
- Uses PyTorch/Transformers with config-driven runs for reproducibility.

## Project structure
- `src/app/`: CLI machinery (arg parsing, config stacking, logging, discovery utilities).
- `src/data/`, `src/train/`, `src/eval/`, `src/test/`: command-specific task modules discovered by the entrypoint.
- `conf/`: layered YAML configs; data source docs and defaults live here.
- `data/`, `result/`, `log/`, `tmp/`: runtime assets and artifacts (keep large files out of git).
- `train/`, `eval/`: notebooks/scripts for experiments; adjust configs via `-c` to stack overrides.
- `requirements.txt`, `pyproject.toml`: dependencies and script entrypoints.

```shell
python -m venv .venv && source .venv/bin/activate
```

```shell
pip install -U pip setuptools
```

```shell
pip install -r requirements.txt
```

## Dataset mining / creation

Create dataset from ESG data source:
```shell
./data create esg
```

Create a dataset from a multilingual keywords matching data source:
```shell
./data create ml-kw-match
```

## Dataset downloading

Download Slavic NER dataset:
```shell
./data download ner
```

Download EURLEX57K dataset:
```shell
./data download eurlex
```

TODO :D
```shell
./data download ner
./data download newsmon
./data download eurlex
./data prepare ner
./data prepare newsmon
./data prepare eurlex
./data embed newsmon -c bge-m3.yaml -c sl.yaml
./data embed newsmon -c m-gte.yaml
./data embed newsmon -c emb-genna3.yaml -c sl.yaml
./data resample newsmon -c sl.yaml
./data resample newsmon -c sr.yaml
./data resample newsmon
./data sample newsmon -c hard_neg.yaml
./data split ner
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
./train token ner -c xlmr.yaml
./train token ner -c m-bert.yaml
./train token ner -c mm-bert.yaml
./train token ner -c genna3-200m.yaml
./train token ner -c genna3-1b.yaml
./train hard_neg newsmon -c bge-m3.yaml
./train hard_neg newsmon -c m-gte.yaml
./train hard_neg newsmon -c emb-genna3.yaml
./eval seqence newsmon -c xlmr.yaml
./eval seqence newsmon -c mm-bert.yaml
./eval token ner -c xlmr.yaml
./eval token ner -c m-bert.yaml
./eval token ner -c mm-bert.yaml
./eval token ner -c genna3-200m.yaml
./eval token ner -c genna3-1b.yaml
```
