# Earnings Call NLP and Quantitative Analysis Pipeline

## Project presentation

This project builds a complete pipeline to transform raw earnings call transcripts into quantitative variables that can be used in financial analysis.

The repository is organized as a sequence of connected stages:

1. transcript collection from Koyfin
2. transcript cleaning and preprocessing
3. speaker parsing and section assignment
4. chunking of transcripts into model-ready text units
5. creation of an annotation sample for supervised learning
6. LLM-based labeling of chunk sentiment/tone
7. fine-tuning of FinBERT on the labeled chunk dataset
8. full-corpus chunk-level scoring with the selected model
9. transcript-level feature engineering
10. merging with market and firm-level data
11. econometric analysis

The goal is not only to classify text, but to convert earnings call language into structured sentiment measures that can be tested against market outcomes.

---

## End-to-end pipeline

### 1. Scraping
The project starts by collecting earnings call transcripts from Koyfin.

This stage is responsible for:
- logging in and navigating the Koyfin interface
- iterating over time windows
- scraping transcript titles, dates, subheaders, bodies, and transcript metadata
- saving the results in parquet shards

The scraping output is the raw transcript dataset.

### 2. Raw data merging and cleanup
The raw parquet shards are merged into a clean transcript-level dataset.

This stage typically:
- merges all source transcript parquet files
- removes duplicate entries
- parses transcript dates
- drops malformed or unusable rows

This produces a single cleaned transcript dataset.

### 3. Transcript parsing and enrichment
The cleaned transcripts are transformed into a more structured format.

This stage usually:
- extracts company name and transcript metadata
- parses speaker names and speaker roles
- splits transcripts into speaker turns or segments
- enriches transcript/company metadata
- validates section balance and transcript quality

At this point, the data is organized around transcript segments rather than one large transcript body.

### 4. Chunking
Each transcript segment is split into smaller chunks so that the model can process manageable text units.

Chunk-level rows typically contain:
- `chunk_id`
- `transcript_id`
- `segment_id`
- `company_name`
- `ticker`
- `date`
- `section`
- `speaker_role`
- `chunk_order`
- `chunk_text`
- `chunk_token_count`

These chunks are the base unit used later for labeling, fine-tuning, and inference.

### 5. Filtering to the target universe
The repository also includes filters to restrict the analysis to a target universe such as the S&P 500.

This step is useful when the downstream financial analysis is meant to be run on a specific investment universe.

### 6. Annotation sample creation
Instead of labeling the full corpus directly, the project first creates an annotation sample.

This sample is built from chunked transcripts using filtering and sampling rules such as:
- dropping empty chunks
- dropping very short chunks
- optionally dropping operator segments
- stratifying the sample across sections like `Prepared`, `Q`, and `A`
- optionally using prior teacher scores to enrich the sample

The output is a manageable subset of chunks selected for labeling.

### 7. LLM labeling
The annotation sample is labeled with an LLM.

Each chunk receives one of three labels:
- `negative`
- `neutral`
- `positive`

The labeling prompt is designed to classify the overall tone of the speaker, not isolated keywords. The prompt pays special attention to the difference between:
- truly negative tone
- neutral factual explanations that contain negative vocabulary

The output is a labeled chunk dataset used for supervised training.

### 8. Fine-tuning
The project fine-tunes `yiyanghkust/finbert-tone` on the labeled chunk dataset.

This stage includes:
- transcript-level train / validation / test split
- tokenization of chunk text
- multi-class sequence classification
- evaluation on held-out data
- saving the final model locally

The final fine-tuned model is stored in:
- `models/finbert_finetuned/`

This model is then used for large-scale chunk scoring.

### 9. Chunk-level scoring
Once the model is trained, the full chunk dataset is scored.

For each chunk, the scoring stage produces:
- `p_negative`
- `p_neutral`
- `p_positive`
- `pred_label`

These outputs are saved as scored chunk parquets and become the input to the feature engineering stage.

### 10. Transcript-level feature engineering
Chunk-level probabilities are aggregated into transcript-level text features.

The main continuous chunk-level signal is:
- `neg_score = p_negative - p_positive`

Chunk scores are then aggregated by transcript section to construct variables such as:
- `NegPrepared`
- `NegQ`
- `NegA`
- `NegQA`
- `NegGap`

These variables summarize the sentiment profile of each earnings call.

### 11. Merge with market and firm-level data
The transcript-level text features are merged with structured financial data such as:
- earnings data
- market returns
- market capitalization
- volume data

This produces a regression-ready dataset.

### 12. Econometric analysis
The final dataset is used for regressions linking earnings-call sentiment to market outcomes.

Typical dependent variables include:
- cumulative abnormal returns
- absolute returns
- short-run volatility

The objective is to test whether the sentiment extracted from management communication helps explain financial market reactions.

---

## Current logical repository structure

Below is the project tree in logical package form, with the current scripts grouped by stage.

```text
.
├── data/
│   ├── raw/
│   ├── curated/
│   ├── processed/
│   ├── training/
│   ├── scored/
│   ├── features/
│   ├── results/
│   └── external/
│       ├── market/
│       ├── marketCap/
│       ├── earning/
│       └── sp500_constituents.csv
│
├── scraping/
│   ├── config.py
│   ├── scraper.py
│   ├── selectors.py
│   ├── storage.py
│   ├── koyfin_helpers.py
│   └── main.py
│
├── preprocessing/
│   ├── config.py
│   ├── merge_raw_parquets.py
│   ├── parsing.py
│   ├── enrichment.py
│   ├── data_io.py
│   ├── validation.py
│   ├── chunking.py
│   ├── pipeline.py
│   └── main.py
│
├── filters/
│   └── sp500_only.py
│
├── llm_finetune/
│   ├── config.py
│   ├── dataset_builder.py
│   ├── llm_labeler.py
│   ├── train.py
│   ├── search_hparams.py
│   ├── read_results.py
│   ├── export_prompt_examples.py
│   └── main.py
│
├── scoring/
│   ├── config.py
│   ├── finbert_scorer.py
│   ├── pipeline.py
│   └── main.py
│
├── features/
│   ├── config.py
│   ├── text_features.py
│   ├── market_features.py
│   ├── regression_prep.py
│   ├── pipeline.py
│   └── main.py
│
├── regressions/
│   ├── config.py
│   ├── pipeline.py
│   └── main.py
│
├── utils/
│   └── logger.py
│
├── models/
│   └── finbert_finetuned/
│       ├── config.json
│       ├── tokenizer files
│       ├── model.safetensors
│       └── checkpoint-*/
│
├── logs/
└── README.md
```

---

## Detailed role of each part

### `scraping/`
This folder handles transcript collection.

Main scripts:
- `scraper.py`: browser-based Koyfin transcript scraping logic
- `selectors.py`: Selenium selectors used by the scraper
- `storage.py`: parquet shard writer
- `koyfin_helpers.py`: helper functions for navigation, waiting, and transcript extraction
- `config.py`: scraping parameters, credentials, paths, and scraping windows
- `main.py`: launch point for scraping

### `preprocessing/`
This folder transforms raw transcripts into clean structured chunk datasets.

Main scripts:
- `merge_raw_parquets.py`: merges raw transcript shards and cleans invalid dates
- `parsing.py`: extracts speaker and transcript metadata
- `enrichment.py`: enriches company metadata and normalizes firm names
- `data_io.py`: parquet loading/saving helpers
- `validation.py`: transcript quality and section-balance checks
- `chunking.py`: chunk generation using a FinBERT tokenizer
- `pipeline.py`: end-to-end preprocessing flow
- `config.py`: preprocessing paths and parameters
- `main.py`: launch point for preprocessing

### `filters/`
This folder contains filtering scripts for universe selection.

Main script:
- `sp500_only.py`: restricts segments and chunks to S&P 500 companies

### `llm_finetune/`
This folder contains the full model-training workflow.

Main scripts:
- `config.py`: paths, prompt, sampling parameters, and training hyperparameters
- `dataset_builder.py`: builds the annotation sample from chunked transcripts
- `llm_labeler.py`: labels chunks using an LLM API
- `train.py`: fine-tunes FinBERT on the labeled dataset
- `search_hparams.py`: hyperparameter search on the training dataset only
- `read_results.py`: reads metrics and evaluation predictions
- `export_prompt_examples.py`: exports informative misclassification examples for prompt analysis
- `main.py`: orchestrates the full build / label / train workflow

### `scoring/`
This folder applies the selected sentiment model to the full chunk dataset.

Main scripts:
- `config.py`: input path, output path, scoring model, batch size, filtering flags
- `finbert_scorer.py`: tokenizer/model loading and batch scoring
- `pipeline.py`: chunk preparation, batching, and parquet writing
- `main.py`: launch point for scoring

### `features/`
This folder converts chunk-level outputs into transcript-level variables and builds the final regression dataset.

Main scripts:
- `config.py`: paths to scored chunks and external data
- `text_features.py`: computes `neg_score` and aggregates transcript features
- `market_features.py`: loads and joins market and earnings features
- `regression_prep.py`: winsorization, logs, filtering, and final prep
- `pipeline.py`: full feature-engineering pipeline
- `main.py`: launch point for feature generation

### `regressions/`
This folder contains the econometric analysis stage.

Main scripts:
- `config.py`: regression dataset path and results folder
- `pipeline.py`: model formulas, controls, fixed effects, clustering, and result tables
- `main.py`: launch point for regressions

### `utils/`
Shared utilities used across stages.

Main script:
- `logger.py`: common logger configuration and handlers

---

## Main outputs by stage

### Scraping output
- raw transcript parquet shards

### Preprocessing output
- merged transcript parquet
- speaker-segment parquet
- chunk parquet
- optional S&P 500 filtered segment/chunk parquets

### LLM fine-tuning output
- annotation sample parquet
- labeled annotation parquet
- train-ready dataset parquet
- training metrics JSON
- evaluation predictions parquet
- saved fine-tuned model directory

### Scoring output
- scored chunk parquet with class probabilities and predicted label

### Features output
- transcript-level sentiment feature parquet
- final regression dataset parquet

### Regression output
- tables and regression results saved under results directories

---

## Typical execution order

1. **Scrape transcripts**  
   `python scraping/main.py`

2. **Preprocess and chunk transcripts**  
   `python preprocessing/main.py`

3. **Optionally filter to the S&P 500 universe**  
   `python filters/sp500_only.py`

4. **Build labels and fine-tune the model**  
   `python llm_finetune/main.py`

5. **Score the full corpus with the selected model**  
   `python scoring/main.py`

6. **Build transcript-level features and the regression dataset**  
   `python features/main.py`

7. **Run the econometric analysis**  
   `python regressions/main.py`

---

## Modeling philosophy

This project treats earnings calls as structured financial communication.

The approach is:
- define a clear text signal at the chunk level
- train a model to recognize that signal
- score the corpus consistently
- aggregate the signal into interpretable transcript-level measures
- test those measures against market outcomes

The pipeline is therefore both an NLP workflow and a quantitative finance workflow.

---

## Summary

The repository transforms raw earnings call transcripts into a complete empirical research pipeline:

**scraping → preprocessing → chunking → LLM labeling → FinBERT fine-tuning → scoring → feature engineering → regressions**

This makes it possible to study whether the tone of earnings call communication contains measurable information for financial markets.
