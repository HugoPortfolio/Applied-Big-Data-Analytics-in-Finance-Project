# Earnings Call NLP and Quantitative Analysis Pipeline

## Project presentation

This project builds a complete pipeline to transform raw earnings call transcripts into quantitative variables that can be used in financial analysis.

The repository is organized as a sequence of connected stages:

1. transcript collection from Koyfin
2. raw data merging and cleanup
3. transcript cleaning and preprocessing
4. speaker parsing and section assignment
5. chunking of transcripts into model-ready text units
6. creation of an annotation sample for supervised learning
7. LLM-based labeling of chunk sentiment/tone
8. fine-tuning of FinBERT on the labeled chunk dataset
9. full-corpus chunk-level scoring with the selected model
10. transcript-level feature engineering
11. merging with market and firm-level data
12. econometric analysis

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
The raw parquet shards are first merged into a single transcript-level dataset.

This stage typically:
- loads all raw Koyfin transcript parquet files
- removes rows with unparsed dates
- concatenates the cleaned files
- removes duplicates
- sorts the final dataset chronologically

The output of this stage is the merged raw transcript parquet

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

The repository also includes a filtering step to restrict the analysis to a specific investment universe.

In the current empirical design, this step is used to keep only firms belonging to the S&P 500 universe used in the final analysis.

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
│   ├── merged_raw_files/
│   ├── curated/
│   ├── processed/
│   ├── models/
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
├── merge_raw/
│   └── merge_raw_parquets.py
│
├── preprocessing/
│   ├── config.py
│   ├── parsing.py
│   ├── enrichment.py
│   ├── data_io.py
│   ├── validation.py
│   ├── chunking.py
│   ├── pipeline.py
│   └── main.py
│
├── llm_finetune/
│   ├── config.py
│   ├── dataset_builder.py
│   ├── llm_labeler.py
│   ├── train.py
│   ├── search_hparams.py
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
│   ├── regression.py
│   ├── descriptive_stats.py
│   └── graphs.py
│
├── utils/
│   └── logger.py
│   └── sp500_only.py
│
├── logs/
└── README.md
```

---

## Detailed role of each part

Main scripts:

### `scraping/`
This folder handles transcript collection.
* `scraper.py`: browser-based Koyfin transcript scraping logic
* `selectors.py`: Selenium selectors used by the scraper
* `storage.py`: parquet shard writer
* `koyfin_helpers.py`: helper functions for navigation, waiting, and transcript extraction
* `config.py`: scraping parameters, credentials, paths, and scraping windows
* `main.py`: launch point for scraping

### `merge_raw/`
This folder handles raw file merging and first-pass cleanup.
* `merge_raw_parquets.py`: merges raw transcript shards, removes invalid dates and duplicates, and writes the final merged parquet

### `llm_finetune/`

This folder contains the full model-training workflow.

Main scripts:

* `config.py`: paths, prompt, sampling parameters, and training hyperparameters
* `dataset_builder.py`: builds the annotation sample from chunked transcripts
* `llm_labeler.py`: labels chunks using an LLM API
* `train.py`: fine-tunes FinBERT on the labeled dataset
* `search_hparams.py`: hyperparameter search on the training dataset
* `main.py`: orchestrates the full build / label / train workflow

### `scoring/`

This folder applies the selected sentiment model to the full chunk dataset.

Main scripts:

* `config.py`: input path, output path, scoring model, batch size, and filtering flags
* `finbert_scorer.py`: tokenizer / model loading and batch scoring
* `pipeline.py`: chunk preparation, batching, and parquet writing
* `main.py`: launch point for scoring

### `features/`

This folder converts chunk-level outputs into transcript-level variables and builds the final regression dataset.

Main scripts:

* `config.py`: paths to scored chunks and external data
* `text_features.py`: computes `NegScore` and aggregates transcript-level tone features
* `market_features.py`: loads and joins market and accounting variables
* `regression_prep.py`: final cleaning, transformations, and regression-dataset preparation
* `pipeline.py`: full feature-engineering pipeline
* `main.py`: launch point for feature generation

### `regressions/`

This folder contains the econometric analysis and empirical outputs.

Main scripts:

* `config.py`: regression dataset path and results folder
* `regression.py`: main specifications and robustness checks
* `descriptive_stats.py`: descriptive tables and sample overview
* `graphs.py`: descriptive and empirical figures

### `utils/`

Shared utilities used across stages.

Main scripts:

* `logger.py`: common logger configuration and handlers
* `sp500_only.py`: restricts segments and chunks to the S&P 500 universe used in the final analysis


## Main outputs by stage

### Scraping output
- raw transcript parquet shards

### Raw merge output
- merged raw transcript parquet

### Preprocessing output
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

2. **Merge raw transcript files**  
   `python merge_raw/merge_raw_parquets.py`

3. **Preprocess and chunk transcripts**  
   `python preprocessing/main.py`

4. **Optionally filter to the S&P 500 universe**  
   `python utils/sp500_only.py`

5. **Build labels and fine-tune the model**  
   `python llm_finetune/main.py`

6. **Score the full corpus with the selected model**  
   `python scoring/main.py`

7. **Build transcript-level features and the regression dataset**  
   `python features/main.py`

8. **Run the econometric analysis and empirical outputs**  
   `python regressions/regression.py`  
   `python regressions/descriptive_stats.py`  
   `python regressions/graphs.py`

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

**scraping → merging raw files → preprocessing → chunking → LLM labeling → FinBERT fine-tuning → scoring → feature engineering → regressions**

This makes it possible to study whether the tone of earnings call communication contains measurable information for financial markets.
