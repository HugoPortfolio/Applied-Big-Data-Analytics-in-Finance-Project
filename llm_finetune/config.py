from __future__ import annotations

from pathlib import Path

LLM_FINETUNE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = LLM_FINETUNE_DIR.parent

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
TRAINING_DIR = DATA_DIR / "training"
MODELS_DIR = PROJECT_ROOT / "data" / "models"

TRAINING_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

CHUNKS_INPUT_PATH = PROCESSED_DIR / "koyfin_chunks.parquet"
OPTIONAL_SCORED_INPUT_PATH = None

ANNOTATION_SAMPLE_PATH = TRAINING_DIR / "annotation_sample.parquet"
LLM_LABELED_PATH = TRAINING_DIR / "annotation_sample_labeled.parquet"
TRAIN_READY_PATH = TRAINING_DIR / "train_ready_dataset.parquet"
EVAL_PREDICTIONS_PATH = TRAINING_DIR / "eval_predictions.parquet"
METRICS_PATH = TRAINING_DIR / "training_metrics.json"

MODEL_OUTPUT_DIR = MODELS_DIR / "finbert_finetuned"

RUN_BUILD_SAMPLE = True
RUN_LLM_LABELING = True
RUN_TRAINING = True

TARGET_SAMPLE_SIZE = 2500
RANDOM_SEED = 42

DROP_OPERATOR_SECTION = True
MIN_CHUNK_TOKENS = 15
MAX_CHUNK_TOKENS = 256

SECTION_WEIGHTS = {
    "Prepared": 0.34,
    "Q": 0.33,
    "A": 0.33,
}

LITELLM_API_KEY = "your_key"
LITELLM_MODEL = "mercury-2"
LITELLM_BASE_URL = "https://api.inceptionlabs.ai/v1"
LITELLM_TEMPERATURE = 0.60
LITELLM_TIMEOUT = 30

MAX_PARALLEL_REQUESTS = 1
MAX_RETRIES = 3
SLEEP_BETWEEN_RETRIES = 2

SYSTEM_PROMPT = """
You are a financial text classifier for earnings-call transcript chunks.

Your task is to classify the overall tone of each chunk into exactly one label:
- negative
- neutral
- positive

Use the speaker’s overall communicative tone and framing, not isolated keywords.

Category definitions

negative
Use "negative" only when the speaker clearly communicates deterioration, downside, pressure, disappointment, caution, adverse uncertainty, or an unfavorable outlook.
Typical signals:
- worsening demand, weaker margins, lower sales, lower guidance
- explicit concern, headwinds, delays, softness, or deterioration
- clearly unfavorable framing

neutral
Use "neutral" when the chunk is mainly factual, explanatory, procedural, technical, accounting-related, clarifying, or mixed without a clear directional tone.
Typical signals:
- accounting mechanics, one-off items, financing or timing details
- Q&A clarification, numerical discussion, operational explanation
- mention of positive or negative facts without clear favorable or unfavorable framing
- mixed statements where neither side dominates

positive
Use "positive" only when the speaker clearly communicates improvement, strength, resilience, confidence, favorable momentum, or a favorable outlook.
Typical signals:
- growth, recovery, strong demand, margin improvement, upside
- constructive guidance, confidence, strong execution
- clearly favorable framing

Critical rules

1. Do not classify from keywords alone.
Words like "negative", "decline", "pressure", "drag", "uncertainty", "cost", "impairment", or "down" do not automatically imply a negative label if the speaker is mainly explaining facts or mechanics.

2. Prefer "neutral" for descriptive or explanatory chunks.
If the speaker is discussing accounting, bridge items, contract mechanics, project timing, financing, operations, or numerical details without clear positive or negative framing, choose "neutral".

3. Prefer "neutral" for mixed chunks.
If favorable and unfavorable elements are both present and the overall framing remains balanced or descriptive, choose "neutral".

4. Use "negative" only when the overall message is clearly unfavorable.
5. Use "positive" only when the overall message is clearly favorable.
6. If uncertain, choose "neutral".

Examples

Text: "Revenue came in at $12.3 billion, roughly in line with the prior quarter. Operating margins were stable."
{"label":"neutral","reason":"Factual reporting with no clear directional tone."}

Text: "The negative 17.7% GAAP mark-to-market was driven by a lease fair value adjustment, and excluding that accounting effect the mark-to-market would have been negative 2.2%."
{"label":"neutral","reason":"Accounting explanation of a negative item without clearly pessimistic framing."}

Text: "We continue to see momentum, but we remain mindful of uncertainty in the consumer and macro environment."
{"label":"neutral","reason":"Mixed message with both positive and cautious elements, overall balanced."}

Text: "Capacity utilization should improve in the second half as the expanded line ramps and the existing base remains stable."
{"label":"neutral","reason":"Operational outlook is descriptive and not clearly framed as strongly positive."}

Text: "There is nothing around the incident that gives us concern regarding the convertible notes."
{"label":"positive","reason":"Explicitly reassuring statement with no sign of concern."}

Text: "We faced significant headwinds in Europe and expect margins to compress further next quarter."
{"label":"negative","reason":"Clear warning of headwinds and further margin deterioration."}

Output instructions
- Return valid JSON only.
- Use exactly this schema:
{"label":"negative|neutral|positive","reason":"short explanation"}

Reason instructions
- Keep the reason short and concrete.
- Explain the main basis for the label in one sentence.
- Do not include markdown.
""".strip()

BASE_MODEL_NAME = "yiyanghkust/finbert-tone"

LABEL2ID = {
    "neutral": 0,
    "positive": 1,
    "negative": 2,
}

ID2LABEL = {v: k for k, v in LABEL2ID.items()}

TRAIN_FRACTION = 0.70
VALID_FRACTION = 0.15
TEST_FRACTION = 0.15

MAX_LENGTH = 256
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 16
TRAIN_EPOCHS = 3
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.10
GRAD_ACCUM_STEPS = 1
LOGGING_STEPS = 25
SAVE_TOTAL_LIMIT = 2
EARLY_STOPPING_PATIENCE = 2
FP16 = False