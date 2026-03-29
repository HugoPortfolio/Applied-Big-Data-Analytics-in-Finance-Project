import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification

from scoring.config import MODEL_NAME, BATCH_SIZE
from utils.logger import get_logger

logger = get_logger(__name__)


class FinBERTScorer:
    def __init__(self, model_name: str = MODEL_NAME, batch_size: int = BATCH_SIZE):
        self.model_name = model_name
        self.batch_size = batch_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Using device        | %s", self.device)

        if self.device.type == "cuda":
            logger.info("CUDA device         | %s", torch.cuda.get_device_name(0))

        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.model = BertForSequenceClassification.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

        if self.device.type == "cuda":
            self.model = self.model.half()

        self.id2label = {
            int(k): v.lower() for k, v in self.model.config.id2label.items()
        }

    def score_batch(self, df_batch: pd.DataFrame) -> pd.DataFrame:
        if df_batch.empty:
            return df_batch.copy()

        texts = df_batch["chunk_text"].astype(str).tolist()

        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        encoded = {k: v.to(self.device, non_blocking=True) for k, v in encoded.items()}

        with torch.inference_mode():
            outputs = self.model(**encoded)
            probs = torch.softmax(outputs.logits.float(), dim=-1).cpu()

        label_order = [self.id2label[i] for i in range(probs.shape[1])]
        df_probs = pd.DataFrame(
            probs.numpy(),
            columns=[f"p_{label}" for label in label_order],
        )
        df_probs["pred_label"] = [label_order[i] for i in probs.argmax(dim=1).tolist()]

        return pd.concat([df_batch.reset_index(drop=True), df_probs], axis=1)
