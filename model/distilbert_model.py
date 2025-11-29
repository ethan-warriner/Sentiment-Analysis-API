from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import torch.nn as nn

class MyDistilBERT(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased-finetuned-sst-2-english"
        )
    
    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits
    