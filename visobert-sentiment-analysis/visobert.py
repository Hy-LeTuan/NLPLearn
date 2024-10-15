from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained(
    "5CD-AI/Vietnamese-Sentiment-visobert")
model = AutoModelForSequenceClassification.from_pretrained(
    "5CD-AI/Vietnamese-Sentiment-visobert")
