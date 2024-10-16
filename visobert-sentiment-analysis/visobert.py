from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained(
    "5CD-AI/Vietnamese-Sentiment-visobert")
model = AutoModelForSequenceClassification.from_pretrained(
    "5CD-AI/Vietnamese-Sentiment-visobert")


def visualize_vocab(tokenizer: AutoTokenizer):
    vocab = tokenizer.vocab
    vocab = sorted((value, key) for key, value in vocab.items())

    with open("./data/sentiment/visualizations/merges.txt", "w", encoding="utf-8") as f:
        f.write("Order, New representation\n")
        for pair in vocab:
            f.write(f"{pair[0]}, -> {pair[1]}\n")


if __name__ == "__main__":
    visualize_vocab(tokenizer=tokenizer)
