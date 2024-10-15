import regex as re


class BaseTokenizer:
    def __init__(self):
        self.merges = {}
        self.vocab = {}

    def get_stats(self, tokens):
        count = {}
        for pair in zip(tokens, tokens[1:]):
            count[pair] = count.get(pair, 0) + 1

        with open("./data/stats.txt", "w", encoding="utf-8") as f:
            for i, (key, value) in enumerate(count.items()):
                content = f"{i}. ({key[0]}, {key[1]} has count: {value})\n"
                f.write(content)

        return count

    def merge(self, tokens, byte_pair_max: tuple, new_pair_representation: int):
        self.merges[byte_pair_max] = new_pair_representation

        i = 0
        new_tokens = []

        while i < len(tokens):
            if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == byte_pair_max:
                new_tokens.append(new_pair_representation)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1

        return new_tokens

    def build_vocab(self):
        vocab = {i: bytes([i]) for i in range(256)}

        # loop through merges in order of insertion
        with open("./data/vocab.csv", "w", encoding="utf-8") as f:
            for key, value in self.merges.items():
                vocab[value] = vocab[key[0]] + vocab[key[1]]

                # char1 = b"".join((vocab[key[0]]))
                # char1 = char1.decode("utf-8", errors="replace")
                char1 = vocab[key[0]].decode("utf-8", errors="replace")
                print(char1)

        self.vocab = vocab

    def visualize_merges(self):
        merges = sorted(((value, key) for key, value in self.merges.items()))

        with open("./data/merges.csv", "w", encoding="utf-8") as f:
            f.write(f"Byte 1, Byte 2, Merged byte value\n")

            for key_value_pair in merges:
                f.write(f"""{key_value_pair[1][0]}, {
                        key_value_pair[1][1]}, -> {key_value_pair[0]}\n""")

    def train(self, text: str, vocab_size: int, verbose: bool = False):
        train_iterations = vocab_size - 256
        tokens = list(text.encode("utf-8", errors="replace"))

        for n in range(train_iterations):

            stats = self.get_stats(tokens=tokens)
            byte_pair_max = max(stats, key=stats.get)

            old_length = len(tokens)
            tokens = self.merge(tokens=tokens, byte_pair_max=byte_pair_max,
                                new_pair_representation=256 + n)

            if verbose:
                print(f"byte pair max: {byte_pair_max}")
                print(f"byte pair max frequency: {stats.get(byte_pair_max)}")

                print(f"Before merging: {old_length}")
                print(f"After merging: {len(tokens)}")

        return tokens


if __name__ == "__main__":
    base_tokenizer = BaseTokenizer()

    file_content = ""
    with open("./data/taylorswift.txt", "r") as f:
        content = f.readline()
        while content:
            file_content += content
            content = f.readline()

    tokens = base_tokenizer.train(
        text=file_content, vocab_size=100+256, verbose=False)

    base_tokenizer.build_vocab()
    base_tokenizer.visualize_merges()
