import regex as re


class BaseTokenizer:
    def __init__(self):
        self.merges = {}
        self.vocab = {}

    def decode(self, ids):
        tokens = b"".join([self.vocab[id] for id in ids])
        string = tokens.decode("utf-8", errors="replace")

        return string

    def encode(self, text):
        tokens = text.encode("utf-8", errors="replace")

        while len(tokens) >= 2:
            success = False

            for key, value in self.merges.items():
                old_length = len(tokens)

                tokens = self.merge(
                    tokens, key, value, save=False)

                if len(tokens) < old_length:
                    success = True

            if not success:
                break

        return tokens

    def get_stats(self, tokens):
        stats = {}
        for pair in zip(tokens, tokens[1:]):
            stats[pair] = stats.get(pair, 0) + 1

        self.save_stats(stats)

        return stats

    def save_stats(self, stats, path="./data/base/stats.csv"):

        with open(path, "w", encoding="utf-8") as f:
            f.write("Byte pair, Frequency\n")
            for i, (key, value) in enumerate(stats.items()):
                content = f"({key[0]}, {key[1]}), -> [{value}]\n"
                f.write(content)

    def merge(self, tokens, byte_pair_max: tuple, new_pair_representation: int, save=True):
        if save:
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

    def build_vocab(self, path="./data/base/vocab.csv"):
        vocab = {i: bytes([i]) for i in range(256)}

        # loop through merges in order of insertion
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"Byte 1, Byte 2, Combined byte\n")
            for key, value in self.merges.items():
                vocab[value] = vocab[key[0]] + vocab[key[1]]

                char1 = vocab[key[0]].decode("utf-8", errors="replace")
                char2 = vocab[key[1]].decode("utf-8", errors="replace")

                char_transform = vocab[value].decode("utf-8", errors="replace")

                f.write(f"[{char1}], [{char2}], -> [{char_transform}]\n")

        self.vocab = vocab

    def visualize_merges(self, path="./data/base/merges.csv"):
        merges = sorted(((value, key) for key, value in self.merges.items()))

        with open(path, "w", encoding="utf-8") as f:
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

        self.build_vocab()
        self.visualize_merges()

        return tokens


class RegixTokenizer(BaseTokenizer):
    def __init__(self):
        super().__init__()
        self.gpt4_split_pattern = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

    def get_stats(self, tokens, stats=dict):
        if len(tokens) >= 2:
            for pair in zip(tokens, tokens[1:]):
                stats[pair] = stats.get(pair, 0) + 1

    def decode(self, ids_chunk: list) -> str:
        tokens = b""
        for ids in ids_chunk:
            current_token = b"".join([self.vocab[id] for id in ids])
            tokens += current_token

        string = tokens.decode(encoding="utf-8", errors="replace")
        return string

    def encode(self, text):
        text_chunks = re.findall(self.gpt4_split_pattern, text)
        tokens = [list(ch.encode("utf-8")) for ch in text_chunks]

        for i in range(len(tokens)):
            ids = tokens[i]

            # deal with current ids
            while len(ids) >= 2:
                success = False

                for key, value in self.merges.items():
                    old_length = len(ids)
                    ids = self.merge(tokens=ids, byte_pair_max=key,
                                     new_pair_representation=value, save=False)

                    if len(ids) < old_length:
                        success = True

                if not success:
                    break

            tokens[i] = ids

        return tokens

    def train(self, text: str, vocab_size: int, verbose: bool = False):
        text_chunks = re.findall(self.gpt4_split_pattern, text)
        tokens = [list(ch.encode("utf-8")) for ch in text_chunks]
        train_iterations = vocab_size - 256

        for n in range(train_iterations):
            stats = {}
            for chunk in tokens:
                self.get_stats(tokens=chunk, stats=stats)
            self.save_stats(stats=stats, path="./data/regex/stats.csv")

            byte_pair_max = max(stats, key=stats.get)

            old_length = sum([len(chunk) for chunk in tokens])
            tokens = [self.merge(tokens=chunk, byte_pair_max=byte_pair_max,
                                 new_pair_representation=256 + n) for chunk in tokens]
            new_length = sum([len(chunk) for chunk in tokens])

            if verbose:
                print(f"byte pair max: {byte_pair_max}")
                print(f"byte pair max frequency: {stats.get(byte_pair_max)}")

                print(f"Before merging: {old_length}")
                print(f"After merging: {new_length}")

            self.build_vocab(path="./data/regex/vocab.csv")
            self.visualize_merges(path="./data/regex/merges.csv")

        return tokens


if __name__ == "__main__":
    regex_tokenizer = RegixTokenizer()

    file_content = ""
    with open("./data/taylorswift.txt", "r") as f:
        content = f.readline()
        while content:
            file_content += content
            content = f.readline()

    regex_tokens = regex_tokenizer.train(
        text=file_content, vocab_size=100+256, verbose=False)
