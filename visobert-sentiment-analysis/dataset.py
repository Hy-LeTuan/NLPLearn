import os


def save_file_for_sentiment(path):
    filename = os.path.basename(path)
    filename = filename.split(".")
    output_file = os.path.join(".", "data", "sentiment", filename[0] + ".csv")

    with open(output_file, "w", encoding="utf-8") as output:
        output.write("id, label, content\n")

        with open(path, "r", encoding="utf-8") as input:
            file_line = input.readline()
            counter = 0
            while file_line:
                file_line = file_line.split(" ")
                label = None
                content = []

                for split in file_line:
                    if "__label__" in split:
                        label = split.split("#")[-1]
                    else:
                        content.append(split)

                content = " ".join(content)
                output.write(f"{counter}, {label}, {content}")

                # update variables
                counter += 1
                file_line = input.readline()


if __name__ == "__main__":
    save_file_for_sentiment("./data/train.txt")
    save_file_for_sentiment("./data/test.txt")
