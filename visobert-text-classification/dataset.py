import os


def save_file_for_sentiment(path):
    filename = os.path.basename(path)
    filename = filename.split(".")
    output_file = os.path.join(".", "data", "sentiment", filename[0] + ".txt")

    with open(output_file, "w", encoding="utf-8") as output:
        output.write("id,label,value\n")

        with open(path, "r", encoding="utf-8") as input:
            file_line = input.readline()
            counter = 0
            while file_line:
                file_line = file_line.split(" ")
                label = None
                value = []

                for split in file_line:
                    if "__label__" in split:
                        label = split.split("#")[-1]
                    else:
                        value.append(split)

                value = " ".join(value).strip()
                output.write(f'{counter},{label},{value}\n')

                # update variables
                counter += 1
                file_line = input.readline()


def read_sentiment_data(path):
    content = {}

    with open(path, "r", encoding="utf-8") as f:
        headers = f.readline()

        for head in headers.split(","):
            head = head.strip()
            content[head] = []

        line = f.readline()

        while line:
            line_content = line.split(",")

            content["id"].append(int(line_content[0]))
            label = line_content[1]
            if label == "positive":
                label = 2
            elif label == "negative":
                label = 1
            else:
                label = 0
            content["label"].append(int(label))
            content["value"].append(str(line_content[2]))

            line = f.readline()

    return content


def save_file_for_intention(path):
    filename = os.path.basename(path)
    filename = filename.split(".")
    output_file = os.path.join(".", "data", "intention", filename[0] + ".txt")

    unique_label = {'TRADEMARK': 0, 'INTEREST_RATE': 1, 'ACCOUNT': 2, 'SECURITY': 3, 'CARD': 4, 'SAVING': 5, 'CUSTOMER_SUPPORT': 6,
                    'PROMOTION': 7, 'MONEY_TRANSFER': 8, 'PAYMENT': 9, 'DISCOUNT': 10, 'LOAN': 11, 'OTHER': 12, 'INTERNET_BANKING': 13}

    with open(output_file, "w", encoding="utf-8") as output:
        output.write("id,label,value\n")

        with open(path, "r", encoding="utf-8") as input:
            file_line = input.readline()
            counter = 0

            # process 1 record at a time
            while file_line:
                file_line = file_line.split(" ")
                labels = [0] * len(unique_label)
                value = []

                for split in file_line:
                    if "__label__" in split:
                        label_value = split.split("#")[0]
                        label_value = label_value.split("__label__")[-1]

                        labels[unique_label[label_value]] = 1
                    else:
                        value.append(split)

                value = " ".join(value).strip()
                output.write(
                    f'{counter},{" ".join(str(x) for x in labels)},{value}\n')

                # update variables
                counter += 1
                file_line = input.readline()


def read_intention_data(path):
    content = {}

    with open(path, "r", encoding="utf-8") as f:
        headers = f.readline()

        for head in headers.split(","):
            head = head.strip()
            content[head] = []

        line = f.readline()

        while line:
            line_content = line.split(",")

            content["id"].append(int(line_content[0]))

            label = line_content[1]
            label = label.split(" ")
            label = [int(x) for x in label]

            content["label"].append(label)

            content["value"].append(str(line_content[2]))

            line = f.readline()

    return content


if __name__ == "__main__":
    save_file_for_intention("./data/train.txt")
    save_file_for_intention("./data/test.txt")
