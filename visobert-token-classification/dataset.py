import os


def build_entity_tokens():
    entity_token_names = ["PATIENT_ID", "NAME",
                          "AGE", "GENDER", "JOB", "LOCATION", "ORGANIZATION", "SYMPTOM_AND_DISEASE", "TRANSPORTATION", "DATE"]

    position_prefix = ["B-", "I-"]

    counter = 0
    entity_tokens = {}
    for token_type in entity_token_names:
        for prefix in position_prefix:
            final_token = prefix + token_type
            entity_tokens[final_token] = counter
            counter += 1

    entity_tokens["O"] = counter

    return entity_tokens


def read_ner_file(path):
    entities = {"words": [], "tokens": []}
    with open(path, "r", encoding="utf-8") as f:
        current_words = []
        current_tokens = []

        line_content = f.readline()

        while line_content:
            if line_content == "\n" and len(current_words) > 0 and len(current_tokens) > 0:
                entities["words"].append(current_words)
                entities["tokens"].append(current_tokens)

                current_tokens = []
                current_words = []
            else:
                line_content = line_content.strip()
                word, token = line_content.split(" ")

                current_words.append(word)
                current_tokens.append(token)

            line_content = f.readline()

    return entities


if __name__ == "__main__":
    # entites = read_ner_file("./data/syllable/train_syllable.conll")
    # print(entites)
    print(build_entity_tokens())
