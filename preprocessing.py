import json
import os
import pandas as pd

from itertools import combinations, product

ALPHABET = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n",
            "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]

def prepare_length_lookup(source="./en_dict/en_370k.txt", output_directory="./en_dict/length_lookup"):
    with open(source, 'r') as f:
        en_dict = f.read().split("\n")

    max_len = max(len(x) for x in en_dict)
    decreasing_int = list(range(max_len, 0, -1))
    output = {word_len: [] for word_len in decreasing_int}

    counter = 0
    for word in en_dict:
        counter += 1
        if counter % 10000 == 0:
            print(f"Processing word No.{counter}")
        word_len = len(word)
        output[word_len].append(word)

    # Write to txt files
    for k, v in output.items():
        if v != []:
            if not os.path.isdir(output_directory):
                os.makedirs(output_directory, exist_ok=True)
            path = os.path.join(output_directory, f"len_{k}.txt")
            with open(path, 'w') as f:
                f.write("\n".join(v))

def prepare_letter_lookup(source="./en_dict/en_370k.txt", output_directory="./en_dict/letter_lookup"):
    with open(source, 'r') as f:
        en_dict = f.read().split("\n")
    
    max_len = max(len(x) for x in en_dict)
    decreasing_int = list(range(max_len, -1, -1))
    len_pos_comb = list(combinations(decreasing_int, 2))
    letter_len_pos_comb = list(product(ALPHABET, len_pos_comb))
    output = {comb: [] for comb in letter_len_pos_comb}

    counter = 0
    for word in en_dict:
        counter += 1
        if counter % 10000 == 0:
            print(f"Processing word No.{counter}")
        letters = list(word)
        word_len = len(word)
        if word_len > max_len:
            continue
        for pos, letter in enumerate(letters):
            comb = (letter, (word_len, pos))
            output[comb].append(word)

    # Write to txt files
    for k, v in output.items():
        if v != []:
            output_dir = os.path.join(output_directory, k[0], f"len_{k[1][0]}")
            if not os.path.isdir(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            path = os.path.join(output_dir, f"pos_{k[1][1]}.txt")
            with open(path, 'w') as f:
                f.write("\n".join(v))

def prepare_word_freq(source="./en_dict/unigram_freq.csv", output_path="./en_dict/en_word_freq_333k.json"):
    df = pd.read_csv(source)
    df["count"] /= 1e9 # Scale it down by a billion. This factor might need to be adjusted based on the situation
    word_freq = {df.loc[idx, "word"]: df.loc[idx, "count"] for idx in range(len(df))}
    with open(output_path, "w") as f:
        json.dump(word_freq, f)

def prepare_noun_noun_freq(source="./en_dict/noun-noun.csv", output_path="./en_dict/noun-noun-freq.json"):
    df = pd.read_csv(source)
    df["frequency"] /= 1e2 # Scale it down by a billion. This factor might need to be adjusted based on the situation
    freq = {" ".join([df.loc[idx, "noun1"], df.loc[idx, "noun2"]]): df.loc[idx, "frequency"] for idx in range(len(df))}
    with open(output_path, "w") as f:
        json.dump(freq, f)

# prepare_length_lookup()
# prepare_letter_lookup()
# prepare_word_freq()
# prepare_noun_noun_freq()