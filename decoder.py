import pandas as pd
from itertools import product, permutations, combinations
import os
from typing import Dict, List, Union, Iterable
import json
import numpy as np
import time

ALPHABET = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n",
            "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]

class Decoder:
    
    def __init__(self):
        self.df = pd.read_csv("./mapping.csv").astype(str)
        self.periodic = pd.read_csv("./periodic.csv").astype(str)
        self.country = pd.read_csv("./country.csv").astype(str)
        self.pokemon = pd.read_csv("./pokemon.csv").astype(str)

        with open("./en_dict/word_freq.json", "r") as f:
            data = json.load(f)
        self.word_freq = data
        self.lowest_freq = min(self.word_freq.values())
    
    def validate(self, cipher_list, cipher_type):
        cipher_list = [cipher.upper() for cipher in cipher_list]
        if cipher_type == "periodic":
            return set(cipher_list).issubset(self.periodic["number"])
        elif cipher_type == "bacon":
            bacon1_ab = set(cipher_list).issubset(self.df["bacon1_ab"])
            bacon1_ab_reverse = set(cipher_list).issubset(self.df["bacon1_ab_reverse"])
            bacon2_ab = set(cipher_list).issubset(self.df["bacon2_ab"])
            bacon2_ab_reverse = set(cipher_list).issubset(self.df["bacon2_ab_reverse"])
            return any([bacon1_ab, bacon1_ab_reverse, bacon2_ab, bacon2_ab_reverse])
        else:
            return set(cipher_list).issubset(self.df[cipher_type])
    
    def analyze(self, cipher_list):
        cipher_list = [cipher.upper() for cipher in cipher_list]
        cipher_type_list = list(self.df.columns[1:])
        cipher_type_list = [cipher_type for cipher_type in cipher_type_list if "bacon" not in cipher_type]
        cipher_type_list.append("periodic")
        cipher_type_list.append("bacon")
        return [cipher_type for cipher_type in cipher_type_list if self.validate(cipher_list, cipher_type)]
    
    def decode_periodic(self, cipher_list):
        cipher_list = [cipher.upper() for cipher in cipher_list]
        cipher_list = [cipher.upper() for cipher in cipher_list]
        mapper = {cipher: answer for cipher, answer in zip(self.periodic["number"],
                                                           self.periodic["element"])}
        return "".join([mapper[cipher] for cipher in cipher_list]).lower()
    
    def decode_polybius(self, cipher_list):
        cipher_list = [cipher.upper() for cipher in cipher_list]
        mapper = {cipher: answer for cipher, answer in zip(self.df["polybius"], self.df["letter"])}
        answer = "".join([mapper[cipher] for cipher in cipher_list])
        if "24" in cipher_list:
            answer2 = answer.replace("j", "i")
            answer = f"{answer} or {answer2}"
        return answer
    
    def decode_bacon(self, cipher_list):
        cipher_list = [cipher.upper() for cipher in cipher_list]
        answer_list = []

        mapper = {cipher: answer for cipher, answer in zip(self.df["bacon1_ab"], self.df["letter"])}
        if set(cipher_list).issubset(set(mapper.keys())):
            answer_list.append("".join([mapper[cipher] for cipher in cipher_list]))

        mapper = {cipher: answer for cipher, answer in zip(self.df["bacon1_ab_reverse"], self.df["letter"])}
        if set(cipher_list).issubset(set(mapper.keys())):
            answer_list.append("".join([mapper[cipher] for cipher in cipher_list]))

        mapper1 = {cipher: answer for cipher, answer in zip(self.df["bacon2_ab"], self.df["letter"])}
        mapper2 = mapper1.copy()
        mapper3 = mapper1.copy()
        mapper4 = mapper1.copy()
        mapper2["ABAAA"] = "i"
        mapper3["BAABB"] = "u"
        mapper4["ABAAA"] = "i"
        mapper4["BAABB"] = "u"
        if set(cipher_list).issubset(set(mapper1.keys())):
            answer_list.append("".join([mapper1[cipher] for cipher in cipher_list]))
            answer_list.append("".join([mapper2[cipher] for cipher in cipher_list]))
            answer_list.append("".join([mapper3[cipher] for cipher in cipher_list]))
            answer_list.append("".join([mapper4[cipher] for cipher in cipher_list]))
        
        mapper1 = {cipher: answer for cipher, answer in zip(self.df["bacon2_ab_reverse"], self.df["letter"])}
        mapper2 = mapper1.copy()
        mapper3 = mapper1.copy()
        mapper4 = mapper1.copy()
        mapper2["ABAAA"] = "i"
        mapper3["BAABB"] = "u"
        mapper4["ABAAA"] = "i"
        mapper4["BAABB"] = "u"
        if set(cipher_list).issubset(set(mapper1.keys())):
            answer_list.append("".join([mapper1[cipher] for cipher in cipher_list]))
            answer_list.append("".join([mapper2[cipher] for cipher in cipher_list]))
            answer_list.append("".join([mapper3[cipher] for cipher in cipher_list]))
            answer_list.append("".join([mapper4[cipher] for cipher in cipher_list]))

        return " or ".join(list(set(answer_list)))
    
    def decode(self, cipher_list, cipher_type):
        cipher_list = [cipher.upper() for cipher in cipher_list]
        if cipher_type == "modulo":
            cipher_list = [self.modulo(cipher) for cipher in cipher_list]
            mapper = {cipher: answer for cipher, answer in zip(self.df["base10"], self.df["letter"])}
            del mapper['26']
            mapper['0'] = 'z'
            return "".join([mapper[cipher] for cipher in cipher_list])

        if self.validate(cipher_list, cipher_type):
            if cipher_type == "periodic":
                return self.decode_periodic(cipher_list)
            elif cipher_type == "polybius":
                return self.decode_polybius(cipher_list)
            elif cipher_type == "bacon":
                return self.decode_bacon(cipher_list)
            else:
                mapper = {cipher: answer for cipher, answer in zip(self.df[cipher_type], self.df["letter"])}
                return "".join([mapper[cipher] for cipher in cipher_list])
        else:
            return ""
        
    def decode_all(self, cipher_list):
        cipher_list = [cipher.upper() for cipher in cipher_list]
        possible_type_list = self.analyze(cipher_list)
        all_answers = {cipher_type: self.decode(cipher_list, cipher_type) for cipher_type in possible_type_list}
        return all_answers
    
    def caesar(self, cipher, shift):
        cipher = cipher.lower()
        letter_list = list(cipher)
        letter_to_number_mapper = {letter: number for letter, number in zip(self.df["letter"], range(26))}
        number_list = [letter_to_number_mapper[letter] for letter in letter_list]
        number_list_shifted = [(number+shift)%26 for number in number_list]
        number_to_letter_mapper = {number: letter for number, letter in zip(range(26), self.df["letter"])}
        letter_list_shifted = [number_to_letter_mapper[number] for number in number_list_shifted]
        return "".join(letter_list_shifted)
    
    def caesar_all(self, cipher):
        all_answers = {shift: self.caesar(cipher, shift) for shift in range(26)}
        return all_answers
    
    def vigenere(self, cipher, key, reverse=False):
        cipher = cipher.lower()
        letter_list = list(cipher)
        letter_to_number_mapper = {letter: number for letter, number in zip(self.df["letter"], range(26))}
        number_list = [letter_to_number_mapper[letter] for letter in letter_list]
        key_length = len(key)
        key_letter_list = [key[index%key_length] for index in range(len(cipher))]
        key_number_list = [letter_to_number_mapper[letter] for letter in key_letter_list]
        number_to_letter_mapper = {number: letter for number, letter in zip(range(26), self.df["letter"])}
        if reverse:
            answer_list = [number_to_letter_mapper[(num1+num2)%26] for num1, num2 \
                           in zip(number_list, key_number_list)]
            answer = "".join(answer_list)
        else:
            answer_list = [number_to_letter_mapper[(num1-num2)%26] for num1, num2 \
                           in zip(number_list, key_number_list)]
            answer = "".join(answer_list)
        return answer
    
    @staticmethod
    def reverse(cipher):
        cipher = cipher.lower()
        letter_list = list(cipher)
        letter_list.reverse()
        return "".join(letter_list)

    @staticmethod
    def modulo(number):
        return str(int(number) % 26)
    
    @staticmethod
    def scytale(cipher, cycle):
        cipher = cipher.lower()
        letter_pos_tuples = [(char, pos) for pos, char in enumerate(cipher)]
        letter_lists = [[char for (char, pos) in letter_pos_tuples if pos%cycle == r] for r in range(cycle)]
        answer_list = ["".join(letter_list) for letter_list in letter_lists]
        return "".join(answer_list)
    
    @staticmethod
    def scytale_all(cipher):
        cipher = cipher.lower()
        all_answers = {cycle: Decoder.scytale(cipher, cycle) for cycle in range(2, len(cipher))}
        return all_answers
    
    def compute_freq(self, word_tuple: Iterable[str]):
        """Compute the product of frequency of a word tuple. E.g. words = ("hello", "world")"""
        keys = self.word_freq.keys()
        output = [self.word_freq[x] if x in keys else self.lowest_freq for x in word_tuple]
        return np.prod(output)
    
    def find_top_k(self, word_tuple_list: List[Iterable[str]], get_top_k: int):
        """
        Get the top k word tuples with the highest frequency product
        E.g. words_list = [("word_1_1", "word_1_2"), (word_2_1, word_2_2), ... (word_N_1, word_N_2)]
        """
        freq = [(self.compute_freq(x), x) for x in word_tuple_list]
        freq = sorted(freq, reverse=True)
        freq = [x[1] for x in freq]
        output = freq[:get_top_k]
        output = [" ".join(x) for x in output]
        return output

    def crossword(self, cipher: str, get_top_k: int = 0, debug=False):
        """The format of the cipher is like 'g*itar**t' for 'guitarist'"""
        first_iter = True
        crossword_lookup_dir = "./en_dict/crossword_lookup"
        word_len = len(cipher)
        letter_info = {idx: letter for idx, letter in enumerate(list(cipher)) if letter != "*"}
        
        # If no letter_info is given, return all words of length word_len
        if letter_info == {}:
            path = f"./en_dict/length_lookup/len_{word_len}.txt"
            with open(path, "r") as f:
                output = f.read().split("\n")
            return output
        
        for pos, letter in letter_info.items():
            path = os.path.join(crossword_lookup_dir, letter, f"len_{word_len}", f"pos_{pos}.txt")

            if not os.path.isfile(path):
                if debug:
                    print(f"No match found for word of length {word_len} with letter {letter} in position {pos} (0-based)")
                return []
            
            with open(path, "r") as f:
                matched = f.read().split("\n")
            if first_iter:
                first_iter = False
                output = set(matched)
            else:
                output = output.intersection(set(matched))
                if len(output) == 0:
                    if debug:
                        print("No match found for the given letter_info")
                    return []
        output = list(output)

        # Get only the top k matches if necessary
        if get_top_k > 0:
            output = [(x,) for x in output]
            output = self.find_top_k(output, get_top_k)

        return output
    
    def crossword2(self, cipher: str, get_top_k: int = 0, debug=False):
        """
        This is used for a cipher made of 2 words. The code will attempt to split by all possible
        positions and return the 2 resulting words
        E.g. The cipher is 'he**ow**ld' for 'hello world'
        """
        word_len = len(cipher)
        if word_len < 2:
            raise ValueError(f"Length of cipher must be at least 2, got {word_len} instead")
        else:
            two_substrings = []
            for first_half_len in range(1, word_len):
                first_half_substring = cipher[:first_half_len]
                second_half_substring = cipher[first_half_len:]
                first_half_match = self.crossword(first_half_substring)
                if first_half_match == []:
                    continue
                second_half_match = self.crossword(second_half_substring)
                if second_half_match == []:
                    continue
                new_matches = list(product(first_half_match, second_half_match))
                two_substrings += new_matches
                if debug:
                    print(f"Found new matches: {new_matches}")

            # Get only the top k matches if necessary
            if get_top_k > 0:
                output = self.find_top_k(two_substrings, get_top_k)
            else:
                output = [" ".join(x) for x in two_substrings]

            return output
    
    def crossword_recursive(self, cipher: str, nbr_words: int = 1):
        """
        This is used for a cipher made of nbr_words words. The code will attempt to split by all possible
        positions and return the resulting words
        E.g. The cipher is 's*e**u*at*r' for 'see you later'
        """
        word_len = len(cipher)
        if nbr_words == 1:
            return self.crossword(cipher)
        elif word_len < nbr_words:
            return []
        else:
            two_substrings = []
            for first_half_len in range(1, word_len):
                first_half_substring = cipher[:first_half_len]
                second_half_substring = cipher[first_half_len:]
                first_half_match = self.crossword(first_half_substring)
                if first_half_match == []:
                    continue
                second_half_match = self.crossword_recursive(second_half_substring, nbr_words-1)
                if second_half_match == []:
                    continue
                new_matches = list(product(first_half_match, second_half_match))
                two_substrings += new_matches
            output = [" ".join(x) for x in two_substrings]
            return output

    def crossword_plus(self, cipher: str, nbr_words: int = 1, get_top_k: int = 0):
        output = self.crossword_recursive(cipher, nbr_words)
        if get_top_k > 0:
            output = [x.split(" ") for x in output]
            output = self.find_top_k(output, get_top_k)
        return output
    
    def crossword_known_space(self, cipher: str, get_top_k: int = 0):
        cipher_list = cipher.split(" ")
        matched_parts = []
        for c in cipher_list:
            matched = self.crossword(c)
            if matched == []:
                return []
            matched_parts.append(matched)
        output = list(product(*matched_parts))
        
        if get_top_k > 0:
            output = self.find_top_k(output, get_top_k)
        else:
            output = [" ".join(x) for x in output]
        return output

    def prefix_search(self, prefix: str, word_len: int):
        """E.g. Find all words that start with 'he' and of length 5"""
        if word_len < len(prefix):
            raise ValueError(f"word_len must be no less than the length of prefix, got prefix: {prefix} and word_len: {word_len} instead")
        
        letters = list(prefix)
        first_iter = True
        for pos, letter in enumerate(letters):
            path = f"./en_dict/crossword_lookup/{letter}/len_{word_len}/pos_{pos}.txt"
            if os.path.isfile(path):
                with open(path, "r") as f:
                    matched = f.read().split("\n")
                if first_iter:
                    first_iter = False
                    output = set(matched)
                else:
                    output = output.intersection(set(matched))
                    if output == set():
                        return []
            else:
                return []
        return list(output)
    
    def permute(self, cipher):
        all_comb = set(permutations(list(cipher)))
        all_comb = ["".join(x) for x in all_comb]
        return all_comb
    
    def anagram_basic(self, cipher):
        """A brute force version that searches any combination of letters through the entire dictionary"""
        all_comb = self.permute(cipher)
        with open("./en_dict/en.txt", "r") as f:
            all_words = f.read().split("\n")
        output = [x for x in all_comb if x in all_words]
        return output

    def anagram(self, cipher, prefix_len: int = 1, get_top_k: int = 0, debug: bool = False):
        """An improved version that searches prefix then words. It seems that prefix_len = 1 gives the best result"""
        word_len = len(cipher)
        if word_len < prefix_len:
            raise ValueError(f"Length of cipher must be no less than the length of prefix, got cipher: {cipher} and prefix_len: {prefix_len} instead")

        all_comb = self.permute(cipher)
        all_prefix = set([x[:prefix_len] for x in all_comb])
        output = []
        for prefix in all_prefix:
            matched = self.prefix_search(prefix, word_len)
            if matched == []:
                continue
            else:
                available_words = [x for x in all_comb if x.startswith(prefix)]
                matched = set(matched).intersection(set(available_words))
                if matched != [] and debug:
                    print(f"Found new matches: {matched}")
                output += list(matched)

        if get_top_k > 0:
            output = [(x,) for x in output]
            output = self.find_top_k(output, get_top_k)

        return output
    
    def anagram2(self, cipher, get_top_k: int = 0, debug=False):
        """Solve anagram made of 2 words"""
        word_len = len(cipher)
        if word_len == 1:
            return cipher
        else:
            all_comb = self.permute(cipher)
            two_substrings = []
            cache = {}
            for comb in all_comb:
                for first_half_len in range(1, word_len):
                    first_half_substring = comb[:first_half_len]
                    first_half_letters = str(sorted(list(first_half_substring)))
                    second_half_substring = comb[first_half_len:]
                    second_half_letters = str(sorted(list(second_half_substring)))
                    if first_half_letters in cache: # in which case second_half_letters must be in cache too
                        continue
                    else:
                        first_half_match = self.anagram(first_half_substring)
                        cache[first_half_letters] = first_half_match
                        second_half_match = self.anagram(second_half_substring)
                        cache[second_half_letters] = second_half_match
                        if first_half_match == [] or second_half_match == []:
                            continue
                        new_matches = list(product(first_half_match, second_half_match))
                        two_substrings += new_matches
                        if debug:
                            print(f"Found new matches: {new_matches}")

            two_substrings += [(x[1], x[0]) for x in two_substrings]
            two_substrings = list(set(two_substrings))
            if get_top_k > 0:
                output = self.find_top_k(two_substrings, get_top_k)
            else:
                output = [" ".join(x) for x in two_substrings]

        return output
    
    def extract_1_letter(self, word_list: List[str]):
        """The goal of this is to extract 1 letter from each word and return a new word using these letters"""
        pass

def prepare_length_lookup(source="./en_dict/en.txt", output_directory="./en_dict/length_lookup"):
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

def prepare_crossword_lookup(source="./en_dict/en.txt", output_directory="./en_dict/crossword_lookup"):
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

def prepare_word_freq(output_path="./en_dict/word_freq.json"):
    df = pd.read_csv("./unigram_freq.csv")
    df["count"] /= 1e9
    word_freq = {df.loc[idx, "word"]: df.loc[idx, "count"] for idx in range(len(df))}
    with open(output_path, "w") as f:
        json.dump(word_freq, f)

# prepare_crossword_lookup()
# prepare_length_lookup()
# prepare_word_freq()
# d = Decoder()