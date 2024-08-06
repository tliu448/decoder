import json
import numpy as np
import os

from itertools import permutations, product
from typing import Callable, Dict, Iterable, List, Union

import time

DICTIONARY_DIR = "./en_dict"

class WordFinder:

    def __init__(self):
        with open(os.path.join(DICTIONARY_DIR, "en_word_freq_333k.json"), "r") as f:
            self.word_freq = json.load(f)
        self.lowest_freq = min(self.word_freq.values())
        self.anagram_cache = {}
    
    @staticmethod
    def replace_letter(word: str, replacement: Dict[int, str]) -> List[str]:
        """
        Replace one or more letters in a word by replacement
        """
        output = []
        letter_list = list(word)
        letters_to_replace = [list(set(x)) for x in replacement.values()]
        all_comb = list(product(*letters_to_replace))
        all_pos = replacement.keys()
        for comb in all_comb:
            new_letter_list = letter_list
            for idx, pos in enumerate(all_pos):
                new_letter_list[pos] = comb[idx]
            new_word = "".join(new_letter_list)
            output.append(new_word)
        return output
    
    def compute_freq(self, word_tuple: Iterable[str]):
        """Compute the product of frequency of a word tuple. E.g. words = ("hello", "world")"""
        keys = self.word_freq.keys()
        output = [self.word_freq[x] if x in keys else self.lowest_freq for x in word_tuple]
        return np.prod(output)
    
    def find_top_k(self, word_list: List[str], get_top_k: int = 0) -> List[str]:
        """
        Get the top k collocations with the highest frequency product
        E.g. words_list = ["word_1_1 word_1_2", "word_2_1 word_2_2", ... "word_N_1 word_N_2"]
        NOTE: This naive approach ignores linguistic correlation between words
        """
        word_list = [x.split(" ") for x in word_list]
        freq = [(self.compute_freq(x), x) for x in word_list]
        freq = sorted(freq, reverse=True)
        freq = [x[1] for x in freq]
        output = freq[:get_top_k]
        output = [" ".join(x) for x in output]
        return output

    def _crossword(self, cipher: str, debug: str = False) -> List[str]:
        first_iter = True
        lookup_dir = os.path.join(DICTIONARY_DIR, "letter_lookup")
        word_len = len(cipher)
        letter_info = {idx: letter for idx, letter in enumerate(list(cipher)) if letter != "*"}
        
        # If no letter_info is given, return all words of length word_len
        if letter_info == {}:
            path = os.path.join(DICTIONARY_DIR, "length_lookup", f"len_{word_len}.txt")
            with open(path, "r") as f:
                output = f.read().split("\n")
            return output
        
        for pos, letter in letter_info.items():
            path = os.path.join(lookup_dir, letter, f"len_{word_len}", f"pos_{pos}.txt")

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

        return output

    def _crossword2(self, cipher: str, debug: bool = False) -> List[str]:
        word_len = len(cipher)
        if word_len < 2:
            raise ValueError(f"Length of cipher must be at least 2, got {word_len} instead")
        else:
            two_substrings = []
            for first_half_len in range(1, word_len):
                first_half_substring = cipher[:first_half_len]
                second_half_substring = cipher[first_half_len:]
                first_half_match = self._crossword(first_half_substring)
                if first_half_match == []:
                    continue
                second_half_match = self._crossword(second_half_substring)
                if second_half_match == []:
                    continue
                new_matches = list(product(first_half_match, second_half_match))
                two_substrings += new_matches
                if debug:
                    print(f"Found new matches: {new_matches}")

            output = [" ".join(x) for x in two_substrings]
            return output
        
    def _crossword_plus(self, cipher: str, nbr_words: int = 1, debug: bool = False) -> List[str]:
        word_len = len(cipher)
        if nbr_words == 1:
            return self._crossword(cipher, debug)
        elif word_len < nbr_words:
            return []
        else:
            two_substrings = []
            for first_half_len in range(1, word_len):
                first_half_substring = cipher[:first_half_len]
                second_half_substring = cipher[first_half_len:]
                first_half_match = self._crossword(first_half_substring)
                if first_half_match == []:
                    continue
                second_half_match = self._crossword_plus(cipher=second_half_substring, nbr_words=nbr_words-1)
                if second_half_match == []:
                    continue
                new_matches = list(product(first_half_match, second_half_match))
                two_substrings += new_matches
            output = [" ".join(x) for x in two_substrings]
            return output

    def _crossword_known_space(self, cipher: str, debug: bool = False) -> List[str]:
        cipher_list = cipher.split(" ")
        matched_parts = []
        for c in cipher_list:
            matched = self._crossword(c)
            if matched == []:
                return []
            matched_parts.append(matched)
        output = list(product(*matched_parts))
        output = [" ".join(x) for x in output]
        return output
        
    def _crossword_loop(
            self, 
            basic_crossword_fn: Callable[[List[str], int, bool], List[str]], 
            cipher: Union[str, List[str]], 
            get_top_k: int = 0, 
            debug: str =False,
            **kwargs
        ) -> List[str]:
        if type(cipher) == str:
            output = basic_crossword_fn(cipher=cipher, debug=debug, **kwargs)
        else:
            output = []
            for c in cipher:
                temp_output = basic_crossword_fn(cipher=c, **kwargs)
                if temp_output != [] and debug:
                    print(f"Found {temp_output}")
                output += temp_output

        # Get only the top k matches if necessary
        if get_top_k > 0:
            output = self.find_top_k(output, get_top_k)

        return output
    
    def crossword(self, cipher: Union[str, List[str]], get_top_k: int = 0, debug=False) -> List[str]:
        """
        Crossword solver
        The format of the cipher can be a str like '**itar**t' for 'guitarist'
        It can also be a List[str] to search for a grid
        E.g. ['**itar**t', '**itan**t'] for 'guitarist
        """
        return self._crossword_loop(basic_crossword_fn=self._crossword, cipher=cipher, get_top_k=get_top_k, debug=debug)
    
    def crossword2(self, cipher: Union[str, List[str]], get_top_k: int = 0, debug=False) -> List[str]:
        """
        Crossword solver for 2 words
        """
        return self._crossword_loop(basic_crossword_fn=self._crossword2, cipher=cipher, get_top_k=get_top_k, debug=debug)
    
    def crossword_known_space(self, cipher: Union[str, List[str]], get_top_k: int = 0, debug=False) -> List[str]:
        """
        Crossword solver where the space information is known
        E.g. 'g*o* d*y' for 'good day'
        """
        return self._crossword_loop(basic_crossword_fn=self._crossword_known_space, cipher=cipher, get_top_k=get_top_k, debug=debug)

    def crossword_plus(self, cipher: Union[str, List[str]], nbr_words: int, get_top_k: int = 0, debug: bool = False) -> List[str]:
        """
        This is used for a cipher made of nbr_words words. The code will attempt to split by all possible
        positions and return the resulting words
        E.g. The cipher is 's*e**u*at*r' for 'see you later'
        """
        return self._crossword_loop(basic_crossword_fn=self._crossword_plus, cipher=cipher, nbr_words=nbr_words, get_top_k=get_top_k, debug=debug)

    def start_with_substring(self, substring: str, word_len: int, get_top_k: int = 0) -> List[str]:
        """E.g. Find all words that start with 'he' and of length 5"""
        if word_len < len(substring):
            raise ValueError(f"word_len must be no less than the length of substring, got substring: {substring} and word_len: {word_len} instead")
        
        path = os.path.join(DICTIONARY_DIR, "length_lookup", f"len_{word_len}.txt")
        if os.path.isfile(path):
            with open(path, "r") as f:
                all_words = f.read().split("\n")
            output = [x for x in all_words if x.startswith(substring)]
            if get_top_k > 0:
                return self.find_top_k(output, get_top_k)
            else:
                return output
        else:
            return []
        
    def end_with_substring(self, substring: str, word_len: int, get_top_k: int = 0) -> List[str]:
        """E.g. Find all words that end with 'lo' and of length 5"""
        if word_len < len(substring):
            raise ValueError(f"word_len must be no less than the length of substring, got substring: {substring} and word_len: {word_len} instead")
        
        path = os.path.join(DICTIONARY_DIR, "length_lookup", f"len_{word_len}.txt")
        if os.path.isfile(path):
            with open(path, "r") as f:
                all_words = f.read().split("\n")
            output = [x for x in all_words if x.endswith(substring)]
            if get_top_k > 0:
                return self.find_top_k(output, get_top_k)
            else:
                return output
        else:
            return []
    
    def contain_substring(self, substring: str, word_len: int, get_top_k: int = 0) -> List[str]:
        """E.g. Find all words that contain 'el' and of length 5"""
        if word_len < len(substring):
            raise ValueError(f"word_len must be no less than the length of substring, got substring: {substring} and word_len: {word_len} instead")
        
        path = os.path.join(DICTIONARY_DIR, "length_lookup", f"len_{word_len}.txt")
        if os.path.isfile(path):
            with open(path, "r") as f:
                all_words = f.read().split("\n")
            output = [x for x in all_words if substring in x]
            if get_top_k > 0:
                return self.find_top_k(output, get_top_k)
            else:
                return output
        else:
            return []
    
    def all_permutation(self, cipher: str) -> List[str]:
        all_comb = set(permutations(list(cipher)))
        all_comb = ["".join(x) for x in all_comb]
        return all_comb
    
    def anagram_brute_force(self, cipher) -> List[str]:
        """
        A brute force version of anagram that searches any combination of letters through the entire dictionary
        NOTE: This can be really slow
        """
        all_comb = self.all_permutation(cipher)
        with open(os.path.join(DICTIONARY_DIR, "en_370k.txt"), "r") as f:
            all_words = f.read().split("\n")
        output = [x for x in all_comb if x in all_words]
        return output

    def anagram(self, cipher, prefix_len: int = 1, get_top_k: int = 0, debug: bool = False) -> List[str]:
        """An improved version that searches prefix then words. It seems that prefix_len = 1 gives the best result"""
        word_len = len(cipher)
        if word_len < prefix_len:
            raise ValueError(f"Length of cipher must be no less than the length of prefix, got cipher: {cipher} and prefix_len: {prefix_len} instead")

        all_comb = self.all_permutation(cipher)
        all_prefix = set([x[:prefix_len] for x in all_comb])
        output = []
        for prefix in all_prefix:
            matched = self.start_with_substring(prefix, word_len)
            if matched == []:
                continue
            else:
                available_words = [x for x in all_comb if x.startswith(prefix)]
                matched = set(matched).intersection(set(available_words))
                if matched != [] and debug:
                    print(f"Found new matches: {matched}")
                output += list(matched)

        if get_top_k > 0:
            output = self.find_top_k(output, get_top_k)

        return output
    
    def anagram2(self, cipher, get_top_k: int = 0, debug=False):
        """Solve anagram made of 2 words"""
        word_len = len(cipher)
        if word_len == 1:
            return cipher
        else:
            all_comb = self.all_permutation(cipher)
            two_substrings = []
            for comb in all_comb:
                for first_half_len in range(1, word_len):
                    first_half_substring = comb[:first_half_len]
                    first_half_letters = str(sorted(list(first_half_substring)))
                    second_half_substring = comb[first_half_len:]
                    second_half_letters = str(sorted(list(second_half_substring)))
                    if first_half_letters in self.anagram_cache: # in which case second_half_letters must be in self.anagram_cache too
                        continue
                    else:
                        first_half_match = self.anagram(first_half_substring)
                        self.anagram_cache[first_half_letters] = first_half_match
                        second_half_match = self.anagram(second_half_substring)
                        self.anagram_cache[second_half_letters] = second_half_match
                        if first_half_match == [] or second_half_match == []:
                            continue
                        new_matches = list(product(first_half_match, second_half_match))
                        two_substrings += new_matches
                        if debug:
                            print(f"Found new matches: {new_matches}")

            two_substrings += [(x[1], x[0]) for x in two_substrings]
            two_substrings = list(set(two_substrings))
            output = [" ".join(x) for x in two_substrings]
            if get_top_k > 0:
                output = self.find_top_k(output, get_top_k)

        return output

    def extract_1_letter(self, word_list: List[str]):
        """The goal of this is to extract 1 letter from each word and return a new word using these letters"""
        pass

