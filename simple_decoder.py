import os
import pandas as pd

from typing import Dict, List

TABLE_DIR = "./tables"

class SimpleDecoder:
    
    def __init__(self):
        self.df = pd.read_csv(os.path.join(TABLE_DIR, "mapping.csv")).astype(str)
        self.periodic = pd.read_csv(os.path.join(TABLE_DIR, "periodic.csv")).astype(str)
    
    def validate(self, cipher_list: List[str], cipher_type: str) -> bool:
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
    
    def analyze(self, cipher_list: List[str]) -> List[str]:
        cipher_list = [cipher.upper() for cipher in cipher_list]
        cipher_type_list = list(self.df.columns[1:])
        cipher_type_list = [cipher_type for cipher_type in cipher_type_list if "bacon" not in cipher_type]
        cipher_type_list.append("periodic")
        cipher_type_list.append("bacon")
        return [cipher_type for cipher_type in cipher_type_list if self.validate(cipher_list, cipher_type)]
    
    def decode_periodic(self, cipher_list: List[str]) -> str:
        cipher_list = [cipher.upper() for cipher in cipher_list]
        cipher_list = [cipher.upper() for cipher in cipher_list]
        mapper = {cipher: answer for cipher, answer in zip(self.periodic["number"],
                                                           self.periodic["element"])}
        return "".join([mapper[cipher] for cipher in cipher_list]).lower()
    
    def decode_polybius(self, cipher_list: List[str]) -> str:
        cipher_list = [cipher.upper() for cipher in cipher_list]
        mapper = {cipher: answer for cipher, answer in zip(self.df["polybius"], self.df["letter"])}
        answer = "".join([mapper[cipher] for cipher in cipher_list])
        if "24" in cipher_list:
            answer2 = answer.replace("j", "i")
            answer = f"{answer} or {answer2}"
        return answer
    
    def decode_bacon(self, cipher_list: List[str]) -> str:
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
    
    def decode(self, cipher_list: List[str], cipher_type: str) -> str:
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
        
    def decode_all(self, cipher_list: List[str]) -> Dict[str, str]:
        cipher_list = [cipher.upper() for cipher in cipher_list]
        possible_type_list = self.analyze(cipher_list)
        all_answers = {cipher_type: self.decode(cipher_list, cipher_type) for cipher_type in possible_type_list}
        return all_answers
    
    def caesar(self, cipher: str, shift: int) -> str:
        cipher = cipher.lower()
        letter_list = list(cipher)
        letter_to_number_mapper = {letter: number for letter, number in zip(self.df["letter"], range(26))}
        number_list = [letter_to_number_mapper[letter] for letter in letter_list]
        number_list_shifted = [(number+shift)%26 for number in number_list]
        number_to_letter_mapper = {number: letter for number, letter in zip(range(26), self.df["letter"])}
        letter_list_shifted = [number_to_letter_mapper[number] for number in number_list_shifted]
        return "".join(letter_list_shifted)
    
    def caesar_all(self, cipher: str) -> Dict[int, str]:
        all_answers = {shift: self.caesar(cipher, shift) for shift in range(26)}
        return all_answers
    
    def vigenere(self, cipher: str, key: str, reverse: bool = False) -> str:
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
    def reverse(cipher: str) -> str:
        cipher = cipher.lower()
        letter_list = list(cipher)
        letter_list.reverse()
        return "".join(letter_list)

    @staticmethod
    def modulo(number: int) -> int:
        return str(int(number) % 26)
    
    @staticmethod
    def scytale(cipher: str, cycle: int) -> str:
        cipher = cipher.lower()
        letter_pos_tuples = [(char, pos) for pos, char in enumerate(cipher)]
        letter_lists = [[char for (char, pos) in letter_pos_tuples if pos%cycle == r] for r in range(cycle)]
        answer_list = ["".join(letter_list) for letter_list in letter_lists]
        return "".join(answer_list)
    
    @staticmethod
    def scytale_all(cipher: str) -> Dict[int, str]:
        cipher = cipher.lower()
        all_answers = {cycle: SimpleDecoder.scytale(cipher, cycle) for cycle in range(2, len(cipher))}
        return all_answers


