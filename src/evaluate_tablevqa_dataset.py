"""
Copied from https://github.com/allenai/allennlp-semparse
"""

import json
import os
import re
import unicodedata
import time
import numpy as np

from abc import ABCMeta, abstractmethod
from codecs import open
from math import isinf, isnan
from typing import List

def evaluate_datasets(input_paths: List[str], output_folder: str, score_keys: str):    
    scores, num_samples = [], []
    for input_path in input_paths:
        data = json.load(open(input_path, "r", encoding="utf-8"))
        input_json_name = input_path.split("/")[-1]
        input_name = input_json_name.split(".")[0]
        output_path = os.path.join(output_folder, input_json_name)   
        if "vwtq" in input_path:
            data, meta = evaluate_wtq(data, score_keys)     
        elif "vtabfact" in input_path:
            data, meta = evaluate_tabfact(data, score_keys)
        elif "fintabnetqa" in input_path:
            data, meta = evaluate_fintabnet(data, score_keys)
        else:
            raise NotImplementedError (f"# [info] {input_path} cannot be evaluated.")
        score = meta['average_scores'][0]
        print(f"# [info] {input_name} Accuracy: {round(score, 1)}")
        scores.append(score)
        num_samples.append(len(data))
        evaluated_data = {"instances": data, "evaluation_meta": meta}
        with open(output_path, "w", encoding="utf-8") as outfile:
            json.dump(evaluated_data, outfile, ensure_ascii=False)
    
    scores = np.array(scores)
    num_samples = np.array(num_samples)
    average_score = sum(scores * num_samples) / sum(num_samples)
    print(f"# [info] Average Accuracy: {round(average_score, 1)}")

def evaluate_tabfact(data, score_keys):
    num_examples = 0
    num_correct = 0
    manual_check = 0
    start_time = time.time()
    for instance in data:
        if instance["pred"] == None:
            instance["pred"] = "none"
        pred = instance["pred"].lower()
        gt = instance["gt"]
        num_examples += 1
        if "true" in pred and "false" in pred:
            manual_check += 1
            score = None
        elif "true" in pred and gt == 1:
            num_correct += 1
            score = 1
        elif "false" in pred  and gt == 0:
            num_correct += 1
            score = 1
        else:
            score = 0
        instance["scores"] = {                    
                    score_keys[0]: score,
                }
    if manual_check > 0:
        print(f"the number of not properly parsed samples: {manual_check}")
    end_time = time.time()
    elapsed_time = end_time - start_time
    Accuracy = round((num_correct + 1e-9) / (num_examples + 1e-9), 8) * 100
    meta = {
            "evaluators": "correctness",
            "score_info": [score_keys[0]],
            "evaluated_time": elapsed_time,
            "total_num_sample": len(data),
            f"average_scores": [Accuracy],            
        }
    return data, meta

def evaluate_wtq(data, score_keys):
    num_examples = 0
    num_correct = 0
    start_time = time.time()
    
    for instance in data:
        pred = instance['pred'].replace("||","|") 
        gt = instance["gt"]
        original_strings = tsv_unescape_list(gt)           
        target_values = to_value_list(original_strings)

        predicted_strings = tsv_unescape_list(pred)
        predicted_values = to_value_list(predicted_strings)
        correct = check_denotation(target_values, predicted_values)       
        num_examples += 1
        score = 0
        if correct:
            num_correct += 1
            score = 1
        instance["scores"] = {                    
                    score_keys[0]: score,
                }

    end_time = time.time()
    elapsed_time = end_time - start_time
    
    Accuracy = round((num_correct + 1e-9) / (num_examples + 1e-9), 8) * 100
    meta = {
            "evaluators": "correctness",
            "score_info": [score_keys[0]],
            "evaluated_time": elapsed_time,
            "total_num_sample": len(data),
            f"average_scores": [Accuracy],            
        }
    return data, meta

def evaluate_fintabnet(data, score_keys):
    num_examples = 0
    num_correct, _num_correct = 0, 0
    start_time = time.time()
    for instance in data:
        pred, preds = fintabnet_normalize(instance["pred"])
        gt, gts = fintabnet_normalize(instance["gt"])
        correct = 1 if gt == pred else 0
        _correct = any(_pred == _gt for _pred in preds for _gt in gts)
        num_examples += 1
        score, _score = 0, 0        
        if correct:
            num_correct += 1
            score = 1
        if _correct:
            _num_correct += 1
            _score = 1       
        instance["scores"] = {                    
                    score_keys[0]: _score,
                    "exact_score": score,
                }    

    end_time = time.time()
    elapsed_time = end_time - start_time
    Accuracy = round((num_correct + 1e-9) / (num_examples + 1e-9), 8) * 100
    _Accuracy = round((_num_correct + 1e-9) / (num_examples + 1e-9), 8) * 100
    meta = { 
            "evaluators": "correctness",
            "score_info": ["relieved_accuracy", score_keys[0]],
            "evaluated_time": elapsed_time,
            "total_num_sample": len(data),
            f"average_scores": [_Accuracy, Accuracy],             
        }
    return data, meta

def fintabnet_normalize(s):
    s = normalize(s)
    remove_words = [
        "dollar", "gallons", "square feet", "shares", "mbtu", "mbpd", "mbbls", "mmbtu", "unit", "gwh", "year", "mmcf", "mile", "mboe"
    ]  

    # Data specific filtering using regular expressions
    # Remove special characters like $, (, and )
    s = re.sub(r'[\$\(\),]', "", s)

    # Replace "dollar" with empty string if it's not part of another word    
    pattern = r"\b(" + "|".join(remove_words) + r")s?\b" 
    s = re.sub(pattern, '', s, flags=re.IGNORECASE)      
    
    # Unit conversion dictionary with regex patterns for flexibility
    unit_conversion = {
        r' \bthousand\b': "e3",
        r' \bmillion\b': "e6",
        r' \bbillion\b': "e9",
        r'\bthousand\b': "e3",
        r'\bmillion\b': "e6",
        r'\bbillion\b': "e9",
        r' ?%': "e-2",
    }
    
    
    # Convert percentages to their decimal representation.
    # Applying this after unit_conversion prevents "percent" from being processed
    # in cases like "million %", which would be incorrect.  
    # s = re.sub(r' ?%', 'e-2', s)
    # s_percent = re.sub(r' ?%', '', s_percent)

    s_unit_free = s

    # Iterate over unit_conversion and apply transformations
    for pattern, value in unit_conversion.items():
        s = re.sub(pattern, value, s)        
        s_unit_free = re.sub(pattern, "", s_unit_free)            

    # Attempt to convert to float
    try:
        return float(s), [float(s), float(s_unit_free)]
    except ValueError as e:
        # Return the original string and the error for debugging purposes
        return s, [s, s_unit_free]
    
def normalize(x):
    if not isinstance(x, str):
        x = x.decode("utf8", errors="ignore")
    # Remove diacritics
    x = "".join(
        c for c in unicodedata.normalize("NFKD", x) if unicodedata.category(c) != "Mn"
    )
    # Normalize quotes and dashes
    x = re.sub(r"[‘’´`]", "'", x)
    x = re.sub(r"[“”]", '"', x)
    x = re.sub(r"[‐‑‒–—−]", "-", x)
    while True:
        old_x = x
        # Remove citations
        x = re.sub(r"((?<!^)\[[^\]]*\]|\[\d+\]|[•♦†‡*#+])*$", "", x.strip())
        # Remove details in parenthesis
        x = re.sub(r"(?<!^)( \([^)]*\))*$", "", x.strip())
        # Remove outermost quotation mark
        x = re.sub(r'^"([^"]*)"$', r"\1", x.strip())
        if x == old_x:
            break
    # Remove final '.'
    if x and x[-1] == ".":
        x = x[:-1]
    # Collapse whitespaces and convert to lower case
    x = re.sub(r"\s+", " ", x, flags=re.U).lower().strip()
    return x


################ Value Types ################


class Value(object):
    __metaclass__ = ABCMeta

    # Should be populated with the normalized string
    _normalized = None

    @abstractmethod
    def match(self, other):
        """Return True if the value matches the other value.

        Args:
            other (Value)
        Returns:
            a boolean
        """
        pass

    @property
    def normalized(self):
        return self._normalized


class StringValue(Value):
    def __init__(self, content):
        assert isinstance(content, str)
        self._normalized = normalize(content)
        self._hash = hash(self._normalized)

    def __eq__(self, other):
        return isinstance(other, StringValue) and self.normalized == other.normalized

    def __hash__(self):
        return self._hash

    def __str__(self):
        return "S" + str([self.normalized])

    def __repr__(self):
        return self.__str__()

    def match(self, other):
        assert isinstance(other, Value)
        return self.normalized == other.normalized


class NumberValue(Value):
    def __init__(self, amount, original_string=None):
        assert isinstance(amount, (int, float))
        if abs(amount - round(amount)) < 1e-6:
            self._amount = int(amount)
        else:
            self._amount = float(amount)
        if not original_string:
            self._normalized = str(self._amount)
        else:
            self._normalized = normalize(original_string)
        self._hash = hash(self._amount)

    @property
    def amount(self):
        return self._amount

    def __eq__(self, other):
        return isinstance(other, NumberValue) and self.amount == other.amount

    def __hash__(self):
        return self._hash

    def __str__(self):
        return "N({})".format(self.amount) + str([self.normalized])

    def __repr__(self):
        return self.__str__()

    def match(self, other):
        assert isinstance(other, Value)
        if self.normalized == other.normalized:
            return True
        if isinstance(other, NumberValue):
            return abs(self.amount - other.amount) < 1e-6
        return False

    @staticmethod
    def parse(text):
        """Try to parse into a number.

        Return:
            the number (int or float) if successful; otherwise None.
        """
        try:
            return int(text)
        except ValueError:
            try:
                amount = float(text)
                assert not isnan(amount) and not isinf(amount)
                return amount
            except ValueError:
                return None


class DateValue(Value):
    def __init__(self, year, month, day, original_string=None):
        """Create a new DateValue. Placeholders are marked as -1."""
        assert isinstance(year, int)
        assert isinstance(month, int) and (month == -1 or 1 <= month <= 12)
        assert isinstance(day, int) and (day == -1 or 1 <= day <= 31)
        assert not (year == month == day == -1)
        self._year = year
        self._month = month
        self._day = day
        if not original_string:
            self._normalized = "{}-{}-{}".format(
                year if year != -1 else "xx",
                month if month != -1 else "xx",
                day if day != "-1" else "xx",
            )
        else:
            self._normalized = normalize(original_string)
        self._hash = hash((self._year, self._month, self._day))

    @property
    def ymd(self):
        return (self._year, self._month, self._day)

    def __eq__(self, other):
        return isinstance(other, DateValue) and self.ymd == other.ymd

    def __hash__(self):
        return self._hash

    def __str__(self):
        return ("D(%d,%d,%d)" % (self._year, self._month, self._day)) + str(
            [self._normalized]
        )

    __repr__ = __str__

    def match(self, other):
        assert isinstance(other, Value)
        if self.normalized == other.normalized:
            return True
        if isinstance(other, DateValue):
            return self.ymd == other.ymd
        return False

    @staticmethod
    def parse(text):
        """Try to parse into a date.

        Return:
            tuple (year, month, date) if successful; otherwise None.
        """
        try:
            ymd = text.lower().split("-")
            assert len(ymd) == 3
            year = -1 if ymd[0] in ("xx", "xxxx") else int(ymd[0])
            month = -1 if ymd[1] == "xx" else int(ymd[1])
            day = -1 if ymd[2] == "xx" else int(ymd[2])
            assert not (year == month == day == -1)
            assert month == -1 or 1 <= month <= 12
            assert day == -1 or 1 <= day <= 31
            return (year, month, day)
        except:
            return None


################ Value Instantiation ################


def to_value(original_string, corenlp_value=None):
    """Convert the string to Value object.

    Args:
        original_string (basestring): Original string
        corenlp_value (basestring): Optional value returned from CoreNLP
    Returns:
        Value
    """
    if isinstance(original_string, Value):
        # Already a Value
        return original_string
    if not corenlp_value:
        corenlp_value = original_string
    # Number?
    amount = NumberValue.parse(corenlp_value)
    if amount is not None:
        return NumberValue(amount, original_string)
    # Date?
    ymd = DateValue.parse(corenlp_value)
    if ymd is not None:
        if ymd[1] == ymd[2] == -1:
            return NumberValue(ymd[0], original_string)
        else:
            return DateValue(ymd[0], ymd[1], ymd[2], original_string)
    # String.
    return StringValue(original_string)


def to_value_list(original_strings, corenlp_values=None):
    """Convert a list of strings to a list of Values

    Args:
        original_strings (list[basestring])
        corenlp_values (list[basestring or None])
    Returns:
        list[Value]
    """
    assert isinstance(original_strings, (list, tuple, set))
    if corenlp_values is not None:
        assert isinstance(corenlp_values, (list, tuple, set))
        assert len(original_strings) == len(corenlp_values)
        return list(
            set(to_value(x, y) for (x, y) in zip(original_strings, corenlp_values))
        )
    else:
        return list(set(to_value(x) for x in original_strings))


################ Check the Predicted Denotations ################


def check_denotation(target_values, predicted_values):
    """Return True if the predicted denotation is correct.

    Args:
        target_values (list[Value])
        predicted_values (list[Value])
    Returns:
        bool
    """
    # Check size
    if len(target_values) != len(predicted_values):
        return False
    # Check items
    for target in target_values:
        if not any(target.match(pred) for pred in predicted_values):
            return False
    return True


################ Batch Mode ################


def tsv_unescape(x):
    """Unescape strings in the TSV file.
    Escaped characters include:
        newline (0x10) -> backslash + n
        vertical bar (0x7C) -> backslash + p
        backslash (0x5C) -> backslash + backslash

    Args:
        x (str or unicode)
    Returns:
        a unicode
    """
    return x.replace(r"\n", "\n").replace(r"\p", "|").replace("\\\\", "\\")


def tsv_unescape_list(x):
    """Unescape a list in the TSV file.
    List items are joined with vertical bars (0x5C)

    Args:
        x (str or unicode)
    Returns:
        a list of unicodes
    """
    return [tsv_unescape(y) for y in x.split("|")]