from __future__ import annotations

import os
from typing import Any
from resiliparse.parse.encoding import detect_encoding
from resiliparse.extract.html2text import extract_plain_text
import fasttext
import re
import mmh3
import unicodedata
import shutil
from collections import defaultdict


lang_model = fasttext.load_model("data/classifiers/lid.176.bin")
own_classifier = fasttext.load_model("data/quality.bin")

def run_extract_text_from_html_bytes(html_bytes: bytes) -> str | None:
    encoding = detect_encoding(html_bytes)
    raw_str = html_bytes.decode(encoding,  errors="ignore")
    return extract_plain_text(raw_str)


def run_identify_language(text: str) -> tuple[Any, float]:
    text = text.replace("\n", " ")
    res = lang_model.predict(text)
    return (res[0][0].removeprefix('__label__'), res[1][0]) 


def run_mask_emails(text: str) -> tuple[str, int]:
    email_re = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    return re.subn(email_re, "|||EMAIL_ADDRESS|||", text)
    

def run_mask_phone_numbers(text: str) -> tuple[str, int]:
    phone_re = r"(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}"
    return re.subn(phone_re, "|||PHONE_NUMBER|||", text)


def run_mask_ips(text: str) -> tuple[str, int]:
    ipv4_re  = r"\b(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\b"
    return re.subn(ipv4_re, "|||IP_ADDRESS|||", text)


def run_classify_nsfw(text: str) -> tuple[Any, float]:
    nsfw = fasttext.load_model("data/classifiers/dolma_fasttext_nsfw_jigsaw_model.bin")
    text = text.replace("\n", " ")
    res = nsfw.predict(text)
    return (res[0][0].removeprefix('__label__'), res[1][0]) 


def run_classify_toxic_speech(text: str) -> tuple[Any, float]:
    hate = fasttext.load_model("data/classifiers/dolma_fasttext_hatespeech_jigsaw_model.bin")
    text = text.replace("\n", " ")
    res = hate.predict(text)
    return (res[0][0].removeprefix('__label__'), res[1][0]) 


def run_classify_quality(text: str) -> tuple[Any, float]:
    text = text.replace("\n", " ")
    res = own_classifier.predict(text)
    return (res[0][0].removeprefix('__label__'), res[1][0]) 


def run_gopher_quality_filter(text: str) -> bool:
    text_no_line = text.replace("\n", " ")
    words = text_no_line.split(" ")
    if len(words) < 50 or len(words) > 100_000:
        return False
    
    word_len_sum = 0
    for w in words:
        word_len_sum += len(w)
    
    mean_word_len = word_len_sum / len(words)

    if mean_word_len < 3 or mean_word_len > 10:
        return False

    lines = text.split("\n")

    if len([l for l in lines if l.endswith("...")])/len(lines) > 0.3:
        return False


    if len([w for w in words if len(w) ==1 ]) / len(words) > 0.2: 
        return False
    
    return True


def run_exact_line_deduplication(
    input_files: list[os.PathLike], output_directory: os.PathLike
):
    counter = defaultdict(int)
    for file_path in input_files:
        with open(file_path, "r") as f:
            for line in f:
                hash = mmh3.hash128(line)
                counter[hash] += 1

    os.makedirs(output_directory, exist_ok=True)

    for file_path in input_files:
        res = []
        with open(file_path, "r") as f:
            for line in f:
                hash = mmh3.hash128(line)
                if counter[hash] == 1:
                    res.append(line)

        out_path = os.path.join(output_directory, os.path.basename(file_path))
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("".join(res))
        

def string_preprocess(text):
    text = unicodedata.normalize("NFD", text)                       # decompose: é -> e + ́
    text = "".join(c for c in text if not unicodedata.combining(c)) # drop the combining marks
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = " ".join(text.split())
    return text

def get_ngrams(text, n):
    curstr = []
    ngrams_set = set()
    words = text.split()

    for i in range(len(words)):
        curstr.append(words[i])
        if i >= n - 1:
            ngrams_set.add(" ".join(curstr))
            curstr  = curstr[1:]
    
    return ngrams_set

def min_hash_one(ngrams, seed):
    hashes = [mmh3.hash(ngram, seed = seed) for ngram in ngrams]
    return min(hashes)

class FileWithHash:
    def __init__(self, file_path, num_hashes, num_bands, ngrams, ind):
        self.file_path = file_path
        self.ngrams = ngrams
        self.ind = ind
        self.ngrams_set = self.get_ngrams_set()
        self.min_hash_vec = [min_hash_one(self.ngrams_set, i) for i in range(num_hashes)]
        band_len = num_hashes // num_bands
        self.bands = []
        for i in range(num_bands):
            startInd = i * band_len
            endInd = (i+1) * band_len
            self.bands.append(tuple(self.min_hash_vec[startInd:endInd]))

    def get_ngrams_set(self):
        with open(self.file_path, "r") as f: 
            text = f.read()
            processed_text = string_preprocess(text)
            ngrams_set = get_ngrams(processed_text, self.ngrams)
            return ngrams_set

def find(parent_dict, x):
    parent_dict.setdefault(x, x)          # new element is its own root
    while parent_dict[x] != x:
        parent_dict[x] = parent_dict[parent_dict[x]]   # path compression (halving)
        x = parent_dict[x]
    return x

def union(parent_dict, a, b):
    parent_dict[find(parent_dict, a)] = find(parent_dict, b)        # point one root at the other


def jaccard(a, b):
    ngrams_a = a.ngrams_set
    ngrams_b = b.ngrams_set

    return len(ngrams_a & ngrams_b) / len(ngrams_a | ngrams_b)
  
def run_minhash_deduplication(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
):

    files = []

    for i, file in enumerate(input_files):
        file_with_hash = FileWithHash(file, num_hashes, num_bands, ngrams, i)
        files.append(file_with_hash)

    # the structure should be map[i][band_tuple] is the set of files that have band_tuple at index i. Meaning they pairwise collides.
    # for each pair, we compute Jacard if positive, we add to union set. 

    parent_dict = {}

    for i in range(num_bands):
        current_band_sets = defaultdict(list)
        for j, file in enumerate(files):
            band = files[j].bands[i]
            current_band_sets[band].append(file)

        for band in current_band_sets:
            colliding_files = current_band_sets[band]

            for k in range(len(colliding_files) -1):
                for l in range(k+1, len(colliding_files)):
                    if jaccard(colliding_files[k], colliding_files[l]) >= jaccard_threshold:
                        union(parent_dict, colliding_files[k].ind, colliding_files[l].ind)
    
    os.makedirs(output_directory, exist_ok=True)


    for i, file in enumerate(input_files):
        if find(parent_dict, i) == i:
            dst = os.path.join(output_directory, os.path.basename(file))
            shutil.copyfile(file, dst) 

