#! usr/bin/env python

# Imports
import nltk

nltk.download("omw-1.4")
nltk.download("wordnet")
import urllib.request

import gensim.downloader as api
import pandas as pd
from gensim.models.word2vec import Word2Vec
from IPython.display import display
from nltk.corpus import wordnet
from scipy import stats


# functions
def get_data(path: str) -> list:
    """reads .txt file into a list of lists of strings"""
    list_of_lines = []
    with open(path, "r") as source:
        for line in source:
            line = line.rstrip().casefold().split("\t")
            if line == False:
                continue
            else:
                list_of_lines.append(line)
    return list_of_lines


def path_similarity(list1: list, list2: list) -> list:
    """Computes the WordNet Path-Based similarity between two words"""
    n = 0
    max = 0
    similarity_results = []

    while n < len(list1):
        synset1 = wordnet.synsets(list1[n])
        synset2 = wordnet.synsets(list2[n])
        for x in synset1:
            for y in synset2:
                similarity = x.path_similarity(y)
                if similarity > max:
                    max = round(similarity, 4)
        similarity_results.append(max)
        n += 1
        max = 0
    return similarity_results


def wup_similarity(list1: list, list2: list) -> list:
    """Computes the WordNet Wu-Palmer similarity between two words"""
    n = 0
    max = 0
    similarity_results = []

    while n < len(list1):
        synset1 = wordnet.synsets(list1[n])
        synset2 = wordnet.synsets(list2[n])
        for x in synset1:
            for y in synset2:
                similarity = x.wup_similarity(y)
                if similarity > max:
                    max = round(similarity, 4)
        similarity_results.append(max)
        n += 1
        max = 0
    return similarity_results


def cos_similarity(list1: list, list2: list, model) -> list:
    """Computes the word embedding cosine similarity using gensim pretrained semantic models"""
    n = 0
    similarity_results = []

    while n < len(list1):
        cos_similarity = round(model.similarity(list1[n], list2[n]), 4)
        similarity_results.append(cos_similarity)
        n += 1
    return similarity_results


def human_score_extractor(word1: str, word2: str, test_set: list) -> str:
    """Extracts human/gold scores"""
    for item in test_set:
        if item[0] == word1 and item[1] == word2:
            return item[2]


def word_list_extractor(corpus: list, column: int) -> list:
    """Extracts a list of words to compare"""
    word_list = []
    for item in corpus:
        word = item[column]
        word_list.append(word)
    return word_list


def human_scores(list1: list, list2: list, test_set: list) -> list:
    """Checks if a pair of words is in the test_set corpus.  
    If present, it assigns it the human/gold score extracted from the test corpus.  
    Otherwise, it assings the score of 0."""
    test_pairs = list(zip(list1, list2))
    word_pairs = []
    human_pairs = []
    human_scores = []

    for item in test_set:
        word_pairs.append([item[0], item[1]])
        word_pairs = [tuple(x) for x in word_pairs]

    for item in test_pairs:
        if item in word_pairs:
            human_pairs.append(
                [item[0], item[1], human_score_extractor(item[0], item[1], test_set)]
            )
        else:
            human_pairs.append([item[0], item[1], 0])

    for item in human_pairs:
        human_scores.append(item[2])

    return human_scores


def wordnet_coverage(list1: list):
    """Computes lexical coverage against WordNet"""
    covered = 0

    for word in list1:
        if wordnet.synsets(word):
            covered += 1

    print(f"Coverage: {round((covered / len(list1)), 2) * 100}%")


def glove_coverage(list1: list, model):
    """Computes lexical coverage against a corpus of word embeddings"""
    covered = 0

    for word in list1:
        if word in model.key_to_index:
            covered += 1

    print(f"Coverage: {round((covered / len(list1)), 2) * 100}%")


# main
def main():
    wordsim_parsed = get_data("wordsim353.txt")
    words_a = ["jaguar", "jaguar", "king", "king", "tiger", "tiger"]
    words_b = ["cat", "car", "queen", "rook", "zoo", "cat"]
    words_1 = word_list_extractor(wordsim_parsed, 0)
    words_2 = word_list_extractor(wordsim_parsed, 1)
    combined_word_set_1 = set(words_a + words_b)
    combined_word_set_2 = set(words_1 + words_2)
    wv_model = api.load("glove-wiki-gigaword-50")
    gold_scores = [item[2] for item in wordsim_parsed]

    print("\nMETRICS COMPUTED FOR THE PROVIDED 6 WORD PAIRS:\n")

    data_table = {
        "Word A": words_a,
        "Word B": words_b,
        "Path Similarity Scores": path_similarity(words_a, words_b),
        "Wu-Palmer Similarity Scores": wup_similarity(words_a, words_b),
        "word2vec Cosine Similarity Scores": cos_similarity(words_a, words_b, wv_model),
    }

    results = pd.DataFrame(data_table)
    display(results)

    print("\nSpearman correlation metrics:")
    print(
        f"WordNet Path Similarity: {round(((stats.spearmanr(path_similarity(words_a, words_b), human_scores(words_a, words_b, wordsim_parsed))).correlation), 4)}"
    )
    print(
        f"word2vec Cosine Similarity: {round(((stats.spearmanr(cos_similarity(words_a, words_b, wv_model), human_scores(words_a, words_b, wordsim_parsed))).correlation), 4)}"
    )

    print("\nChecking the wordnet coverage:")
    wordnet_coverage(combined_word_set_1)

    print("\nChecking the GloVe coverage:")
    glove_coverage(combined_word_set_1, wv_model)
    print("\n")

    print("METRICS COMPUTED FOR THE 203 WORD PAIRS FROM WORDSIMILARITY-503:\n")

    data_table = {
        "Word A": words_1,
        "Word B": words_2,
        "Path Similarity Scores": path_similarity(words_1, words_2),
        "Wu-Palmer Similarity Scores": wup_similarity(words_1, words_2),
        "word2vec Cosine Similarity Scores": cos_similarity(words_1, words_2, wv_model),
    }

    results = pd.DataFrame(data_table)
    display(results)

    print("\nSpearman correlation metrics:")
    print(
        f"WordNet Path Similarity: {round(((stats.spearmanr(path_similarity(words_1, words_2), gold_scores)).correlation), 4)}"
    )
    print(
        f"word2vec Cosine Similarity: {round(((stats.spearmanr(cos_similarity(words_1, words_2, wv_model), gold_scores)).correlation), 4)}"
    )

    print("\nChecking the wordnet coverage:")
    wordnet_coverage(combined_word_set_2)

    print("\nChecking the GloVe coverage:")
    glove_coverage(combined_word_set_2, wv_model)
    print("\n")


if __name__ == "__main__":
    main()
