# src/vectorizer.py

import numpy as np

class BagOfWords:
    def __init__(self):
        self.vocab = {}

    def fit(self, sentences):
        word_set = set()
        for sentence in sentences:
            words = sentence.split()
            word_set.update(words)
        sorted_words = sorted(word_set)
        self.vocab = {word: idx for idx, word in enumerate(sorted_words)}

    def transform(self, sentences):
        """Vectorize a list of sentences."""
        vectors = []
        for sentence in sentences:
            vec = np.zeros(len(self.vocab), dtype=int)
            for word in sentence.split():
                if word in self.vocab:
                    vec[self.vocab[word]] += 1
            vectors.append(vec)
        return np.array(vectors)
