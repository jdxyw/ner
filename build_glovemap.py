import numpy as np

if __name__ == "__main__":
    with open("*/vocab.words.txt") as f:
        word2idx = {line.strip(): idx for idx, line in enumerate(f)}
    vocab_size = len(word2idx)

    embeddings = np.zeros((vocab_size, 300))

    with open("*/glove.840B.300d.txt") as f:
        for line_idx, line in enumerate(f):
            line = line.strip().split()
            if len(line) != 301:
                continue
            word = line[0]
            embedding = line[1:]
            if word in word2idx:
                word_idx = word2idx[word]
                embeddings[word_idx] = np.asarray(embedding)

    np.savez_compressed('glove.npz', embeddings=embeddings)