from collections import Counter

min_count = 1

if __name__ == '__main__':

    def words(name):
        return '{}.words.txt'.format(name)

    counter_word = Counter()
    for n in ['train', 'test', 'valid']:
        with open("*" + words(n)) as f:
            for line in f:
                counter_word.update(line.strip().split())

    vocab_words = {w for w, c in counter_word.items() if c > min_count}

    with open("*/vocab.words.txt", "w") as f:
        for w in sorted(list(vocab_words)):
            f.write('{}\n'.format(w))

    def tags(name):
        return '{}.tags.txt'.format(name)

    vocab_tags = set()
    with open("*/" + tags("train")) as f:
        for line in f:
            vocab_tags.update(line.strip().split())

    with open("*/vocab.tags.txt", "w") as f:
        for t in sorted(list(vocab_tags)):
            f.write('{}\n'.format(t))
