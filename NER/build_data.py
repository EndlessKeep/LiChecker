from ner_model.config import Config
from ner_model.data_utils import CoNLLDataset, get_vocabs, write_vocab, load_vocab


def main():
    """Procedure to build data

    This script iterates over the whole dataset (train, dev, and test) and extracts
    the vocabularies in terms of words and tags. Since we are using BERT, we no longer
    need to handle GloVe vectors or character-level vocabularies.

    Args:
        config: (instance of Config) has attributes like hyper-parameters...
    """
    # Load configuration
    config = Config(load=False)

    # Generators
    print("Training data path:", config.filename_train)
    print("Development data path:", config.filename_dev)
    print("Test data path:", config.filename_test)

    # Load datasets
    dev = CoNLLDataset(config.filename_dev)
    test = CoNLLDataset(config.filename_test)
    train = CoNLLDataset(config.filename_train)

    print("Number of sentences in dev set:", len(dev))
    print("Number of sentences in test set:", len(test))
    print("Number of sentences in train set:", len(train))

    # Build Word and Tag vocab
    vocab_words, vocab_tags = get_vocabs([train, dev, test])

    # Save vocab
    write_vocab(vocab_words, config.filename_words)
    write_vocab(vocab_tags, config.filename_tags)

    print("Vocabulary files created successfully.")


if __name__ == "__main__":
    main()