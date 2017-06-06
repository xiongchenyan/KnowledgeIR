from nltk.tokenize import sent_tokenize


def reformat_text(text):
    """
    Heuristically handling lower cased text.
    :param text: 
    :return: 
    """
    formatted_sents = []

    for sent in split_sentences(text):
        formatted_sents.append(sent.capitalize() + "\n\n")

    return " ".join(formatted_sents)


def split_sentences(text, limit=100):
    for sent in sent_tokenize(text):
        tokens = sent.split()
        if len(tokens) > limit:
            for i in range(0, len(tokens), limit):
                yield " ".join(tokens[i: i + limit])
        else:
            yield sent
