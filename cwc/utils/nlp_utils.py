from nltk.tokenize import sent_tokenize


def reformmat_text(text):
    """
    Heuristically handling lower cased text.
    :param text: 
    :return: 
    """
    return sent_tokenize(text)
