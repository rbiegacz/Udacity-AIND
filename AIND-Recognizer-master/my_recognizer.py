import warnings
from asl_data import SinglesData


DEBUG = False


def print_debug(msg):
    """Printing debugging information"""
    if DEBUG:
        print(msg)

def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []

    # T O D O implement the recognizer
    # we want to iterate thru:
    # - all words
    # - all models
    for word, model in models.items():
        top_prob = float("-inf")
        top_word = None
        word_probabilities = {}
        print_debug("Going thru: word {} and model {}".format(word, model))
        for idx in range(test_set.num_items):
            test_sequence, sequence_lengths = test_set.get_item_Xlengths(idx)
            try:
                word_probabilities[word] = model.score(test_sequence, sequence_lengths)
            except:
                word_probabilities[word] = float("-inf")
            if word_probabilities[word] > top_prob:
                top_prob, top_word = word_probabilities[word], word
        probabilities.append(word_probabilities)
        guesses.append(top_word)
    return probabilities, guesses

