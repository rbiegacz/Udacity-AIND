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
    for idx in range(test_set.num_items):
        # we will to 'guessing' for every word in test set
        seq, lengths = test_set.get_item_Xlengths(idx)
        best_prob = float("-inf")
        best_word = None
        word_probabilities = {}
        for word, model in models.items():
            print_debug("Examining model {}".format(model))
            print_debug("Calculating probability for word {}".format(word))
            try:
                word_probabilities[word] = model.score(seq, lengths)
            except:
                word_probabilities[word] = float("-inf")
            if word_probabilities[word] > best_prob:
                best_prob, best_word = word_probabilities[word], word
                print_debug("Probability for word {} is {}".format(best_prob, best_word))
        # adding a list of probabilities to the list of probabilities
        probabilities.append(word_probabilities)
        # adding a top word to the guesses list
        guesses.append(best_word)
    return probabilities, guesses
