import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


DEBUG = False


def print_debug(msg):
    """Printing debugging information"""
    if DEBUG:
        print(msg)


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # T O D O implement model selection based on BIC scores
        # BIC score formula = - 2 * log(L) + p*log(N)
        resulting_score = float("inf")
        resulting_model = None
        for n_components in range(self.min_n_components, self.max_n_components + 1):
            try:
                model = self.base_model(n_components)
                logL = model.score(self.X, self.lengths)
                n_features = self.X.shape[1]
                n_params = n_components * (n_components - 1) + 2 * n_features * n_components
                logN = np.log(self.X.shape[0])
                bic = -2 * logL + n_params * logN
                if bic < resulting_score:
                    resulting_score, resulting_model = bic, model
            except:
                continue
        if resulting_model is None:
            resulting_model = self.base_model(self.n_constant)
        return resulting_model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    models, values = {}, {}

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # DIC score formula = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))

        # T O D O implement model selection based on DIC scores

        resulting_score = float("-inf")
        resulting_model = None
        # implement model selection based on DIC scores
        if len(SelectorDIC.models) == 0:
            for n_components in range(self.min_n_components, self.max_n_components + 1):
                n_components_models, n_components_ml = {}, {}
                model = GaussianHMM(n_components=n_components, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False)
                for word in self.words.keys():
                    X, lengths = self.hwords[word]
                    try:
                        model = model.fit(X, lengths)
                        logL = model.score(X, lengths)
                        n_components_models[word] = model
                        n_components_ml[word] = logL
                    except:
                        continue
                SelectorDIC.models[n_components] = n_components_models
                SelectorDIC.values[n_components] = n_components_ml

        for n_components in range(self.min_n_components, self.max_n_components + 1):
            models, ml = SelectorDIC.models[n_components], SelectorDIC.values[n_components]
            if self.this_word not in ml:
                continue
            avg = np.mean([ml[word] for word in ml.keys() if word != self.this_word])
            dic = ml[self.this_word] - avg
            if dic > resulting_score:
                resulting_score, resulting_model = dic, models[self.this_word]
        return resulting_model if resulting_model is not None else self.base_model(self.n_constant)


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # T O D O implement model selection using CV
        resulting_score = float("-inf")
        resulting_model = None
        n_splits = 2

        for n_components in range(self.min_n_components, self.max_n_components + 1):
            # skipping situations when number of splits is greater than number of samples
            print_debug("n_components {}, n_splits {}".format(n_components, n_splits))
            if len(self.sequences) < n_splits:
                break

            # list to store scores, model object and logL score
            # score variable it will be a list containing the log likelihood for a model
            # we will calculate the average of the results stored to calculate a score for a given model
            scores, log_likelihood = [], None
            # GaussianHMM - model is going to store an object of Gausian HMM initialized appropriately
            model = None

            # Description of KFold
            # http: // scikit - learn.org / stable / modules / generated / sklearn.model_selection.KFold.html
            split_method = KFold(random_state=self.random_state, n_splits=n_splits)
            for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                x_train, lengths_train = combine_sequences(cv_train_idx, self.sequences)
                x_test, lengths_test = combine_sequences(cv_test_idx, self.sequences)
                model = GaussianHMM(n_components=n_components, covariance_type="diag", n_iter=1000, random_state=self.random_state, verbose=False)
                try:
                    model = model.fit(x_train, lengths_train)
                    log_likelihood = model.score(x_test, lengths_test)
                    scores.append(log_likelihood)
                except Exception as exception_msg:
                    print_debug("Unfortunately, there was exception for components {}".format(n_components))
                    print_debug("Exception message: {}".format(exception_msg))
                    break
            avg = np.average(scores) if len(scores) > 0 else float("-inf")
            if avg > resulting_score:
                print_debug("Better model found: {} with a score {}".format(model, avg))
                resulting_score, resulting_model = avg, model
        if resulting_model:
            print_debug("The best model found: {} with a score {}".format(resulting_model, resulting_score))
        return resulting_model
