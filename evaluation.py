"""
What does evaluation involve?
    1. Check the input word against the dictionary
       1. if match, return final_word_base, final_root, and
          final_segments
       2. else return OOV mark
    2. Assemble a list of appropriate forms for morfessor evaluation
       1. for words in dictionary, append final_word_base or
          init_word_base (in that order)
       2. for OOV words, run through same affix filter
          as model and return result
    3. Give the morfessor evaluation tool a list of "gold standard"
       segmentations and evaluate the words
"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import morfessor


class ModelEvaluator(object):

    def __init__(self, morfessor_model, feature_dict, affix_list, gold_standard_file):
        """An object for evaluating modified morfessor.Baseline segmenters.
        """

        self._model = morfessor_model
        self._feature_dict = feature_dict
        self._affix_list = affix_list
        self._gstandard = gold_standard_file

    def _build_word_list(self):

        pass

    def _call_evaluation(self):

        io = morfessor.MorfessorIO()

        goldstd_data = io.read_annotations_file(self._gstandard)
        eval_obj = morfessor.MorfessorEvaluation(goldstd_data)
        results = eval_obj.evaluate_model(self._model)

        wsr = morfessor.WilcoxonSignedRank()
        r = wsr.significance_test(results)
        morfessor.WilcoxonSignedRank.print_table(r)
