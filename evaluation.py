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

import logging
import os.path
import tempfile

from morfessor.io import MorfessorIO
from morfessor.evaluation import MorfessorEvaluation, FORMAT_STRINGS, WilcoxonSignedRank

from preprocessor import AffixFilter
from segmenter import segment_main

_logger = logging.getLogger(__name__)


class InfixerEvaluation(object):

    def __init__(self, morfessor_model, feature_dict, affix_list):
        """An object for evaluating modified Morfessor Baseline segmenters.

        :param morfessor_model: a trained Morfessor Baseline object
        :param feature_dict: the output dictionary from ModelBuilder object
        :param affix_list:
        """

        # save input
        self._model = morfessor_model
        self._feature_dict = feature_dict

        self._affix_filter = AffixFilter(affix_list)

        # set up morfessor's IO class
        self._io_manager = MorfessorIO()

    def _update_compounds(self, word):
        """Return the appropriate form the word in an annotation file.

        For words in the supplied feature dictionary (i.e., in vocabulary words),
        the final transformed word form is returned if it exists, and otherwise
        the same word is returned. If the word is OOV, it is filtered using the
        supplied list of affixes and returned.
        """

        if word in self._feature_dict:
            return self._feature_dict[word].get('final_word_base', word)

        # TODO: build AffixFilter.filter_word(word)
        else:
            return self._affix_filter.filter_word(word)

    def _read_annotations_file(self, file_name, construction_separator=' ', analysis_sep=','):
        """Convert annotation file to generator.

        This is based off a method of the same name in the MorfessorIO class. It
        was modified so that the compound (for our purposes, the word being
        segmented) would undergo the same filtering as the training set, to ensure
        continuity.

        Each line has the format:
        <compound> <constr1> <constr2>... <constrN>, <constr1>...<constrN>, ...

        Yield tuples (compound, list(analyses)).
        """

        with open(file_name, 'r') as f:
            file_data = f.read().split('\n')
            annotation_list = [line for line in file_data if line != '']

        annotations = {}
        _logger.info(
            "Reading gold standard annotations from '%s'..." % file_name)
        for line in annotation_list:
            # analyses = []
            # print(line)
            compound, analyses_line = line.split(None, 1)

            # apply filtering transformations if needed
            compound_mod = self._update_compounds(compound)

            if compound_mod not in annotations:
                annotations[compound_mod] = []

            if analysis_sep is not None:
                for analysis in analyses_line.split(analysis_sep):
                    analysis = analysis.strip()
                    annotations[compound_mod].append(
                        analysis.strip().split(construction_separator))
            else:
                annotations[compound_mod].append(
                    analyses_line.split(construction_separator))

        _logger.info("Done.")
        return annotations

    def evaluate_model(self, gold_standard_file, wilcoxon=False):

        annotations = self._read_annotations_file(gold_standard_file)
        eval_obj = MorfessorEvaluation(annotations)
        results = eval_obj.evaluate_model(self._model)

        print(results.format(FORMAT_STRINGS['default']))

        if wilcoxon:
            wsr = WilcoxonSignedRank()
            r = wsr.significance_test(results)
            WilcoxonSignedRank.print_table(r)


class InfixerSegmenter(InfixerEvaluation):

    def _process_infile(self, infile):

        with open(infile, 'r') as f:
            data = f.read().split('\n')

        data_filtered = []
        for word in data:
            data_filtered.append(self._update_compounds(word))

        temp_dir = tempfile.TemporaryDirectory()
        container_file = os.path.join(temp_dir.name, 'temp_file.txt')

        with open(container_file, 'w') as f:
            f.write('\n'.join(data_filtered))

        return container_file

    def segment_file(self, infile, outfile):

        infile_new = self._process_infile(infile)

        segment_main(self._model, outfile, infile_new)

        # TODO: why did a bunch of the words disappear in the outfile?
        # TODO: distinguish between in-vocab and OOV in output
