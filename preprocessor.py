#!/usr/bin/env python3

"""The model for the preprocessor:

    - Take a list of affixes as a list
        + Have a defined format for these affixes
    - Remove the affixes and store as a feature
        + Should they be removed from the already morph-parsed list, or from
          the unsegmented words? Probably both.
    - Run morfessor-train on data without affixes
    - Analyze the data tagged with affixes to determine whether it caused the
      segmenter to split them better
"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import collections
import json
import logging
import os
import pickle
import re
import tempfile

import morfessor

# from resources.segmenter import morfessor_main, make_temp_file
from segmenter import morfessor_main

_logger = logging.getLogger(__name__)


class AffixFilter(object):

    def __init__(self, affix_list):

        self.affix_list = affix_list

    @staticmethod
    def make_group_pattern(regex):
        """Build an appropriate group string for an affix regular expression.

        Morfessor's default force-split character is '-', so the rearranged morphemes
        are separated with that in order to ensure correct segmentation.

        :param regex: regular expressions for an affix
        :return: a string containing the correct replacement form for the input regex
        """

        # TODO: write affix-removed tokens in such a way that the segmenter
        # picks it up correctly
        if regex.groups == 4:
            out_str = r'<redup>-<\g<2>>-\g<4>'
        else:
            out_str = r'<\g<2>>-\g<1>\g<3>'

        return out_str

    def _format_affixes(self, affix_list):
        """Reformat input affix patterns into something usable by the class.

        :param affix_list: a list of regular expression defining affixes
        :return: a list of tuples containing a compiled regular expression
                 and a group pattern to preserve a word without that affix
        """

        # TODO: make this method applicable to more languages than Tagalog

        affix_tuples = list()

        for affix in affix_list:

            # compiled regular expression
            regex_affix = re.compile(affix)

            # replacement form, using customizable method
            regex_repl = self.make_group_pattern(regex_affix)

            # whether reduplication tag exists
            has_redup = True if re.search('redup', regex_affix.pattern) else False

            # orthographic form of the infix/affix
            affix_form = re.search('\W*?(\w+)\W*', regex_affix.pattern).group(1)

            # return all as a 4-tuple
            affix_tuples.append(
                (regex_affix, regex_repl, has_redup, affix_form))

        return affix_tuples

    def filter_affixes(self, affixes):
        """Cycle through each word and remove the affixes if present.

        :param affixes: a list of regular expression defining affixes
        """

        self.affix_list = affixes

        # test to ensure this cycle is completed
        if self._cycle_test is False:

            affixes_formatted = self._format_affixes(affixes)

            for word in self._feature_dict:

                for affix_tuple in affixes_formatted:

                    # unpack return from _format_affixes
                    affix_regex, affix_repl, has_redup, affix_form = affix_tuple

                    _logger.debug("Testing affix '{}' on word '{}'".format(
                        affix_regex.pattern, word))

                    if affix_regex.search(word):

                        # the affixes are mutually exclusive, if a filter matches,
                        # the loop breaks and the word is sent to be appended
                        # TODO: make this more generally applicable

                        self._feature_dict[word]['test_infix'] = affix_form
                        self._feature_dict[word]['test_has_redup'] = has_redup
                        self._feature_dict[word][
                            'test_transformed'] = affix_regex.sub(affix_repl, word)

                        break

            _logger.info("Feature dictionary updated with affix testing.")

        else:

            _logger.warning(
                "Method 'filter_affixes' cannot be called a second time.")


class ModelBuilder(object):
    """An object that pre-processes data for a Morfessor model."""

    def __init__(self, target, is_json_file=False):
        """An object that pre-processes data for a Morfessor model."""

        # set variables for progress tracking
        self._cycle_init = True
        self._cycle_test = False
        self._cycle_final = False

        # initialize variables for later assignment
        self.affix_list = []
        self.word_changes = []
        self.final_model = None

        # builds initial feature dict from Morfessor Baseline object
        if isinstance(target, morfessor.BaselineModel):

            # get the segmentation generator objects from the model
            model_segments = target.get_segmentations()
            segments_for_analysis = self.flatten_segments(model_segments)

            # extract the initial feature dictionary
            self._extract_feature_dict(segments_for_analysis)

        # builds initial feature dict from JSON file
        elif is_json_file is True:

            # sets self._feature_dict from json file
            self.load_init_json(target)

        else:
            _logger.warning("Input not valid, aborting")
            raise ValueError("Input must be a Morfessor Baseline model or file")

    # ---------------------------- helper methods ----------------------------

    @staticmethod
    def flatten_to_generator(iterable):
        """Return a flattened generator for an iterable of mixed iterables and non-iterables.

        :param iterable: an iterable with any combination of iterable and non-iterable components
        """

        for item in iterable:
            if isinstance(item, list) or isinstance(item, tuple):
                for sub_item in item:
                    yield sub_item
            else:
                yield item

    @staticmethod
    def make_group_pattern(regex):
        """Build an appropriate group string for an affix regular expression.

        Morfessor's default force-split character is '-', so the rearranged morphemes
        are separated with that in order to ensure correct segmentation.

        :param regex: regular expressions for an affix
        :return: a string containing the correct replacement form for the input regex
        """

        # TODO: write affix-removed tokens in such a way that the segmenter
        # picks it up correctly
        if regex.groups == 4:
            out_str = r'<redup>-<\g<2>>-\g<4>'
        else:
            out_str = r'<\g<2>>-\g<1>\g<3>'

        return out_str

    @staticmethod
    def _train_morfessor_model(train_words, dampening='none', cycle='test', save_file=None):
        """Call the Morfessor Baseline model main on the supplied word list.

        :param train_words: an iterable containing word strings for retraining
        :param dampening: model dampening type in {'none', 'ones', 'log'}
        :return:
        """

        # TODO: ensure make_temp_file works when this object is imported into a
        # script
        temp_dir = tempfile.TemporaryDirectory()
        container_file = os.path.join(temp_dir.name, 'tempfile.txt')

        with open(container_file, 'w') as f:
            f.write('\n'.join(train_words))

        # input file must always be a list! the call fails otherwise
        # TODO: ensure morfessor_main works when this object is imported into a
        # script
        model = morfessor_main([container_file], dampening, cycle, save_file)

        return model

    def flatten_segments(self, model_segments):
        """Flatten init_segments from morfessor baseline model output.

        :param model_segments: output from morfessor.Baseline.get_segmentations()
        :return: a list of flattened segmentation tuples
        """

        segments_out = []

        for tup in model_segments:

            tup_flat = self.flatten_to_generator(tup)
            segments_out.append(list(tup_flat))

        return segments_out

    # ---------------------- feature generation methods ----------------------

    def _extract_feature_dict(self, segmentation_list, run_cycle=1):
        """Extract a nested dictionary of features (init_root, init_segments, count) for each word.

        :type segmentation_list: list of flattened tuples, output from self.flatten_segments
        :type run_cycle: int for which cycle the analysis project is being called
        """

        # set a default dict with default arg as an empty dictionary
        feature_dict = collections.defaultdict(dict)

        for word_tuple in segmentation_list:

            # store word count, init_segments, and base form
            word_count, *segment_list = word_tuple
            word_base = re.sub(r'\+', '', ''.join(segment_list))

            # append morpheme boundary markers (+)
            if len(segment_list) > 1:
                temp_seg = []

                for segment in segment_list[1:-1]:

                    temp_s = '+' + segment + '+'
                    temp_seg.append(temp_s)

                temp_seg.insert(0, segment_list[0] + '+')
                temp_seg.append('+' + segment_list[-1])
                segment_list = temp_seg

            # get hypothesized init_root, remove its '+'
            # TODO: account for morphs of the same length
            # TODO: account for problem prefixes like 'nakaka-'
            morph_list = sorted(segment_list, key=len)
            word_root = morph_list.pop().replace('+', '')

            # for the __init__ cycle
            if run_cycle == 1:

                # construct the initial dictionary
                feature_dict[word_base] = dict(count=word_count,
                                               init_word_base=word_base,
                                               init_segments=segment_list,
                                               init_root=word_root)

                # store dictionaries in class variables
                self._feature_dict = feature_dict

            # second run-through, add new values to _feature_dict
            elif run_cycle == 2:

                self._feature_dict[word_base]['test_segments'] = segment_list
                self._feature_dict[word_base]['test_root'] = word_root

                # set marker for completion
                self._cycle_test = True

            elif run_cycle == 3:

                self._feature_dict[word_base]['final_segments'] = segment_list
                self._feature_dict[word_base]['final_root'] = word_root

        if run_cycle == 1:
            _logger.info("Feature dictionary extracted.")
        elif run_cycle == 2:
            _logger.info("Feature dictionary updated with test values.")
        elif run_cycle == 3:
            _logger.info("Feature dictionary updated with final values.")

    def _get_init_roots(self):
        """Get Counter for all values of init_root from the feature dictionary."""

        roots_list = [self._feature_dict[word]['init_root']
                      for word in self._feature_dict]

        return collections.Counter(roots_list)

    def _get_morphemes(self):
        """Get Counter for all morphemes in the init_segments value of the feature dictionary."""

        morph_counter = collections.Counter()

        for word in self._feature_dict:
            morph_counter.update(self._feature_dict[word]['init_segments'])

        return morph_counter

    # ---------------------- testing and retraining --------------------------

    def _format_affixes(self, affix_list):
        """Reformat input affix patterns into something usable by the class.

        :param affix_list: a list of regular expression defining affixes
        :return: a list of tuples containing a compiled regular expression
                 and a group pattern to preserve a word without that affix
        """

        # TODO: make this method applicable to more languages than Tagalog

        affix_tuples = list()

        for affix in affix_list:

            # compiled regular expression
            regex_affix = re.compile(affix)

            # replacement form, using customizable method
            regex_repl = self.make_group_pattern(regex_affix)

            # whether reduplication tag exists
            has_redup = True if re.search('redup', regex_affix.pattern) else False

            # orthographic form of the infix/affix
            affix_form = re.search('\W*?(\w+)\W*', regex_affix.pattern).group(1)

            # return all as a 4-tuple
            affix_tuples.append(
                (regex_affix, regex_repl, has_redup, affix_form))

        return affix_tuples

    def filter_affixes(self, affixes):
        """Cycle through each word and remove the affixes if present.

        :param affixes: a list of regular expression defining affixes
        """

        self.affix_list = affixes

        # test to ensure this cycle is completed
        if self._cycle_test is False:

            affixes_formatted = self._format_affixes(affixes)

            for word in self._feature_dict:

                for affix_tuple in affixes_formatted:

                    # unpack return from _format_affixes
                    affix_regex, affix_repl, has_redup, affix_form = affix_tuple

                    _logger.debug("Testing affix '{}' on word '{}'".format(
                        affix_regex.pattern, word))

                    if affix_regex.search(word):

                        # the affixes are mutually exclusive, if a filter matches,
                        # the loop breaks and the word is sent to be appended
                        # TODO: make this more generally applicable

                        self._feature_dict[word]['test_infix'] = affix_form
                        self._feature_dict[word]['test_has_redup'] = has_redup
                        self._feature_dict[word][
                            'test_transformed'] = affix_regex.sub(affix_repl, word)

                        break

            _logger.info("Feature dictionary updated with affix testing.")

        else:

            _logger.warning(
                "Method 'filter_affixes' cannot be called a second time.")

    def _get_retrain_words(self):
        """Get right words for retrain and a dict with transformed-to-original mapping.

        :return retrain_words: a list of the original training words, with transformed
                               versions substituted when they exist
        :return original_transformed_map: a dictionary of transformed_word:original_word
                                          pairs, so that the _feature_dict can be repaired
        """

        retrain_words = list()
        original_transformed_map = dict()

        # determine lookup based on phase so this method can be reused
        if self._cycle_final is True:
            lookup_feature = 'final_word_base'
        else:
            lookup_feature = 'test_transformed'

        for word in self._feature_dict:

            # if transformed exists, append it to retrain_words and add a dict
            # entry
            if self._feature_dict[word].get(lookup_feature, None):

                transformed = self._feature_dict[word][lookup_feature]
                original_transformed_map[transformed] = word
                retrain_words.append(transformed)

            else:
                retrain_words.append(word)

        return retrain_words, original_transformed_map

    def _rebuild_original_wordforms(self, transformed_original_dict):
        """Remove transformed mappings as values and store results under original base form.

        :param transformed_original_dict: dictionary of {transformed_word: original_word} pairs
        """

        counter = 0
        if self._cycle_final is True:
            feature_keys = ['final_root', 'final_segments']
            cycle_name = 'FINAL CYCLE'
        else:
            # TODO: insert feature keys from third part of __init__ recall
            feature_keys = ['test_infix', 'test_has_redup',
                            'test_transformed', 'test_segments', 'test_root']
            cycle_name = 'TEST CYCLE'

        # temporary dictionary to prevent RuntimeErrors due to dictionary
        # changing size
        feature_dict_copy = self._feature_dict.copy()

        for word in feature_dict_copy:

            # checks to see if key is in _get_retrain_words output
            if word in transformed_original_dict.keys():

                base_form = transformed_original_dict[word]
                transformed_values = self._feature_dict.get(word)

                # moves the values from the second cycle into the correct
                # _feature_dict entry
                for val in feature_keys:

                    # prevent overwriting extant values with None
                    dict_val = transformed_values.get(val, None)

                    if dict_val is not None:
                        self._feature_dict[base_form][val] = dict_val

                # remove transformed word
                self._feature_dict.pop(word)

                counter += 1

        _logger.info('{}: {} word forms rebuilt in the feature dictionary.'.format(cycle_name, counter))

    def build_test_model(self, dampening='none', cycle='test', save_file=None):
        """Run the Morfessor Baseline model on the transformed data.

        This method creates a second model, modified from the first, where the
        transformed words are substituted in place of their original forms.
        After the model is built, it is run through feature extraction
        again, only this time the values found are saved as updates in the main
        feature dictionary (self._feature_dict). The model is then repaired
        by using a correspondence dictionary between the transformed forms
        and the original forms.

        :param dampening: model dampening type in {'none', 'ones', 'log'}
        """

        # get training word set and key for remapping after the cycle
        words_for_retraining, transformed_mapping = self._get_retrain_words()

        # train model and get segmentations
        _logger.info("TEST CYCLE: training Morfessor Baseline model")
        model = self._train_morfessor_model(words_for_retraining, dampening, cycle, save_file)
        segments = model.get_segmentations()

        # re-runs the __init__ cycle, but updates with test values instead
        segments_for_analysis = self.flatten_segments(segments)
        self._extract_feature_dict(segments_for_analysis, run_cycle=2)

        # repair self._feature_dict and correctly maps new values to right keys
        self._rebuild_original_wordforms(transformed_mapping)

        _logger.info("Test model built.")

    # -------------------------- evaluate retraining -------------------------

    def _set_retrain_values(self):
        """Set counts and probabilities for each root affected by affix filter.

        This method checks to see whether the hypothesized root from transformed
        version of the word base can be found in the original data set. The
        assumption is that if a root from a transformed word is found in the original
        data set, then there is a higher likelihood that the transformation was
        a good idea.

        The "percent" value is a mostly arbitrary stand-in for some better method of
        evaluating relative likelihood. The more frequent a root is found in the
        original data set, the more likely it is that it is not a mistaken form.
        """

        init_roots = self._get_init_roots()     # returns a Counter object
        roots_sum = sum(init_roots.values())    # for root percentages

        for word in self._feature_dict:

            test_root = self._feature_dict[word].get('test_root', None)
            test_transformed = self._feature_dict[
                word].get('test_transformed', None)

            # only words with a found affix have a test_transformed feature
            if test_transformed is not None:

                test_root_count = init_roots.get(test_root, 0)
                self._feature_dict[word]['test_root_count'] = test_root_count
                self._feature_dict[word]['test_root_per'] = test_root_count / roots_sum

        _logger.debug(
            'Counts and percents set for roots after training with transformed forms.')

    def _assign_final_values(self, threshold=0):
        """Assigns the final versions of roots and segmentations using retrain values.

        This method uses the values assigned in self._get_retrain_values to determine
        which segmentation will be kept in the final model.

        :param: threshold: default=0; lower bound on whether transformed form should be used in final model
        """

        # TODO: consider putting in a feature that cancels any alteration

        changed_words = list()

        for word in self._feature_dict:

            _logger.debug("Processing word: '{}'".format(word))
            features = self._feature_dict[word]

            # TODO: reduce redundancy in this nested if
            if features.get('test_root_per', None) is not None:

                if features['test_root_per'] > threshold:

                    new_word_base = features['test_transformed']
                    self._feature_dict[word]['final_word_base'] = new_word_base

                    changed_words.append((word, new_word_base))

                    _logger.debug("TRAINING WORD CHANGED: '{}' changed to '{}'".format(
                        word, new_word_base))

                else:
                    _logger.debug("TRAINING WORD UNCHANGED: '{}'".format(word))

            else:
                _logger.debug("NO CHANGE NEEDED: '{}'".format(word))

        unchanged_words = len(list(self._feature_dict)) - len(changed_words)
        _logger.info("{} tokens changed, {} tokens unchanged".format(
            len(changed_words), unchanged_words))

        self.word_changes = changed_words

    def build_final_model(self, dampening='none', threshold=0, cycle='final', save_file=None):
        """Build the final model using the transformed and tested word list.

        :param threshold: default=0; lower bound on whether transformed form should be used in final model
        :param dampening: model dampening type in {'none', 'ones', 'log'}
        """

        # set tracker variable
        self._cycle_final = True

        # sets up test_root counts and percents
        self._set_retrain_values()

        # determines final training values based on threshold
        self._assign_final_values(threshold)

        # get training word set and key for remapping after the cycle
        final_word_list, transformed_mapping = self._get_retrain_words()

        # train model and get segmentations
        _logger.info("FINAL CYCLE: training Morfessor Baseline model")
        self.final_model = self._train_morfessor_model(final_word_list, dampening, cycle, save_file)
        segments = self.final_model.get_segmentations()

        # re-runs the __init__ cycle, but updates with final values instead
        segments_for_analysis = self.flatten_segments(segments)
        self._extract_feature_dict(segments_for_analysis, run_cycle=3)

        # repair self._feature_dict and correctly maps new values to right keys
        self._rebuild_original_wordforms(transformed_mapping)

    # ---------------------------- loading methods ---------------------------

    @staticmethod
    def _load_json_dict(in_file):
        """Build a defaultdict from a JSON input file."""

        with open(in_file, 'r') as f:
            json_dict = json.load(f)

        json_defdict = collections.defaultdict(dict, json_dict)

        _logger.info("Feature dictionary loaded from '{}'".format(in_file))

        return json_defdict

    def load_init_json(self, in_file):

        self._feature_dict = self._load_json_dict(in_file)

        self._cycle_init = True
        self._cycle_test = False
        self._cycle_final = False

    def load_test_json(self, in_file):

        self._feature_dict = self._load_json_dict(in_file)

        self._cycle_init = True
        self._cycle_test = True
        self._cycle_final = False

    def load_final_json(self, in_file):

        self._feature_dict = self._load_json_dict(in_file)

        self._cycle_init = True
        self._cycle_test = True
        self._cycle_final = True

    # ---------------------------- output methods ----------------------------

    def feature_dictionary(self):
        """Return the feature dictionary for the input file."""

        return self._feature_dict

    def get_final_model(self):

        return self.final_model

    def write_feature_dict(self, out_file, output_format):
        """Write feature set to output format (JSON).

        :param out_file: the destination file; do not use file extension
        :param output_format: JSON or pickle
        """

        if output_format.lower() not in {'json', 'pickle'}:
            _logger.error(
                'ERROR: unrecognized output format: {}'.format(output_format))
            raise ValueError("output_format: {'json', 'pickle'}")

        elif output_format.lower() == 'json':

            with open(out_file + '.json', 'w') as f:
                json.dump(self._feature_dict, f)

        elif output_format.lower() == 'pickle':

            with open(out_file + '.pickle', 'w') as f:
                pickle.dump(self._feature_dict, f)

        out_name = os.path.basename(out_file)
        out_msg = "Feature set dictionary written to {}".format(
            out_name + '.' + output_format.lower())
        _logger.info(out_msg)

    def write_changed_tokens(self, out_file):
        """Write a text file of the final word set, with changes marked.

        :param out_file: the text file to be written to
        """

        with open(out_file, 'w') as f:
            for item in self.word_changes:
                f.write('{}\t{}\n'.format(item[0], item[1]))

        out_name = os.path.basename(out_file)
        _logger.info("Word changes list written to {}".format(out_name))
