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

import collections
import json
import logging
import os
import pickle
import re

import morfessor

from resources.segmenter import morfessor_main, make_temp_file

_logger = logging.getLogger(__name__)


class AffixFilter(object):
    """An object for filtering specified affixes from a list of words."""

    def __init__(self, word_file, affix_list):

        with open(word_file, 'r') as f:
            self.word_list = f.readlines()

        self.affixes = self._format_affixes(affix_list)
        self.filtered_words = self._remove_affixes_from_words(self.word_list)

    @staticmethod
    def _make_group_pattern(regex):
        """Build an appropriate group string for an affix regular expression."""

        # TODO: write affix-removed tokens in such a way that the segmenter picks it up correctly
        if regex.groups == 4:
            out_str = r'\g<4><\g<2>><redup>'
        else:
            out_str = r'\g<1>\g<3><\g<2>>'

        return out_str

    def _format_affixes(self, affix_list):
        """Reformat input affix patterns into something usable by the class.

        - inputs: a list of regular expression defining affixes
        - outputs: a list of tuples containing a compiled regular expression
                   and a group pattern to preserve a word without that affix
        """

        affix_tuples = list()

        for affix in affix_list:
            regex_affix = re.compile(affix)
            regex_repl = self._make_group_pattern(regex_affix)

            # the regex plus the number of groups in it
            affix_tuples.append((regex_affix, regex_repl))

        return affix_tuples

    def _remove_affixes_from_words(self, word_list):
        """Cycle through each word and remove the affixes if present."""

        filtered_words = list()

        for word in word_list:

            # return the same word unless one of the filters is applied
            word_to_append = word

            for affix_tuple in self.affixes:

                affix_regex, affix_repl = affix_tuple

                if affix_regex.search(word):

                    # the affixes are mutually exclusive, if a filter matches,
                    # the loop breaks and the word is sent to be appended

                    word_to_append = affix_regex.sub(affix_repl, word)
                    break

            filtered_words.append(word_to_append)

        return filtered_words

    def filtered_words(self):
        """Return list of filtered words to users."""

        return self.filtered_words

    def output_text(self, out_file):
        """Write parsed words to a text file."""

        out_list = self.filtered_words

        with open(out_file, 'w') as f:
            f.write(''.join(out_list))


class ModelExaminer(object):
    """An object that tokenizes a Morfessor model."""

    def __init__(self, target):
        """An object that tokenizes Morfessor model text files."""

        # for taking output from morfessor BaselineModel object in generator form
        if isinstance(target, morfessor.BaselineModel):

            # get the segmentation generator objects from the model
            self.model_segments = target.get_segmentations()
            segments_for_analysis = self.flatten_segments(self.model_segments)

            self._extract_feature_dict(segments_for_analysis)

            # set variables for progress tracking
            self._cycle_init = True
            self._cycle_test = False
            self._cycle_final = False

            # set for assignment later
            self.word_changes = None

        else:
            _logger.warning("Input not a Morfessor Baseline model, aborting")
            raise ValueError("Input must be a Morfessor Baseline model")

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

        :param regex: regular expressions for an affix
        :return: a string containing the correct replacement form for the input regex
        """

        # TODO: write affix-removed tokens in such a way that the segmenter picks it up correctly
        if regex.groups == 4:
            out_str = r'\g<4><\g<2>><redup>'
        else:
            out_str = r'\g<1>\g<3><\g<2>>'

        return out_str

    @staticmethod
    def _train_morfessor_model(train_words, dampening='none'):
        """Call the Morfessor Baseline model main on the supplied word list.

        :param train_words: an iterable containing word strings for retraining
        :param dampening: model dampening type in {'none', 'ones', 'log'}
        :return:
        """

        container_file = make_temp_file()
        container_file.write('\n'.join(train_words))

        # input file must always be a list! the call fails otherwise
        model = morfessor_main([container_file], dampening)

        return model

    def flatten_segments(self, model_segments):
        """Flatten init_segments from morfessor baseline model output.

        :param model_segments: output from morfessor.Baseline.get_segmentations()
        :return: a list of flattened segmentation tuples
        """

        segments_out = []

        for tup in model_segments:

            tup2 = self.flatten_to_generator(tup)
            segments_out.append(list(tup2))

        return segments_out

    # ---------------------- feature generation methods ----------------------

    def _extract_feature_dict(self, segmentation_list, run_cycle=1):
        """Extract a nested dictionary of features (init_root, init_segments, count) for each word.

        :type segmentation_list: list of flattened tuples, output from self.flatten_segments
        :type run_cycle: int for which cycle the analysis project is being called
        """

        # build dictionary of words and features
        feature_dict = dict()

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

            # get hypothesized init_root
            # TODO: account for morphs of the same length
            # TODO: account for problem prefixes like 'nakaka-'
            morph_list = sorted(segment_list, key=len)
            word_root = morph_list.pop()

            # for the __init__ cycle
            if run_cycle == 1:

                # construct the initial dictionary
                feature_dict[word_base] = dict(count=word_count,
                                               init_segments=segment_list,
                                               init_root=word_root)

                # store dictionaries in class variables
                _logger.info("Feature dictionary extracted.")
                self._feature_dict = feature_dict

            # second run-through, add new values to _feature_dict
            elif run_cycle == 2:

                self._feature_dict[word_base]['test_segments'] = segment_list
                self._feature_dict[word_base]['test_root'] = word_root

                # set marker for completion
                self._cycle_test = True

                _logger.info("Feature dictionary updated with test values.")

    def _get_init_roots(self):

        roots_list = [self._feature_dict[word]['init_root'] for word in self._feature_dict]

        return collections.Counter(roots_list)

    def _get_morphemes(self):

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
            affix_tuples.append((regex_affix, regex_repl, has_redup, affix_form))

        return affix_tuples

    def filter_affixes(self, affix_list):
        """Cycle through each word and remove the affixes if present.

        :param affix_list: a list of regular expression defining affixes
        """

        # test to ensure this cycle is completed
        if self._cycle_test is False:

            for word in self._feature_dict:

                for affix_tuple in affix_list:

                    affix_regex, affix_repl, has_redup, affix_form = affix_tuple

                    _logger.info("Testing affix: '{}'".format(affix_regex.pattern))

                    if affix_regex.search(word):

                        # the affixes are mutually exclusive, if a filter matches,
                        # the loop breaks and the word is sent to be appended
                        # TODO: make this more generally applicable

                        self._feature_dict[word]['test_infix'] = affix_form
                        self._feature_dict[word]['test_has_redup'] = has_redup
                        self._feature_dict[word]['test_transformed'] = affix_regex.sub(affix_repl, word)

                        break

            _logger.info("Feature dictionary updated with affix testing.")

        else:

            _logger.warning("Method 'filter_affixes' cannot be called a second time.")

    def _get_retrain_words(self):
        """Get right words for retrain and a dict with transformed-to-original mapping.

        :return retrain_words: a list of the original training words, with transformed
                               versions substituted when they exist
        :return original_transformed_map: a dictionary of transformed_word:original_word
                                          pairs, so that the _feature_dict can be repaired
        """

        retrain_words = list()
        original_transformed_map = dict()

        for word in self._feature_dict:

            if self._feature_dict[word].get('test_infix', None):

                transformed = self._feature_dict[word]['test_transformed']
                original_transformed_map[transformed] = word

            else:
                retrain_words.append(word)

        return retrain_words, original_transformed_map

    def _match_cycle_output(self, transformed_original_dict):
        """Remove transformed mappings as values and store results under original base form.

        :type transformed_original_dict: dictionary of transformed_word:original_word pairs
        """

        for word in self._feature_dict:

            # checks to see if key is in _get_retrain_words output
            if word in transformed_original_dict.keys():

                base_form = transformed_original_dict[word]
                transformed_value = self._feature_dict[word]

                # moves the values from the second cycle into the correct _feature_dict entry
                for key in transformed_value:
                    self._feature_dict[base_form][key] = transformed_value[key]

                self._feature_dict.pop(word)

    def retrain_model(self, dampening='none'):
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
        model = self._train_morfessor_model(words_for_retraining, dampening)
        segments = model.get_segmentations()

        # re-runs the __init__ cycle, but updates instead
        segments_for_analysis = self.flatten_segments(segments)
        self._extract_feature_dict(segments_for_analysis, run_cycle=2)

        # repair self._feature_dict and correctly maps new values to right keys
        self._match_cycle_output(transformed_mapping)

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

        init_roots = self._get_init_roots()
        total = sum(init_roots.values())

        for word in self._feature_dict:

            test_root = self._feature_dict[word].get('test_root', None)

            if test_root is not None:

                test_root_count = init_roots[test_root]
                self._feature_dict[word]['test_root_count'] = test_root_count
                self._feature_dict[word]['test_root_per'] = test_root_count / total

        _logger.debug('Counts and percents set for roots after affix filtering.')

    def _assign_final_values(self, threshold=0):
        """Assigns the final versions of roots and segmentations using retrain values.

        This method uses the values assigned in self._get_retrain_values to determine
        which segmentation will be kept in the final model.
        """

        word_changes = list()

        for word in self._feature_dict:

            features = self._feature_dict[word]

            if features['test_root_per'] > threshold:

                new_word_base = features['test_transformed']
                self._feature_dict[word]['final_word_base'] = new_word_base

                word_changes.append((word, new_word_base))

                _logger.debug("CHANGE: '{}' changed to '{}'".format(word, new_word_base))

            else:

                old_word_base = features['word_base']
                self._feature_dict[word]['final_word_base'] = old_word_base

                word_changes.append(old_word_base)

                _logger.debug("NO CHANGE: '{}'".format(word))

        # get counts of changes for log
        changed_words = sum([1 for item in word_changes if isinstance(item, tuple)])
        unchanged_words = sum([1 for item in word_changes if isinstance(item, str)])

        _logger.info("{} tokens changed, {} tokens unchanged".format(changed_words, unchanged_words))

        self.word_changes = word_changes

    def evaluate_retrain(self, threshold, out_file=None):
        """Set the token forms for the final model using thresholds.

        :param threshold: default=0; lower bound on whether transformed form should be
                          used in final model
        :param out_file: if present, writes filtered tokens and their change status to
                         a text file
        """

        self._set_retrain_values()

        self._assign_final_values(threshold)

        if out_file:

            self.write_changed_tokens(out_file)

    # ---------------------------- output methods ----------------------------

    def feature_dictionary(self):
        """Return the feature dictionary for the input file."""

        return self._feature_dict

    def write_feature_dict(self, out_file, output_format):
        """Write feature set to output format (JSON).

        :param out_file: the destination file; do not use file extension
        :param output_format: JSON or pickle
        """

        if output_format.lower() not in {'json', 'pickle'}:
            _logger.error('ERROR: unrecognized output format: {}'.format(output_format))
            raise ValueError("output_format: {'json', 'pickle'}")

        elif output_format.lower() == 'json':

            with open(out_file + '.json', 'w') as f:
                json.dump(self._feature_dict, f)
                # _logger.info('Feature set written to {}.json'.format(out_file))

        elif output_format.lower() == 'pickle':

            with open(out_file + '.pickle', 'w') as f:
                pickle.dump(self._feature_dict, f)
                # _logger.info('Feature set written to {}.pickle'.format(out_file))

        out_name = os.path.basename(out_file)
        out_msg = "Feature set dictionary written to {}".format(out_name + '.' + output_format.lower())
        _logger.info(out_msg)

    def write_changed_tokens(self, out_file):

        with open(out_file, 'w') as f:

            for item in self.word_changes:

                if isinstance(item, tuple):
                    f.write('{}\t\t{}'.format(item[0], item[1]))

                else:
                    f.write(item)

        out_name = os.path.basename(out_file)
        out_msg = "Word changes list written to {}".format(out_name)
        _logger.info(out_msg)
