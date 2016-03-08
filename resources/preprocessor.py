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

            # cycle through the generator to get the appropriate format
            # for feature extraction
            segments_for_analysis = []
            for tup in self.model_segments:

                # flatten tuples
                tup2 = self.flatten_to_generator(tup)
                segments_for_analysis.append(list(tup2))

            self._extract_feature_dict(segments_for_analysis)

        else:
            _logger.warning("Input not a Morfessor Baseline model, aborting")
            raise ValueError("Input must be a Morfessor Baseline model")

    # ---------------------------- static methods ----------------------------

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
        """Build an appropriate group string for an affix regular expression."""

        # TODO: write affix-removed tokens in such a way that the segmenter picks it up correctly
        if regex.groups == 4:
            out_str = r'\g<4><\g<2>><redup>'
        else:
            out_str = r'\g<1>\g<3><\g<2>>'

        return out_str

    # ---------------------- feature generation methods ----------------------

    def _extract_feature_dict(self, segmentation_list):
        """
        """

        # build dictionary of words and features
        feature_dict = dict()

        for word_tuple in segmentation_list:

            # store word count, segments, and base form
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

            # get hypothesized root
            # TODO: account for morphs of the same length
            # TODO: account for problem prefixes like 'nakaka-'
            morph_list = sorted(segment_list, key=len)
            word_root = morph_list.pop()

            # construct the feature dictionary
            feature_dict[word_base] = dict(count=word_count,
                                           segments=segment_list,
                                           root=word_root)

        # store dictionaries in class variables
        _logger.info("Feature dictionary extracted.")

        self._feature_dict = feature_dict

    def _get_roots(self):

        return {self._feature_dict[word]['root'] for word in self._feature_dict}

    def _get_morphemes(self):

        morph_counter = collections.Counter()

        for word in self._feature_dict:
            morph_counter.update(self._feature_dict[word]['segments'])

        return morph_counter

    # -------------------------- analysis methods --------------------------

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

    def check_affixes(self, affix_list):
        """Cycle through each word and remove the affixes if present.

        :param affix_list: a list of regular expression defining affixes
        """

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
