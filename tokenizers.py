#!/usr/bin/env python3

"""A collection of tokenizer classes for different tasks in the project."""

import collections
import glob
import io
import json
import logging
from operator import itemgetter
import os.path
import pickle
import re
from types import GeneratorType

from nltk.corpus import names
from nltk.tokenize import wordpunct_tokenize

_logger = logging.getLogger(__name__)


class GeneralTokenizer(object):
    """A superclass for deriving specialized tokenizers."""

    def __init__(self, target):
        """A basic class for tokenizing text files."""

        if os.path.isdir(target):

            self.target = target        # stored for evaluation in other methods
            self.dir = target
            self._tokens = self._get_dir_tokens(self.dir)

        elif os.path.isfile(target):

            self.target = target        # stored for evaluation in other methods
            # self.dir = os.path.split(os.path.abspath(target))[0]
            # self.filename = target
            target_path = os.path.abspath(target)
            self.dir, self.filename = os.path.split(target_path)
            self._tokens = self._get_file_tokens(self.filename)

        else:
            raise ValueError("File or directory not readable.")

        self._data_len = len(self._tokens)

    def __len__(self):
        """Return the current number of tokens stored in the object."""

        return self._data_len

    # methods for extracting tokens from files and/or directories of files

    def _extract_tokens(self, file_text):
        """Extract tokens from a file and return a Counter dictionary.

        This method is designed specifically so that it can be overridden
        easily while maintaining _get_file_tokens and _get_dir_tokens.
        """

        token_dict = collections.Counter()

        # does a simple word and punctuation tokenization on the text
        tokens = wordpunct_tokenize(file_text)

        for token in tokens:
            token_dict[token] += 1

        return token_dict

    def _get_file_tokens(self, filename):
        """Get all unique tokens from a text file.

        This method needs to have a return instead of direct assignment to
        self._tokens so that it can be called directly or as a subroutine
        of _get_dir_tokens, as needed.
        """

        with open(filename, 'r') as infile:
            infile_text = infile.read()

        # extract and add unique tokens to out_set
        token_dict = self._extract_tokens(infile_text)

        # logging
        filename = os.path.basename(infile.name)
        token_count = len(token_dict)

        # log info for each file in a dir only if logging.DEBUG
        if os.path.isdir(self.target):
            _logger.debug("{} token types in {}".format(token_count, filename))
        else:
            _logger.info("{} token types in {}".format(token_count, filename))

        return token_dict

    def _get_dir_tokens(self, directory):
        """Get all unique tokens from a directory of text files.

        This method needs to have a return instead of direct assignment to
        self._tokens so that _get_file_tokens can be called directly or as
        a subroutine, as needed.
        """

        tokens_all = collections.Counter()

        files = glob.glob(directory + "*")

        for f in files:
            tokens_file = self._get_file_tokens(f)
            tokens_all.update(tokens_file)

        n_out = len(list(tokens_all.keys()))

        # logging
        _logger.info('{} token types found in {} files'.format(n_out, len(files)))

        return tokens_all

    # methods for removing various types of unwanted data

    def clean_tokens(self, rm_dups=True, rm_names=True, rm_nonwords=True,
                     rm_nonlatin=True, rm_uppercase=True):
        """Call methods for removing various types of unwanted data in batch fashion.

        :param rm_dups: remove duplicate upper-case tokens, preserving case and counts
        :param rm_names: remove names present in NLTK's names corpus
        :param rm_nonwords: remove digits, non-alphanumeric tokens, and all-caps words
        :param rm_nonlatin: remove non-Latin extended unicode characters
        :param rm_uppercase: remove upper-case words
        """

        if rm_dups:
            self._remove_duplicates()

        if rm_names:
            self._remove_names()

        if rm_nonwords:
            self._remove_non_words()

        if rm_nonlatin:
            self._remove_non_latin()

        if rm_uppercase:
            self._remove_uppercase()

    def _remove_duplicates(self):
        """Remove duplicate upper-case tokens, preserving case and counts."""

        dupes = {key: count for (key, count) in self._tokens.items()
                 if key in self._tokens and key.lower() in self._tokens}

        no_dupes = {key: count for (key, count) in self._tokens.items()
                    if key not in dupes}

        # use Counter.update() method to preserve counts for duplicates
        dupes_lower = collections.Counter()

        for (key, count) in self._tokens.items():
            dupes_lower[key.lower()] = count

        no_dupes.update(dupes_lower)

        # logging
        _logger.info('{} duplicate tokens removed'.format(len(dupes)))

        self._tokens = collections.Counter(no_dupes)

    def _remove_names(self):
        """Remove names present in NLTK's names corpus."""

        name_set = set(names.words())

        no_names = {key: count for (key, count) in self._tokens.items()
                    if key not in name_set}

        # logging
        num_removed = len(self._tokens) - len(no_names)
        _logger.info(('{} name tokens removed').format(num_removed))

        self._tokens = collections.Counter(no_names)

    def _remove_non_words(self):
        """Remove digits, non-alphanumeric tokens, and all-caps words."""

        # pre-filter count of self.tokens for later comparison and logging
        base_len = len(self._tokens)

        regex = re.compile(r'(^\w*\d+\w*$|^\W*$|^[A-Z]*$|^.*_.*$)')

        matches_out = {key: count for (key, count) in self._tokens.items()
                       if regex.search(key) is None}

        # logging
        num_removed = len(self._tokens) - base_len
        _logger.info('{} non-word tokens removed'.format(num_removed))

        self._tokens = collections.Counter(matches_out)

    def _remove_non_latin(self):
        """Remove non-Latin extended unicode characters."""

        regex = re.compile(r'[^\x00-\x7F\x80-\xFF\u0100-\u017F\u0180-\u024F\u1E00-\u1EFF]')

        matches_out = {key: count for (key, count) in self._tokens.items()
                       if regex.search(key) is None}

        _logger.info('Non-latin tokens removed')

        self._tokens = collections.Counter(matches_out)

    def _remove_uppercase(self):
        """Remove upper-case words. Only run AFTER remove_duplicates."""

        tokens_caps = {key for key in self._tokens.keys()
                       if key[0].isupper()}

        no_caps = {key: count for (key, count) in self._tokens.items()
                   if key not in tokens_caps}

        _logger.info('Uppercase tokens removed')

        self._tokens = collections.Counter(no_caps)

    def filter_tokens(self, filter_source, length=10000):
        """Filter the tokens using a specified outside word list."""

        with open(filter_source, 'r') as f:
            data = f.read()

        # structured for a file of entries in the form 'word ###\n'
        regex = re.compile(r'(\w*) \d*')
        filter_set = set(regex.findall(data)[:length])

        tokens_filtered = {key: count for (key, count) in self._tokens.items()
                           if key.lower() not in filter_set}

        _logger.info('Tokens filtered using {}'.format(filter_source))

        self._tokens = collections.Counter(tokens_filtered)

    # methods for writing the token sets to various types of files in various
    # configurations

    def get_tokens(self, output_type='items'):

        # create the correct output for type, or print error to screen
        if output_type == 'items':
            out_list = self._tokens.keys()

        elif output_type == 'elements':
            out_list = self._tokens.elements()

        elif output_type == 'counts':
            out_list = ['{}\t{}'.format(key, count) for (key, count)
                        in self._tokens.items()]
        else:
            err_msg = "output_type: 'items' (default), 'elements', 'counts'"
            raise ValueError(err_msg)

        return out_list

    def output_text(self, outfile, output_type='items'):
        """Output tokens to a text file."""

        # create the correct output for type, or print error to screen
        if output_type == 'items':
            out_list = self._tokens.keys()

        elif output_type == 'elements':
            out_list = self._tokens.elements()

        elif output_type == 'counts':
            out_list = ['{}\t{}'.format(key, count) for (key, count)
                        in self._tokens.items()]
        else:
            err_msg = "output_type: 'items' (default), 'elements', 'counts'"
            raise ValueError(err_msg)

        with open(outfile, 'w') as out_file:
            out_file.write('\n'.join(out_list))

        out_msg = "{} tokens written to {}".format(len(self._tokens), outfile)

        _logger.info(out_msg)
        print(out_msg)

    def output_file_buffer(self, output_type='items'):
        """Output tokens to a text file."""

        # create the correct output for type, or print error to screen
        if output_type == 'items':
            out_list = self._tokens.keys()

        elif output_type == 'elements':
            out_list = self._tokens.elements()

        elif output_type == 'counts':
            out_list = ['{}\t{}'.format(key, count) for (key, count)
                        in self._tokens.items()]
        else:
            err_msg = "output_type: 'items' (default), 'elements', 'counts'"
            raise ValueError(err_msg)

        buffer = io.StringIO()               # create a fake buffer
        buffer.write('\n'.join(out_list))    # write list to it
        buffer.seek(0)                       # 'rewind' the buffer

        return buffer

        # out_msg = "{} tokens written to {}".format(len(self._tokens), outfile)
        #
        # _logger.info(out_msg)
        # print(out_msg)


class WikipediaTokenizer(GeneralTokenizer):
    def __init__(self, target):
        """An object that tokenizes and filters Wikipedia dump text files."""

        GeneralTokenizer.__init__(self, target)

    # methods for extracting tokens from files and/or directories of files

    def _extract_tokens(self, file_text):
        """Extract tokens from a file and return a Counter dictionary."""

        token_dict = collections.Counter()

        # matches and removes beginning and end tags
        regex = re.compile(r'(<doc id.*>|<\/doc>)')
        data = regex.sub('', file_text)

        tokens = wordpunct_tokenize(data)

        for token in tokens:
            token_dict[token] += 1

        return token_dict


class BabelTokenizer(GeneralTokenizer):
    """A class for tokenizing Babel files, based on WikipediaTokenizer."""

    def __init__(self, target):
        """An object that tokenizes and filters Wikipedia dump text files."""

        GeneralTokenizer.__init__(self, target)

    def _extract_tokens(self, file_text):
        """Extract tokens from a Babel file and return a Counter dictionary."""

        token_dict = collections.Counter()

        # matches and removes beginning and end tags
        regex = re.compile(r'\[\d*\.\d*\]\n(.*)')
        matches = regex.findall(file_text)

        tokens = set()
        for match in matches:
            wp_tokenized = wordpunct_tokenize(match)
            tokens.update(wp_tokenized)

        for token in tokens:
            token_dict[token] += 1

        return token_dict


class MorfessorTokenizer(object):
    """An object that tokenizes a Morfessor model."""

    def __init__(self, target):
        """An object that tokenizes Morfessor model text files."""

        self.is_generator = False

        if isinstance(target, str):

            if os.path.isfile(target):

                # store file information
                self.file_path = os.path.abspath(target)
                self.directory, self.filename = os.path.split(self.file_path)

                # get file text
                with open(self.file_path, 'r') as f:
                    file_text = f.read()
                    _logger.info("File {} opened and read.".format(self.file_path))

                # build feature set
                self.file_segmentations = self._get_data_from_str(file_text)
                self._extract_feature_set(self.file_segmentations)

            else:
                # for taking output from morfessor BaselineModel object in string form

                self.file_path = None
                self.directory = None
                self.filename = None

                # extract list of tuples from string
                self.file_segmentations = self._get_data_from_str(target)

                # get feature set from
                self._extract_feature_set(self.file_segmentations)

        # for taking output from morfessor BaselineModel object in list form
        elif isinstance(target, GeneratorType):

            self.is_generator = True

            self.file_path = None
            self.directory = None
            self.filename = None

            self.file_segmentations = []

            for tup in target:

                # flatten tuples
                tup2 = self.flatten_to_generator(tup)
                self.file_segmentations.append(list(tup2))

            self._extract_feature_set(self.file_segmentations)

        else:
            _logger.warning("No input file found.")
            raise ValueError("input must be a file, not a directory")

    @staticmethod
    def flatten_to_generator(iterable):
        """Return a flattened generator for an iterable of mixed iterables and non-iterables.

        :param iterable: an iterable with any combination of iterable and non-iterable components
        """

        for item in iterable:
            if isinstance(item, list):
                for sub_item in item:
                    yield sub_item
            elif isinstance(item, tuple):
                for sub_item in item:
                    yield sub_item
            else:
                yield item

    def _get_data_from_str(self, file_text):
        """Return word count + init_segments tuples as a list for _extract_feature_set

        :param file_text:
        :return:
        """

        # used to find the morpheme divider, which is replaced with '+ +' below
        # to keep morpheme dividers attached in the token counts
        plus_regexp = re.compile(r' \+ ')

        # matches morphemes after counts
        line_regexp = re.compile(r'(\d*) (.*)')
        segmented = line_regexp.findall(file_text)

        segmented_fixed = []

        for tup in segmented:
            word_count = tup[0]
            segment_list = plus_regexp.sub('+ +', tup[1]).split(' ')
            tup2 = self.flatten_to_generator((word_count, segment_list))
            segmented_fixed.append(tup2)

        return segmented_fixed

    def _extract_feature_set(self, segmentation_list):
        """
        """

        # build dictionary of words and features
        feature_dict = dict()
        token_counter = collections.Counter()
        root_counter = collections.Counter()
        morph_counter = collections.Counter()

        for word_tuple in segmentation_list:

            # store word count, init_segments, and base form
            word_count, *segment_list = word_tuple
            word_base = ''.join(segment_list)
            word_base = re.sub(r'\+', '', word_base)

            # if data was from a generator object, fix morpheme boundary marks
            if self.is_generator:

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
            # TODO: account for problem initials like 'nakaka-'
            morph_list = sorted(segment_list, key=len)
            word_root = morph_list.pop()

            # add items to other stored dictionaries for fast access
            token_counter[word_root] += 1
            root_counter[word_root] += 1
            morph_counter.update(morph_list)

            # construct the feature dictionary
            feature_dict[word_base] = dict(count=word_count,
                                           segments=segment_list,
                                           root=word_root)

        # store dictionaries in class variables
        _logger.info("Feature dictionary extracted.")
        _logger.info("Morpheme, init_root, and token counters extracted.")

        self.feature_dict = feature_dict
        self.morph_counter = morph_counter
        self.root_counter = root_counter
        self.token_counter = token_counter

    def get_features(self):
        """Return the feature dictionary for the input file."""

        return self.feature_dict

    def get_morphs(self):
        """Return the morpheme counter for the input file."""

        return self.morph_counter

    def get_roots(self):
        """Return the init_root counter for the input file."""

        return self.root_counter

    def get_tokens(self):
        """Return the token counter for the input file."""

        return self.token_counter

    def write_tokens(self, out_file, token_set='tokens', output_type='counts'):
        """Output token sets to a text file.

        :param out_file: the destination file
        :param token_set: {'tokens', 'morphs', 'roots'}
        :param output_type: {'items', 'elements', 'counts'}
        """

        # test and choose correct token set
        if token_set not in {'tokens', 'morphs', 'roots'}:
            _logger.error("ERROR: Invalid token set: {}".format(token_set))
            err_msg = "token_set: {'tokens', 'morphs', 'roots'}"
            raise ValueError(err_msg)
        else:
            token_sets = {'tokens': self.token_counter,
                          'morphs': self.morph_counter,
                          'roots': self.root_counter}

            out_dict = token_sets[token_set]

        # test and choose correct Counter method for output
        if output_type not in {'items', 'elements', 'counts'}:
            _logger.error("ERROR: Invalid output type: {}".format(output_type))
            err_msg = "output_type: {'items', 'elements', 'counts'}"
            raise ValueError(err_msg)
        else:
            output_types = {'items': out_dict.keys(),
                            'elements': out_dict.elements(),
                            'counts': ['{}\t{}'.format(key, count) for (key, count)
                                      in sorted(out_dict.items(), key=itemgetter(1), reverse=True)]}

            out_list = output_types[output_type]

        with open(out_file, 'w') as f:
            f.write('\n'.join(out_list))

        out_name = os.path.basename(out_file)
        out_msg = "{} tokens written to {}".format(len(out_dict), out_name)

        _logger.info(out_msg)
        print(out_msg)

    def write_features(self, out_file, output_format):
        """Write feature set to output format (JSON).

        :param out_file: the destination file; do not use file extension
        :param output_format: JSON or pickle
        """

        if output_format.lower() not in {'json', 'pickle'}:
            _logger.error('ERROR: unrecognized output format: {}'.format(output_format))
            raise ValueError("output_format: {'json', 'pickle'}")

        elif output_format.lower() == 'json':

            with open(out_file + '.json', 'w') as f:
                json.dump(self.feature_dict, f)
                # _logger.info('Feature set written to {}.json'.format(out_file))

        elif output_format.lower() == 'pickle':

            with open(out_file + '.pickle', 'w') as f:
                pickle.dump(self.feature_dict, f)
                # _logger.info('Feature set written to {}.pickle'.format(out_file))

        out_name = os.path.basename(out_file)
        out_msg = "Feature set dictionary written to {}".format(out_name + '.' + output_format.lower())
        _logger.info(out_msg)
        # print(out_msg)
