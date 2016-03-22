
import collections
import logging
import os.path
import re
import tempfile

from segmenter import morfessor_main

_logger = logging.getLogger(__name__)


def train_morfessor_from_list(train_words, dampening='none', cycle='', save_file=None):
    """Call the Morfessor Baseline model main on the supplied word list.

    :param train_words: an iterable containing word strings for retraining
    :param dampening: model dampening type in {'none', 'ones', 'log'}
    :param cycle:
    :param save_file:
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


def train_morfessor_from_file(train_file, dampening='none', cycle='', save_file=None):
    """Call the Morfessor Baseline model main on the supplied word list.

    :param train_file:
    :param dampening: model dampening type in {'none', 'ones', 'log'}
    :param cycle:
    :param save_file:
    :return:
    """

    # input file must always be a list! the call fails otherwise
    # TODO: ensure morfessor_main works when this object is imported into a
    # script
    model = morfessor_main([train_file], dampening, cycle, save_file)

    return model


class ModelBuilder(object):

    def __init__(self, word_list_file, affix_list, dampening='none'):

        self.cycle_name = 'INIT'
        self.affix_list = affix_list

        # build initial morfessor.Baseline model
        self.init_model = train_morfessor_from_file(word_list_file,
                                                    dampening,
                                                    cycle='init',
                                                    save_file=None)

    def _extract_features(self, word_tuple):

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

        return word_count, word_base, word_root, segment_list

    def _build_feature_dict(self, morfessor_model):
        """Extract a nested dictionary of features (init_root, init_segments, count) for each word.

        :param morfessor_model:
        """

        # get segmentations from morfessor.Baseline model
        segmentation_list = morfessor_model.get_segmentations()

        # set a default dict with default arg as an empty dictionary
        feature_dict = collections.defaultdict(dict)

        for word_tuple in segmentation_list:

            wc, word_base, root, segments = self._extract_features(word_tuple)

            # for the __init__ cycle
            if self.cycle_name == 'INIT':

                # construct the initial dictionary
                feature_dict[word_base] = dict(count=wc,
                                               init_word_base=word_base,
                                               init_segments=segments,
                                               init_root=root)

                # store dictionaries in class variables
                self._feature_dict = feature_dict

            # second run-through, add new values to _feature_dict
            elif self.cycle_name == 'TEST':

                self._feature_dict[word_base]['test_segments'] = segments
                self._feature_dict[word_base]['test_root'] = root

            elif self.cycle_name == 'FINAL':

                self._feature_dict[word_base]['final_segments'] = segments
                self._feature_dict[word_base]['final_root'] = root

        if self.cycle_name == 'INIT':
            _logger.info("Feature dictionary extracted.")
        elif self.cycle_name == 'TEST':
            _logger.info("Feature dictionary updated with test values.")
        elif self.cycle_name == 'FINAL':
            _logger.info("Feature dictionary updated with final values.")


