#!/usr/bin/env python3

import logging
import sys

from morfessor.io import MorfessorIO

from evaluation import InfixerEvaluation
from preprocessor import InfixerModel

PY3 = sys.version_info.major == 3

_logger = logging.getLogger(__name__)


def output_model(train_file, out_file_bin, out_file_json, affix_list,
                 dampening='none', threshold=1):

    # morfessor model trained on the base set
    infixer_model = InfixerModel(train_file, affix_list, dampening)

    # build the test model
    infixer_model.build_test_model()

    # build the final model
    infixer_model.build_final_model(threshold, save_file=out_file_bin)
    infixer_model.write_feature_dict(out_file_json, 'json')


def main():

    # logging format
    log_level = logging.INFO
    logging_format = '%(asctime)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    default_formatter = logging.Formatter(logging_format, date_format)
    logging.basicConfig(level=log_level)

    # get logger working
    main_logger = logging.StreamHandler()
    main_logger.setLevel(log_level)
    main_logger.setFormatter(default_formatter)
    _logger.addHandler(main_logger)

    # # Settings for when log_file is present
    # if args.log_file is not None:
    #     fh = logging.FileHandler(args.log_file, 'w')
    #     fh.setLevel(loglevel)
    #     fh.setFormatter(default_formatter)
    #     _logger.addHandler(fh)
    #     # If logging to a file, make INFO the highest level for the
    #     # error stream
    #     ch.setLevel(max(loglevel, logging.INFO))

    # ----------------------------------------------------------------------

    # for affix filtering
    affix_list_redup_infixes = [r'^(?P<con>\w)(um)(?P<vow>\w)((?P=con)(?P=vow).*)',
                                r'^(?P<con>\w)(in)(?P<vow>\w)((?P=con)(?P=vow).*)',
                                r'^(i?.)(um)(.*)',
                                r'^(i?.)(in)(.*)']
    affix_list_infixes = [r'^(i?.)(um)(.*)', r'^(i?.)(in)(.*)']

    # ----------------------------------------------------------------------

    models_to_train = [('data/tl_wiki_tokens.txt',
                        'output/models/tl_wiki_ri_none',
                        'output/featdicts/tl_wiki_ri_none',
                        affix_list_redup_infixes,
                        'none'),
                       ('data/tl_wiki_tokens.txt',
                        'output/models/tl_wiki_ri_ones',
                        'output/featdicts/tl_wiki_ri_ones',
                        affix_list_redup_infixes,
                        'ones'),
                       ('data/tl_wiki_tokens.txt',
                        'output/models/tl_wiki_ri_log',
                        'output/featdicts/tl_wiki_ri_log',
                        affix_list_redup_infixes,
                        'log'),
                       ('data/tl_wiki_tokens.txt',
                        'output/models/tl_wiki_i_none',
                        'output/featdicts/tl_wiki_i_none',
                        affix_list_infixes,
                        'none'),
                       ('data/tl_wiki_tokens.txt',
                        'output/models/tl_wiki_i_ones',
                        'output/featdicts/tl_wiki_i_ones',
                        affix_list_infixes,
                        'ones'),
                       ('data/tl_wiki_tokens.txt',
                        'output/models/tl_wiki_i_log',
                        'output/featdicts/tl_wiki_i_log',
                        affix_list_infixes,
                        'log'),
                       ]

    for train, out_bin, out_json, affixl, damp in models_to_train:

        output_model(train, out_bin, out_json, affixl, dampening=damp, threshold=1)

if __name__ == '__main__':
    main()
