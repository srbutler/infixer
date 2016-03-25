#!/usr/bin/env python3

import logging
import sys

from morfessor.io import MorfessorIO

from evaluation import InfixerEvaluation
from preprocessor import InfixerModel

PY3 = sys.version_info.major == 3

_logger = logging.getLogger(__name__)


def test_preprocessor():

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

    # files and directories
    # base_dir = "/Users/srbutler/Documents/CUNY/Classes/S4/thesis/"
    train_file = 'data/tl_babel_tokens.txt'
    base_outfile_name = 'babel_thresh1'

    # ----------------------------------------------------------------------

    # for affix filtering
    # test_affix_list = ['^(?P<con>\w)(um)(?P<vow>\w)((?P=con)(?P=vow).*)',
    #                    '^(?P<con>\w)(in)(?P<vow>\w)((?P=con)(?P=vow).*)',
    #                    r'^(i?.)(um)(.*)',
    #                    r'^(i?.)(in)(.*)']
    test_affix_list = [r'^(i?.)(um)(.*)', r'^(i?.)(in)(.*)']

    # morfessor model trained on the base set
    # infixer_model = InfixerModel(train_file, test_affix_list, dampening='none')
    # infixer_model.write_feature_dict('output/REWRITE_init_' + base_outfile_name, 'json')

    # ----------------------------------------------------------------------

    # build the test model
    # infixer_model.build_test_model()
    # infixer_model.write_feature_dict('output/REWRITE_retrain_' + base_outfile_name, 'json')

    # ----------------------------------------------------------------------

    # build the final model
    # infixer_model.build_final_model(threshold=1, save_file='trained_models/babel_final_binary')
    # infixer_model.write_changed_tokens('output/REWRITE_changed_tokens_' + base_outfile_name + '.txt')
    # infixer_model.write_feature_dict('output/REWRITE_final_' + base_outfile_name, 'json')

    # ----------------------------------------------------------------------

    # get stored model and feature dict
    io = MorfessorIO(encoding='utf-8', compound_separator='\s+', atom_separator=None, lowercase=False)
    morf_model = io.read_binary_model_file('trained_models/babel_final_binaryFINAL_bin')
    feature_dict = InfixerModel.get_features_dict_from_file('output/REWRITE_final_babel_thresh1.json')

    file_to_segment = 'data/eval_tl_verbs.txt'
    segment_outfile = 'output/segmentation_test_6.csv'

    gold_standard = 'data/eval_tl_verbs_segmented.txt'
    gs_outfile = 'output/GS_segmentation_test_1'

    # morf_model = infixer_model.get_model('final')
    # feature_dict = infixer_model.feature_dictionary()

    # eval and segment to file
    evaluator = InfixerEvaluation(morf_model, feature_dict, test_affix_list)
    evaluator.segment_file(file_to_segment, segment_outfile)
    evaluator.segment_gold_standard(gold_standard, gs_outfile)

if __name__ == '__main__':
    test_preprocessor()
