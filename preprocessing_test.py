#!/usr/bin/env python3

import logging
import sys

from preprocessor import ModelBuilder
from preprocessor2 import InfixerModel, AffixFilter

from segmenter import morfessor_main
from evaluation import InfixerEvaluation, InfixerSegmenter

PY3 = sys.version_info.major == 3

_logger = logging.getLogger(__name__)


def preprocessor1():

    # logging format
    log_level = logging.INFO
    logging_format = '%(asctime)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    default_formatter = logging.Formatter(logging_format, date_format)
    logging.basicConfig(level=log_level)

    # get logger working
    main_logger = logging.FileHandler('output/logs/wiki_tokens.log')
    main_logger.setLevel(log_level)
    main_logger.setFormatter(default_formatter)
    _logger.addHandler(main_logger)

    # ----------------------------------------------------------------------

    # files and directories
    # base_dir = "/Users/srbutler/Documents/CUNY/Classes/S4/thesis/"
    train_file = 'data/tl_wiki_tokens.txt'
    base_outfile_name = 'wiki_2'

    # ----------------------------------------------------------------------

    # morfessor model trained on the base set
    _logger.info("INIT CYCLE: training Morfessor Baseline model")
    model1 = morfessor_main([train_file], dampening='none')
    model_examiner = ModelBuilder(model1)
    model_examiner.write_feature_dict(
        'output/init_' + base_outfile_name, 'json')

    # ----------------------------------------------------------------------

    # affix filtering
    test_affix_list = [r'^(.)(um)(.*)', r'^(.)(in)(.*)']
    model_examiner.filter_affixes(test_affix_list)

    # build the test model
    model_examiner.build_test_model(
        dampening='none', cycle='test', save_file='tl_babel_')
    model_examiner.write_feature_dict(
        'output/retrain_' + base_outfile_name, 'json')

    # ----------------------------------------------------------------------

    # build the final model
    model_examiner.build_final_model(
        dampening='none', threshold=0, cycle='final', save_file='tl_babel_')
    model_examiner.write_changed_tokens(
        'output/changed_tokens_' + base_outfile_name + '.txt')
    model_examiner.write_feature_dict(
        'output/final_' + base_outfile_name, 'json')


def preprocessor2():

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
    base_outfile_name = 'babel_thresh2'

    # ----------------------------------------------------------------------

    # for affix filtering
    test_affix_list = [r'^(.)(um)(.*)', r'^(.)(in)(.*)']

    # morfessor model trained on the base set
    infixer_model = InfixerModel(train_file, test_affix_list, dampening='none')
    # infixer_model.write_feature_dict('output/REWRITE_init_' + base_outfile_name, 'json')

    # ----------------------------------------------------------------------

    # build the test model
    infixer_model.build_test_model(save_file='REWRITE_tl_babel_')
    # infixer_model.write_feature_dict('output/REWRITE_retrain_' + base_outfile_name, 'json')

    # ----------------------------------------------------------------------

    # build the final model
    infixer_model.build_final_model(threshold=1, save_file='REWRITE_tl_babel_')
    # infixer_model.write_changed_tokens('output/REWRITE_changed_tokens_' + base_outfile_name + '.txt')
    infixer_model.write_feature_dict(
        'output/REWRITE_final_' + base_outfile_name, 'json')

    # ----------------------------------------------------------------------

    gold_standard = 'eval_tagalog_verbs.txt'

    morf_model = infixer_model.get_model('final')
    feature_dict = infixer_model.feature_dictionary()
    # evaluator = InfixerEvaluation(morf_model, feature_dict, test_affix_list)
    # evaluator.evaluate_model(gold_standard_file=gold_standard, wilcoxon=True)

    file_to_segment = 'tagalog_verbs_plain.txt'
    outfile = 'output/SEGMENTATION_TEST.txt'

    model_segmenter = InfixerSegmenter(
        morf_model, feature_dict, test_affix_list)
    model_segmenter.segment_file(file_to_segment, outfile)

if __name__ == '__main__':
    preprocessor2()
