#!/usr/bin/env python3


import logging
import sys

# import morfessor

from preprocessor import ModelBuilder
from segmenter import morfessor_main

PY3 = sys.version_info.major == 3

_logger = logging.getLogger(__name__)


def main():

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
    model_examiner.write_feature_dict('output/init_' + base_outfile_name, 'json')

    # ----------------------------------------------------------------------

    # affix filtering
    test_affix_list = [r'^(.)(um)(.*)', r'^(.)(in)(.*)']
    model_examiner.filter_affixes(test_affix_list)

    # build the test model
    model_examiner.build_test_model(dampening='none', cycle='test', save_file='tl_babel_')
    model_examiner.write_feature_dict('output/retrain_' + base_outfile_name, 'json')

    # ----------------------------------------------------------------------

    # build the final model
    model_examiner.build_final_model(dampening='none', threshold=0, cycle='final', save_file='tl_babel_')
    model_examiner.write_changed_tokens('output/changed_tokens_' + base_outfile_name + '.txt')
    model_examiner.write_feature_dict('output/final_' + base_outfile_name, 'json')

if __name__ == '__main__':
    main()
