
import glob
import logging
import os.path
import re

from morfessor.io import MorfessorIO

from evaluation import InfixerEvaluation
from preprocessor import InfixerModel

_logger = logging.getLogger(__name__)


def segment_list(word_list_file, segment_outfile, trained_model_bin,
                 feature_dict_json, affix_list):

    # get stored model and feature dict
    io = MorfessorIO(encoding='utf-8', compound_separator='\s+', atom_separator=None, lowercase=False)
    trained_model = io.read_binary_model_file(trained_model_bin)
    feature_dict = InfixerModel.get_features_dict_from_file(feature_dict_json)

    # eval and segment to file
    evaluator = InfixerEvaluation(trained_model, feature_dict, affix_list)
    evaluator.segment_file(word_list_file, segment_outfile)


def segment_gold_standard(gs_file, segment_outfile, trained_model_bin,
                          feature_dict_json, affix_list):

    # get stored model and feature dict
    io = MorfessorIO(encoding='utf-8', compound_separator='\s+', atom_separator=None, lowercase=False)
    trained_model = io.read_binary_model_file(trained_model_bin)
    feature_dict = InfixerModel.get_features_dict_from_file(feature_dict_json)

    # eval and segment to file
    evaluator = InfixerEvaluation(trained_model, feature_dict, affix_list)
    evaluator.segment_gold_standard(gs_file, segment_outfile)


def format_gold_standard(gs_file, new_gs_outfile, trained_model_bin,
                         feature_dict_json, affix_list):

    # get stored model and feature dict
    io = MorfessorIO(encoding='utf-8', compound_separator='\s+', atom_separator=None, lowercase=False)
    trained_model = io.read_binary_model_file(trained_model_bin)
    feature_dict = InfixerModel.get_features_dict_from_file(feature_dict_json)

    # eval and segment to file
    evaluator = InfixerEvaluation(trained_model, feature_dict, affix_list)
    evaluator.output_modified_gold_standard(gs_file, new_gs_outfile)


def evaluate_goldstandard(gs_file, trained_model_bin, feature_dict_json, affix_list):

    # get stored model and feature dict
    io = MorfessorIO(encoding='utf-8', compound_separator='\s+', atom_separator=None, lowercase=False)
    trained_model = io.read_binary_model_file(trained_model_bin)
    feature_dict = InfixerModel.get_features_dict_from_file(feature_dict_json)

    # eval and segment to file
    evaluator = InfixerEvaluation(trained_model, feature_dict, affix_list)
    evaluator.evaluate_model(gs_file)


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

    gold_standard = 'data/eval_tl_verbs_segmented.txt'

    new_gs_outfile = 'output/segmentation/eval_tl_verbs_babel_ri_none.txt'

    segmentation_outfile = 'output/segmentation/tl_babel_ri_none_segments.csv'
    model = 'output/models/tl_babel_ri_none_bin'
    feat_dict = 'output/featdicts/tl_babel_ri_none.json'
    affix_list = affix_list_redup_infixes

    for path in glob.glob('output/models/*_bin'):

        base = os.path.basename(path)
        new_gs = 'output/gs_files/eval_' + base + '.txt'
        model = path
        feat_dict = 'output/featdicts/' + base[:-4] + '.json'

        if re.search('ri', base):
            affix_list = affix_list_redup_infixes
        else:
            affix_list = affix_list_infixes

        format_gold_standard(gold_standard, new_gs, model, feat_dict, affix_list)

if __name__ == '__main__':
    main()