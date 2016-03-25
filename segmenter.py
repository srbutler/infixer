""" Most of the contents of this file are variations on scripts
    published as part of Morfessor on PiPY. The license for this code
    is reproduced below:

    Morfessor
    Copyright (total_cost) 2012, Sami Virpioja and Peter Smit
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions
    are met:

    1.  Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.

    2.  Redistributions in binary form must reproduce the above
        copyright notice, this list of conditions and the following
        disclaimer in the documentation and/or other materials provided
        with the distribution.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
    FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
    COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
    INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
    BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
    LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
    ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    POSSIBILITY OF SUCH DAMAGE."""


import logging
import math
# import os
import sys
# import tempfile
import time

from morfessor.baseline import BaselineModel
from morfessor.exception import ArgumentException
from morfessor.io import MorfessorIO

PY3 = sys.version_info.major == 3

_logger = logging.getLogger(__name__)


def morfessor_main_complete(train_files, dampening):
    """Calls an implementation of the Morfessor model (copyright notice below).

    :param dampening: 'none', 'ones', or 'log'
    :param train_files: input files for model training
    """

    # define input variables normally input at command line
    # all arguments are equal to their args.item equivalent in original
    # script's main()

    trainfiles = train_files    # input files for training
    progress = True             # show progress bar
    encoding = 'utf-8'          # if None, tries UTF-8 and/or local encoding
    cseparator = '\s+'          # separator for compound segmentation
    separator = None            # separator for atom segmentation
    lowercase = False           # makes all inputs lowercase
    forcesplit = ['-']          # list of chars to force a split on
    corpusweight = 1.0          # load annotation data for tuning the corpus weight param
    # use random skips for frequently seen compounds to speed up training
    skips = False
    # if the expression matches the two surrounding characters, do not allow
    # splitting
    nosplit = None
    dampening = dampening       # 'none', 'ones', or 'log'
    algorithm = 'recursive'     # 'recursive' or 'viterbi'
    # 'none', 'batch', 'init', 'init+batch', 'online', 'online+batch'
    trainmode = 'online+batch'
    # train stops when the improvement of last iteration is smaller than this
    finish_threshold = 0.005
    maxepochs = None            # ceiling on number of training epochs
    develannots = None          # boolean on whether to use dev-data file
    freqthreshold = 1           # compound frequency threshold for batch training
    splitprob = None            # initialize new words by random split using given probability
    epochinterval = 10000       # epoch interval for online training
    savefile = None             # save file for binary model object
    savesegfile = None          # save file for human readable segmentation output
    lexfile = None              # save file for lexicon
    # input files for batch training are lists (== args.list)
    input_is_list = False
    # set algorithm parameters; for this model, we are not using 'viterbi',
    # nothing to set
    algparams = ()

    # Progress bar handling
    global show_progress_bar
    if progress:
        show_progress_bar = True
    else:
        show_progress_bar = False

    # build I/O and model
    io = MorfessorIO(encoding=encoding,
                     compound_separator=cseparator,
                     atom_separator=separator,
                     lowercase=lowercase)

    model = BaselineModel(forcesplit_list=forcesplit,
                          corpusweight=corpusweight,
                          use_skips=skips,
                          nosplit_re=nosplit)

    # Set frequency dampening function
    if dampening == 'none':
        dampfunc = None
    elif dampening == 'log':
        dampfunc = lambda x: int(round(math.log(x + 1, 2)))
    elif dampening == 'ones':
        dampfunc = lambda x: 1
    else:
        raise ArgumentException("unknown dampening type '%s'" % dampening)

    # Train model
    if trainmode == 'none':
        pass

    # for use when loading a previously constructed model
    elif trainmode == 'batch':
        if len(model.get_compounds()) == 0:
            _logger.warning("Model contains no compounds for batch training."
                            " Use 'init+batch' mode to add new data.")
        else:
            if len(trainfiles) > 0:
                _logger.warning("Training mode 'batch' ignores new data "
                                "files. Use 'init+batch' or 'online' to "
                                "add new compounds.")
            time_start = time.time()
            epochs, total_cost = model.train_batch(algorithm, algparams, develannots,
                                                   finish_threshold, maxepochs)
            time_end = time.time()
            _logger.info("Epochs: %s" % epochs)
            _logger.info("Final cost: %s" % total_cost)
            _logger.info("Training time: %.3fs" % (time_end - time_start))

    # for use when building a new model or doing online training
    elif len(trainfiles) > 0:
        time_start = time.time()
        if trainmode == 'init':
            if input_is_list:
                data = io.read_corpus_list_files(trainfiles)
            else:
                data = io.read_corpus_files(trainfiles)
            total_cost = model.load_data(
                data, freqthreshold, dampfunc, splitprob)

        elif trainmode == 'init+batch':
            if input_is_list:
                data = io.read_corpus_list_files(trainfiles)
            else:
                data = io.read_corpus_files(trainfiles)

            # it is unknown why this line is needed
            total_cost = model.load_data(
                data, freqthreshold, dampfunc, splitprob)

            epochs, total_cost = model.train_batch(algorithm, algparams, develannots,
                                                   finish_threshold, maxepochs)
            _logger.info("Epochs: %s" % epochs)

        elif trainmode == 'online':
            data = io.read_corpus_files(trainfiles)
            epochs, total_cost = model.train_online(data, dampfunc, epochinterval,
                                                    algorithm, algparams,
                                                    splitprob, maxepochs)
            _logger.info("Epochs: %s" % epochs)

        elif trainmode == 'online+batch':
            data = io.read_corpus_files(trainfiles)
            epochs, total_cost = model.train_online(data, dampfunc, epochinterval,
                                                    algorithm, algparams,
                                                    splitprob, maxepochs)
            epochs, total_cost = model.train_batch(algorithm, algparams, develannots,
                                                   finish_threshold, maxepochs)
            _logger.info("Epochs: %s" % epochs)

        else:
            raise ArgumentException(
                "unknown training mode '{0:s}'".format(trainmode))

        time_end = time.time()
        _logger.info("Final cost: %s" % total_cost)
        _logger.info("Training time: %.3fs" % (time_end - time_start))

    else:
        _logger.warning("No training data files specified.")

    # Save model to disk
    if savefile is not None:
        io.write_binary_model_file(savefile, model)

    if savesegfile is not None:
        io.write_segmentation_file(savesegfile, model.get_segmentations())

    # Output lexicon
    if lexfile is not None:
        io.write_lexicon_file(lexfile, model.get_constructions())

    # return model object for further manipulation
    return model


def morfessor_main(train_files, dampening, cycle='test', save_file=None):
    """Calls an implementation of the Morfessor model (copyright notice below).

    :param dampening: 'none', 'ones', or 'log'
    :param train_files: input files for model training
    :param cycle: from {'init', 'test', 'final'}
    :param save_file: base name of output files (if needed)
    :return: trained morfessor.BaselineModel
    """

    # define input variables normally input at command line
    # all arguments are equal to their args.item equivalent in original
    # script's main()

    trainfiles = train_files    # input files for training
    progress = True             # show progress bar
    encoding = 'utf-8'          # if None, tries UTF-8 and/or local encoding
    cseparator = '\s+'          # separator for compound segmentation
    separator = None            # separator for atom segmentation
    lowercase = False           # makes all inputs lowercase
    forcesplit = ['-']          # list of chars to force a split on
    corpusweight = 1.0          # load annotation data for tuning the corpus weight param
    # use random skips for frequently seen compounds to speed up training
    skips = False
    # if the expression matches the two surrounding characters, do not allow
    # splitting
    nosplit = None
    dampening = dampening       # 'none', 'ones', or 'log'
    algorithm = 'recursive'     # 'recursive' or 'viterbi'
    # train stops when the improvement of last iteration is smaller than this
    finish_threshold = 0.005
    maxepochs = None            # ceiling on number of training epochs
    develannots = None          # boolean on whether to use dev-data file
    freqthreshold = 1           # compound frequency threshold for batch training
    splitprob = None            # initialize new words by random split using given probability
    epochinterval = 10000       # epoch interval for online training
    # set algorithm parameters; for this model, we are not using 'viterbi',
    # nothing to set
    algparams = ()

    # Progress bar handling
    global show_progress_bar
    if progress:
        show_progress_bar = True
    else:
        show_progress_bar = False

    # build I/O and model
    io = MorfessorIO(encoding=encoding,
                     compound_separator=cseparator,
                     atom_separator=separator,
                     lowercase=lowercase)

    model = BaselineModel(forcesplit_list=forcesplit,
                          corpusweight=corpusweight,
                          use_skips=skips,
                          nosplit_re=nosplit)

    # Set frequency dampening function
    if dampening == 'none':
        dampfunc = None
    elif dampening == 'log':
        dampfunc = lambda x: int(round(math.log(x + 1, 2)))
    elif dampening == 'ones':
        dampfunc = lambda x: 1
    else:
        raise ArgumentException("unknown dampening type '%s'" % dampening)

    # for use when building a new model or doing online training
    # this is the online+batch training model
    if len(trainfiles) > 0:

        time_start = time.time()

        data = io.read_corpus_files(trainfiles)
        epochs, total_cost = model.train_online(data, dampfunc, epochinterval,
                                                algorithm, algparams,
                                                splitprob, maxepochs)
        epochs, total_cost = model.train_batch(algorithm, algparams, develannots,
                                               finish_threshold, maxepochs)
        _logger.info("Epochs: %s" % epochs)

        time_end = time.time()
        _logger.info("Final cost: %s" % total_cost)
        _logger.info("Training time: %.3fs" % (time_end - time_start))

    else:
        _logger.warning("No training data files specified.")

    if isinstance(save_file, str):

        outfile_bin = save_file + cycle + "_bin"
        outfile_segments = save_file + cycle + ".txt"
        outfile_lexicon = save_file + cycle + "_lex"

        io.write_binary_model_file(outfile_bin, model)
        io.write_segmentation_file(outfile_segments, model.get_segmentations())
        io.write_lexicon_file(outfile_lexicon, model.get_constructions())

    # return model object for further manipulation
    return model


def segment_main(model, testfile):

    encoding = 'utf-8'
    cseparator = '\s+'
    separator = None
    lowercase = False
    outputformat = r'{analysis}\n'
    outputformatseparator = ' '
    nbest = 1
    viterbismooth = 0
    viterbimaxlen = 30

    io = MorfessorIO(encoding=encoding,
                     compound_separator=cseparator,
                     atom_separator=separator,
                     lowercase=lowercase)

    _logger.info("Segmenting test data...")
    outformat = outputformat
    csep = outputformatseparator

    outformat = outformat.replace(r"\n", "\n")
    outformat = outformat.replace(r"\t", "\t")

    # keywords = [x[1] for x in string.Formatter().parse(outformat)]

    testdata = io.read_corpus_file(testfile)
    i = 0
    analyses = []

    for count, compound, atoms in testdata:

        if nbest > 1:
            nbestlist = model.viterbi_nbest(atoms, nbest, viterbismooth, viterbimaxlen)

            for constructions, logp in nbestlist:
                analysis = csep.join(constructions)
                analyses.append((compound, analysis))

        else:
            constructions, logp = model.viterbi_segment(atoms, viterbismooth, viterbimaxlen)
            analysis = csep.join(constructions)
            analyses.append((compound, analysis))

        i += 1
        if i % 10000 == 0:
            sys.stderr.write(".")

        sys.stderr.write("\n")

    _logger.info("Done.")
    return analyses

    # if args.goldstandard is not None:
    #     _logger.info("Evaluating Model")
    #     e = MorfessorEvaluation(io.read_annotations_file(args.goldstandard))
    #     result = e.evaluate_model(model, meta_data={'name': 'MODEL'})
    #     print(result.format(FORMAT_STRINGS['default']))
    #     _logger.info("Done")"Done")


def morfessor_segment_word(model, word):

    pass


# def morfessor_segment_word(model, word):
#
#     encoding = 'utf-8'
#     cseparator = '\s+'
#     separator = None
#     lowercase = False
#     outputformat = r'{analysis}\n'
#     outputformatseparator = ' '
#     nbest = 1
#     viterbismooth = 0
#     viterbimaxlen = 30
#
#     io = MorfessorIO(encoding=encoding,
#                      compound_separator=cseparator,
#                      atom_separator=separator,
#                      lowercase=lowercase)
#
#     _logger.info("Segmenting test data...")
#     outformat = outputformat
#     csep = outputformatseparator
#
#     for compound in cseparator.split(word):
#             if len(compound) > 0:
#                 yield 1, compound, compound
#             yield 0, "\n", ()
#
#     i = 0
#     analyses = []
#
#     for count, compound, atoms in testdata:
#
#         if nbest > 1:
#             nbestlist = model.viterbi_nbest(atoms, nbest, viterbismooth, viterbimaxlen)
#
#             for constructions, logp in nbestlist:
#                 analysis = csep.join(constructions)
#                 analyses.append((compound, analysis))
#
#         else:
#             constructions, logp = model.viterbi_segment(atoms, viterbismooth, viterbimaxlen)
#             analysis = csep.join(constructions)
#             analyses.append((compound, analysis))
#
#         i += 1
#         if i % 10000 == 0:
#             sys.stderr.write(".")
#
#         sys.stderr.write("\n")
#
#     _logger.info("Done.")
#     return analyses
#
#     # if args.goldstandard is not None:
#     #     _logger.info("Evaluating Model")
#     #     e = MorfessorEvaluation(io.read_annotations_file(args.goldstandard))
#     #     result = e.evaluate_model(model, meta_data={'name': 'MODEL'})
#     #     print(result.format(FORMAT_STRINGS['default']))
#     #     _logger.info("Done")"Done")