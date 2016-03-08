import logging
import math
import sys
import time

from morfessor.baseline import BaselineModel
from morfessor.exception import ArgumentException
from morfessor.io import MorfessorIO

PY3 = sys.version_info.major == 3

_logger = logging.getLogger(__name__)


def morfessor_model_from_file(train_files, dampening):
    """Calls an implementation of the Morfessor model (copyright notice below).

    :param dampening: 'none', 'ones', or 'log'
    :param train_files: input files for model training

    Morfessor
    Copyright (c) 2012, Sami Virpioja and Peter Smit
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
    POSSIBILITY OF SUCH DAMAGE.
    """

    # define input variables normally input at command line
    # all arguments are equal to their args.item equivalent in original script's main()

    trainfiles = train_files    # input files for training
    progress = True             # show progress bar
    encoding = 'utf-8'          # if None, tries UTF-8 and/or local encoding
    cseparator = '\s+'          # separator for compound segmentation
    separator = None            # separator for atom segmentation
    lowercase = False           # makes all inputs lowercase
    forcesplit = ['-']          # list of chars to force a split on
    corpusweight = 1.0          # load annotation data for tuning the corpus weight param
    skips = False               # use random skips for frequently seen compounds to speed up training
    nosplit = None              # if the expression matches the two surrounding characters, do not allow splitting
    dampening = dampening       # 'none', 'ones', or 'log'
    algorithm = 'recursive'     # 'recursive' or 'viterbi'
    trainmode = 'online+batch'    # 'none', 'batch', 'init', 'init+batch', 'online', 'online+batch'
    finish_threshold = 0.005    # train stops when the improvement of last iteration is smaller than this
    maxepochs = None            # ceiling on number of training epochs
    develannots = None          # boolean on whether to use dev-data file
    freqthreshold = 1           # compound frequency threshold for batch training
    splitprob = None            # initialize new words by random split using given probability
    epochinterval = 10000       # epoch interval for online training
    savefile = None             # save file for binary model object
    savesegfile = None          # save file for human readable segmentation output
    lexfile = None              # save file for lexicon
    input_is_list = False       # input files for batch training are lists (== args.list)

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

    # Set algorithm parameters; for this model, we are not using 'viterbi',
    # so nothing needs to be set
    algparams = ()

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
            ts = time.time()
            e, c = model.train_batch(algorithm, algparams, develannots,
                                     finish_threshold, maxepochs)
            te = time.time()
            _logger.info("Epochs: %s" % e)
            _logger.info("Final cost: %s" % c)
            _logger.info("Training time: %.3fs" % (te - ts))

    # for use when building a new model or doing online training
    elif len(trainfiles) > 0:
        ts = time.time()
        if trainmode == 'init':
            if input_is_list:
                data = io.read_corpus_list_files(trainfiles)
            else:
                data = io.read_corpus_files(trainfiles)
            c = model.load_data(data, freqthreshold, dampfunc, splitprob)

        elif trainmode == 'init+batch':
            if input_is_list:
                data = io.read_corpus_list_files(trainfiles)
            else:
                data = io.read_corpus_files(trainfiles)

            # it is unknown why this line is needed
            # c = model.load_data(data, freqthreshold, dampfunc, splitprob)

            e, c = model.train_batch(algorithm, algparams, develannots,
                                     finish_threshold, maxepochs)
            _logger.info("Epochs: %s" % e)

        elif trainmode == 'online':
            data = io.read_corpus_files(trainfiles)
            e, c = model.train_online(data, dampfunc, epochinterval,
                                      algorithm, algparams,
                                      splitprob, maxepochs)
            _logger.info("Epochs: %s" % e)

        elif trainmode == 'online+batch':
            data = io.read_corpus_files(trainfiles)
            e, c = model.train_online(data, dampfunc, epochinterval,
                                      algorithm, algparams,
                                      splitprob, maxepochs)
            e, c = model.train_batch(algorithm, algparams, develannots,
                                     finish_threshold, maxepochs)
            _logger.info("Epochs: %s" % e)

        else:
            raise ArgumentException("unknown training mode '{0:s}'".format(trainmode))

        te = time.time()
        _logger.info("Final cost: %s" % c)
        _logger.info("Training time: %.3fs" % (te - ts))

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

