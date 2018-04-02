#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Arne F. Meyer <arne.f.meyer@gmail.com>
# License: GPLv3

"""
    a full feature extraction/classification/validation pipeline using
    sklearn's basic architecture:
    http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html

    Todo: replace database input by more general format
"""

import os
import os.path as op
import numpy as np
import matplotlib.pyplot as plt
import click
from sklearn import cross_validation
from sklearn import lda

from lindenlab.io import database as lldb
from lindenlab import util as llutil

from BehavioralScoring.features import STFTFeatureExtractor
from BehavioralScoring.classify import LinearSVM, ParallelLinearSVC
from BehavioralScoring.groups import MOUSE_GROUPS_SIMPLE


# behavioral state groups
_GROUPS = {0: 'Unclassified',
           1: 'Active exploration',
           2: 'Quiet wakefulness',
           3: 'Grooming',
           4: 'Rearing',
           5: 'Eating',
           6: 'Sleeping'}


# -----------------------------------------------------------------------------
# helper functions
# -----------------------------------------------------------------------------

def load_data(rec_paths, feature_extractor, samplerate=100.):

    from scipy.interpolate import interp1d

    X = []
    Y = []
    n_channels = None
    groups = None

    for rec_path in rec_paths:

        print "loading data from", rec_path

        # read accelerometer data
        session_path = op.join(rec_path, '..')
        reader = lldb.DataIO(session_path)
        seg = reader.read_segment(op.basename(rec_path), dtypes=['aux'],
                                  preprocess_aux=False)
        aux_sig = seg.analogsignals[0]

        x = aux_sig.base
        fs = aux_sig.sampling_rate.base
        t_aux = aux_sig.times
        n_channels = x.shape[1]

        try:
            # read annotations
            annotations = lldb.load_video_annotations(rec_path)

            # create label for each time step (using linear interpolation)
            timestamps = annotations['timestamps']
            labels = annotations['assigned_groups']
            groups = annotations['groups']

            f = interp1d(timestamps, labels, kind='nearest',
                         fill_value='extrapolate')
            y = f(t_aux)
            print "labels:", np.unique(y)

        except BaseException:
            y = None

        if fs != samplerate:

            subsample = int(np.round(fs / float(samplerate)))
            x = x[::subsample, :]
            if y is not None:
                y = y[::subsample]
            fs = float(fs) / subsample

        xx = feature_extractor.transform(x)

        X.append(xx)

        if y is not None:
            Y.append(y)

    X = np.concatenate(tuple(X), axis=0)
    if len(Y) > 0:
        Y = np.concatenate(tuple(Y), axis=0)

        # let's replace sleep by quiescent state as long as we don't have
        # reliable observables (eye, LFP) for sleep
        Y[Y == 6] = 2

    else:
        Y = None

    assert n_channels is not None

    return X, Y, fs, n_channels, groups


def get_sklearn_minor_version():

    import sklearn
    v = sklearn.__version__
    _, minor, _ = v.split('.')

    return int(minor)


def get_annotated_recordings(paths):

    rec_paths = []

    for root, dirs, files in os.walk(paths):

        if 'video_annotations.npz' in files:
            rec_paths.append(root)

    rec_paths.sort()

    return rec_paths


def get_recordings(paths):

    rec_paths = []

    for root, dirs, files in os.walk(paths):

        if 'aux_channels.npz' in files:
            rec_paths.append(root)

    rec_paths.sort()

    return rec_paths


def get_annotated_recordings_from_file(file_path, database=None):

    if database is not None:
        database = op.realpath(database)
    else:
        database = ''

    rec_paths = []
    with open(file_path, 'r') as f:

        for line in f:
            line = line.strip()
            if len(line) > 0:
                p = op.join(database, line)
                rec_paths.extend(get_annotated_recordings(p))

    return rec_paths


def compute_error_rates(y, y_pred, labels):

    n_labels = len(labels)
    tp_rate = np.zeros((n_labels,))
    fp_rate = np.zeros((n_labels))

    for i, label in enumerate(labels):

        v = y == label
        tpr = np.sum(y[v] == y_pred[v]) / float(np.sum(v))

        fp = np.sum(np.logical_and(y_pred == label, y != label))
        tn = np.sum(np.logical_and(y != label, y_pred != label))
        fpr = fp / float(fp + tn)

        tp_rate[i] = tpr
        fp_rate[i] = fpr

    return tp_rate, fp_rate


# -----------------------------------------------------------------------------
# pipeline
# -----------------------------------------------------------------------------

def train_classifier(rec_paths, run_cv=True, samplerate=100., dest_path=None,
                     algorithm='SVM', use_cluster=False,
                     f_upper=10.):

    # ----- parameters -----
    n_folds = 4
    feature_extractor = STFTFeatureExtractor(nfft=2*256, shift=1,
                                             samplerate=samplerate,
                                             f_lower=0, f_upper=f_upper,
                                             winlen=2*256)

    if dest_path is None:
        dest_path = op.join(op.expanduser('~'), 'research', 'data',
                            'experiments', 'BehavioralScoring')
    llutil.makedirs_save(dest_path)

    fig_formats = ['pdf', 'png']

    # ----- load and convert data -----
    X, y, fs, n_channels, groups = load_data(rec_paths, feature_extractor,
                                             samplerate=samplerate)
    assert fs == samplerate
    freqs = feature_extractor.get_frequencies()

    all_labels = np.asarray(groups.keys())
    n_labels = len(all_labels)

    print "total data size: {} observations x {} features".format(X.shape[0],
                                                                  X.shape[1])

    # ----- set up estimation/validation methods -----

    if algorithm.upper() == 'SVM':

        if use_cluster:
            tmp_path = op.join(dest_path, 'tmp')
            model = ParallelLinearSVC(tmp_path=tmp_path, local=False)

        else:
            cv = cross_validation.StratifiedKFold(y, n_folds=n_folds,
                                                  shuffle=True,
                                                  random_state=0)
            minor_version = get_sklearn_minor_version()
            if minor_version <= 16:
                class_weight = 'auto'
            else:
                class_weight = 'balanced'
            C_values = 2. ** np.linspace(-3, 5, 10)
            model = LinearSVM(n_folds=n_folds, verbose=False,
                              class_weight=class_weight,
                              dual=False, penalty='l2',
                              max_iter=1000,
                              C_values=C_values)

    elif algorithm.upper() == 'LDA':

        priors = []
        for i, label in enumerate(all_labels):
            n = np.sum(y == label)
            if n > 0:
                priors.append(n / float(y.shape[0]))

        model = lda.LDA(priors=np.asarray(priors))

    # ----- cross-validation -----

    if run_cv:
        # cross-validation

        cv = cross_validation.StratifiedKFold(y, n_folds=n_folds,
                                              shuffle=True, random_state=0)

        cv_results = {'score': np.zeros((n_folds,)),
                      'tpr': np.zeros((n_labels, n_folds)),
                      'fpr': np.zeros((n_labels, n_folds,))}

        for i, (train_ind, test_ind) in enumerate(cv):

            print("Fold {}/{}".format(i+1, n_folds))

            model.fit(X[train_ind, :], y[train_ind])

            cv_results['score'][i] = model.score(X[test_ind], y[test_ind])

            y_hat = model.predict(X[test_ind, :])
            tpr, fpr = compute_error_rates(y[test_ind], y_hat, all_labels)
            cv_results['tpr'][:, i] = tpr
            cv_results['fpr'][:, i] = fpr

        print "Cross-validation results:"
        for k in ['score', 'tpr', 'fpr']:
            print "{}: {:.2f} += {:.2f}".format(k, np.mean(cv_results[k]),
                                                np.std(cv_results[k]))

    else:
        cv_results = None

    # ----- fit model using all data (and do in-sample prediction) -----

    model.fit(X, y)
#    clf = model.best_estimator_
    y_pred = model.predict(X)
    if isinstance(model, LinearSVM):
        print "Best misclassification cost:", model.best_param_

    tp_rate, fp_rate = compute_error_rates(y, y_pred, all_labels)

    print "in-sample prediction error rate:", np.mean(y_pred != y)

    for i, g in enumerate(all_labels):
        print "label {} ({}): TPR={:.2f}, FPR={:.2f}".format(
            g, groups[g], tp_rate[i], fp_rate[i])

    # ----- save results -----
    result_file = op.join(dest_path, 'classification_results.npz')
    np.savez(result_file,
             rec_paths=rec_paths,
             run_cv=run_cv,
             samplerate=samplerate,
             cv_results=cv_results,
             tp_rate=tp_rate,
             fp_rate=fp_rate,
             estimator=model,
             groups=groups,
             y=y.astype(np.uint8),
             y_pred=y_pred.astype(np.uint8),
             frequency=freqs,
             n_channels=n_channels)

    # ----- show in-sample predictions -----
    fig, axarr = plt.subplots(nrows=2, ncols=1, sharex=True)

    ax = axarr[0]
    t = np.arange(y.shape[0]) / float(fs)
    ax.plot(t, y, 'r-', lw=2, label='label')
    ax.plot(t, y_pred, '--', color=3*[.25], lw=1, label='pred')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Label')
    ax.set_xlim(t[0], t[-1])
    ax.legend(loc='best')

    ax = axarr[1]
    ax.plot(t, y_pred != y, 'k-', lw=1)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Misclassified')
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(0, 1.2)

    for ax in axarr.flat:

        # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # Only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')

        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.xaxis.label.set_fontsize(9)
        ax.xaxis.label.set_fontname('Arial')
        ax.yaxis.label.set_fontsize(9)
        ax.xaxis.label.set_fontname('Arial')

        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))

    fig.set_size_inches(7, 3.5)
    fig.subplots_adjust()

    if dest_path is not None:
        for ff in fig_formats:
            fig.savefig(op.join(dest_path, 'prediction_insample.' + ff),
                        format=ff)

    # ----- show separating hyperplanes -----
    n_classes = len(model.classes_)
    colors = [llutil.get_nice_color('blue'),
              llutil.get_nice_color('red'),
              llutil.get_nice_color('gray')]

    if n_classes > 2:
        # one hyperplane per class for OVR classifier
        W = model.coef_
    else:
        # single hyperplance for binary classifier
        W = np.vstack((model.coef_, model.coef_))

    fig, axarr = plt.subplots(nrows=n_classes, ncols=1,
                              sharex=True, sharey=True)

    for i, cls in enumerate(model.classes_):

        ax = axarr[i]
        w = W[i, :]
        n = w.shape[0] / n_channels
        for j in range(n_channels):
            ax.plot(freqs, w[j*n:(j+1)*n], '-', color=colors[j])

        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Weight')
        ax.set_title('class {} ({})'.format(int(cls), groups[cls]))

        llutil.set_font_axes(ax, add_size=2)
        llutil.simple_xy_axes(ax)

    fig.set_size_inches(7, 2*n_classes)
    fig.tight_layout()
    if dest_path is not None:
        for ff in fig_formats:
            fig.savefig(op.join(dest_path, 'classifier_weights.' + ff),
                        format=ff)

    # ----- error rates -----
    n_rows = 1+int(run_cv)
    fig, axarr = plt.subplots(nrows=n_rows, ncols=1,
                              sharex=True, sharey=True)
    axarr = np.atleast_1d(axarr)

    xticklabels = [groups[k].replace(' ', '\n') for k in all_labels]
    for i in range(n_rows):

        if i == 0:
            tpr = tp_rate
            fpr = fp_rate
            title = 'in-sample'
        else:
            tpr = np.mean(cv_results['tpr'], axis=1)
            fpr = np.mean(cv_results['fpr'], axis=1)
            title = 'cross-validated'

        ax = axarr[i]
        ax.set_title(title)
        ax.bar(all_labels, tpr, width=0.35, color=3*[.25], label='TPR')
        ax.bar(all_labels + 0.35, fpr, width=0.35, color=3*[.75],
               label='FPR')

        ax.set_xticks(all_labels + .35)
        ax.set_xticklabels(xticklabels)
        ax.set_ylabel('Rate')

        ax.set_xlim(0, all_labels.max() + 1)
        ax.set_ylim(0, 1.2)
        ax.axhline(1, linestyle='-', color=3*[0], linewidth=1.5)
        ax.axhline(.9, linestyle='--', color=3*[.5])
        ax.axhline(.1, linestyle='--', color=3*[.5])

        ax.legend(loc='best')

        llutil.set_font_axes(ax, add_size=2)
        llutil.simple_xy_axes(ax)

    fig.set_size_inches(7, 5)
    fig.tight_layout()

    if dest_path is not None:
        for ff in fig_formats:
            fig.savefig(op.join(dest_path, 'error_rates.' + ff),
                        format=ff)

    plt.show()


def summarize_results(model_path, dest_path=None, fig_formats=['pdf']):

    if dest_path is None:
        dest_path = op.split(model_path)[0]

    results = np.load(model_path)

    y_pred = results['y_pred']
    y = results['y']
    fs = results['samplerate']
    model = results['estimator'].item()
    cv_results = results['cv_results'].item()

    if 'frequency' in results:
        freqs = results['frequency']
    else:
        feature_extractor = STFTFeatureExtractor(nfft=2*256, shift=1,
                                                 samplerate=fs,
                                                 f_lower=0, f_upper=10.,
                                                 winlen=2*256)
        freqs = feature_extractor.get_frequencies()

    if 'n_channels' in results:
        n_channels = results['n_channels']
    else:
        n_channels = 3

    all_labels = np.unique(MOUSE_GROUPS_SIMPLE.keys())

    # ----- show in-sample predictions -----

    fig, ax = plt.subplots()

    t = np.arange(y.shape[0]) / float(fs)
    ax.plot(t, y, 'r-', lw=1, label='Label')
    ax.plot(t, y_pred, '-', color=3*[.25], lw=.5, label='Predicted')
    ax.set_xlabel('Time (s)')
    ax.set_xlim(t[0], t[-1])
    ax.legend(loc='best', fontsize=8)
    ax.set_yticks(all_labels)
    ax.set_yticklabels([MOUSE_GROUPS_SIMPLE[k] for k in all_labels])

    llutil.set_font_axes(ax, add_size=2)
    llutil.simple_xy_axes(ax)

    fig.set_size_inches(10, 3)
    fig.tight_layout()
    if dest_path is not None:
        for ff in fig_formats:
            fig.savefig(op.join(dest_path, 'prediction_insample.' + ff),
                        format=ff)

    # ----- show separating hyperplanes -----
    n_classes = len(model.classes_)
    colors = [llutil.get_nice_color('blue'),
              llutil.get_nice_color('red'),
              llutil.get_nice_color('gray')]

    if n_classes > 2:
        # one hyperplane per class for OVR classifier
        W = model.coef_
    else:
        # single hyperplance for binary classifier
        W = np.vstack((model.coef_, model.coef_))

    fig, axarr = plt.subplots(nrows=n_classes, ncols=1,
                              sharex=True, sharey=True)

    for i, cls in enumerate(model.classes_):

        ax = axarr[i]
        w = W[i, :]
        n = w.shape[0] / n_channels
        for j in range(n_channels):
            ax.plot(freqs, w[j*n:(j+1)*n], '-', color=colors[j],
                    lw=1.5)

        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Weight')
        ax.set_title('class {} ({})'.format(int(cls),
                                            MOUSE_GROUPS_SIMPLE[cls]))

        llutil.set_font_axes(ax, add_size=2)
        llutil.simple_xy_axes(ax)
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))
        ax.set_ylim(W.min(), W.max())

    fig.set_size_inches(5, 1.1*n_classes)
    fig.tight_layout()
    if dest_path is not None:
        for ff in fig_formats:
            fig.savefig(op.join(dest_path, 'classifier_weights.' + ff),
                        format=ff)

    # ----- error rates -----
    fig, ax = plt.subplots()

    xticklabels = [MOUSE_GROUPS_SIMPLE[k].replace(' ', '\n')
                   for k in all_labels]
    tpr = np.mean(cv_results['tpr'], axis=1)
    fpr = np.mean(cv_results['fpr'], axis=1)

    ax.bar(all_labels, tpr, width=0.35, color=3*[.2], label='TPR')
    ax.bar(all_labels + 0.35, fpr, width=0.35, color=3*[.75],
           label='FPR')

    ax.set_xticks(all_labels + .35)
    ax.set_yticks([0, .25, .5, .75, 1])
#        ax.set_xticklabels([groups[k] for k in all_labels])
    ax.set_xticklabels(xticklabels)
    ax.set_ylabel('Rate')

    ax.set_xlim(0, all_labels.max() + 1)
    ax.set_ylim(0, 1.2)
    ax.axhline(1, linestyle='-', color=3*[0], linewidth=1.5)
    ax.axhline(.9, linestyle='--', color=3*[.5])
    ax.axhline(.1, linestyle='--', color=3*[.5])

    ax.legend(loc='best', fontsize=8)

    llutil.set_font_axes(ax, add_size=3)
    llutil.simple_xy_axes(ax)

    fig.set_size_inches(7, 3)
    fig.tight_layout()

    if dest_path is not None:
        for ff in fig_formats:
            fig.savefig(op.join(dest_path, 'error_rates.' + ff),
                        format=ff)

    plt.show()


def classify_recordings(model_path, rec_paths, dest_path=None):

    # load trained model
    model_data = np.load(model_path)

    estimator = model_data['estimator'].item()
    samplerate = model_data['samplerate'].item()
    groups = model_data['groups'].item()

    feature_extractor = STFTFeatureExtractor(nfft=2*256, shift=1,
                                             samplerate=samplerate,
                                             f_lower=0, f_upper=10.,
                                             winlen=2*256)

    # load and convert data
    for rec_path in rec_paths:

        X, y, fs, n_channels, groups = load_data([rec_path], feature_extractor,
                                                 samplerate=samplerate)
        assert fs == samplerate

        y_pred = estimator.predict(X)
        times = np.arange(y_pred.shape[0]) / float(samplerate)

        if dest_path is None:
            result_file = op.join(rec_path, 'classified_behavior.npz')
        else:
            result_file = op.join(dest_path, 'classified_behavior.npz')

        print "saving data to {}".format(result_file)
        np.savez(result_file, times=times, pred_labels=y_pred,
                 labels=groups, samplerate=samplerate,
                 algorithm=estimator.__class__.__name__,
                 features=feature_extractor.__class__.__name__)


@click.command(name='train')
@click.argument('paths', nargs=-1)
@click.option('--database', '-d', default=None)
@click.option('--validate', '-v', is_flag=True)
@click.option('--samplerate', '-s', default=100.)
@click.option('--output', '-o', default=None)
@click.option('--algorithm', '-a', default='SVM')
@click.option('--cluster', '-c', is_flag=True)
@click.option('--upper', '-u', default=10., type=float)
def cli_train(paths=None, database=None, validate=False,
              output=None, samplerate=100., algorithm='SVM',
              cluster=False, upper=10., **kwargs):
    """run classification pipeline"""

    assert paths is not None

    if len(paths) == 1 and op.isfile(paths[0]):
        rec_paths = get_annotated_recordings_from_file(paths[0],
                                                       database=database)
    else:
        rec_paths = []
        for p in paths:
            rec_paths.extend(get_annotated_recordings(p))

    train_classifier(rec_paths, run_cv=validate, samplerate=samplerate,
                     dest_path=output, algorithm=algorithm,
                     use_cluster=cluster, f_upper=upper)


@click.command(name='classify')
@click.argument('model_path', type=click.Path(exists=True))
@click.argument('paths', nargs=-1)
@click.option('--recursive', '-r', is_flag=True)
@click.option('--output', '-o', default=None)
def cli_classify(model_path=None, paths=None, output=None,
                 recursive=False, **kwargs):
    """run classification pipeline"""

    assert paths is not None

    rec_paths = []
    for p in paths:
        rec_paths.extend(get_recordings(p))

    classify_recordings(model_path, rec_paths, dest_path=output)


@click.command(name='summarize')
@click.argument('model_path', type=click.Path(exists=True))
@click.argument('paths', nargs=-1)
@click.option('--recursive', '-r', is_flag=True)
@click.option('--output', '-o', default=None)
def cli_summarize(model_path=None, paths=None, output=None,
                  recursive=False, **kwargs):
    """run classification pipeline"""

    assert paths is not None

    rec_paths = []
    for p in paths:
        rec_paths.extend(get_recordings(p))

    summarize_results(model_path, dest_path=output)


@click.group()
def cli():
    pass


cli.add_command(cli_train)
cli.add_command(cli_classify)
cli.add_command(cli_summarize)


if __name__ == '__main__':
    cli()
