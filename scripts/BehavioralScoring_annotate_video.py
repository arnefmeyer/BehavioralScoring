#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Arne F. Meyer <arne.f.meyer@gmail.com>
# License: GPLv3

"""
    A simple Qt-based GUI to manually annotate recorded videos

    Note that not much attention has been payed to make it particularly
    thread-safe.

    Todo:
    - replace slow matplotlib-based class assignment panel by qt, pyqtgraph or
      similar
    - move to "widgets" submodule and pass functions or classes that determine
      how to read/convert the video data
"""

import os
import os.path as op
import sys
import glob
import numpy as np
from scipy import ndimage
from tqdm import tqdm
import cv2
import time
import collections
from functools import partial
import click
import pickle
import traceback
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg

try:
    from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg \
        as NavigationToolbar
except ImportError:
    from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT \
        as NavigationToolbar

try:
    from PyQt4 import QtGui, QtCore
    qw = QtGui
except ImportError:
    from PyQt5 import QtGui, QtCore
    import PyQt5.QtWidgets as qw

# workaround to make code work with opencv 2.4.x and 3.x
try:
    from cv2 import cv
    CV_VERSION = 2
except ImportError:
    cv = cv2
    cv.CV_WINDOW_NORMAL = cv2.WINDOW_NORMAL
    cv.CV_BGR2GRAY = cv2.COLOR_BGR2GRAY
    cv.CV_GRAY2RGB = cv2.COLOR_GRAY2RGB
    cv.CreateTrackbar = cv2.createTrackbar
    CV_VERSION = 3

print "Using opencv version:", CV_VERSION


def read_frames(frame_path, start_index=0, subsample=2,
                max_num_frames=np.Inf,
                use_memmap=True,
                force_reload=False):

    ts_file = glob.glob(op.join(frame_path, '*.csv'))
    if len(ts_file) > 0:
        ts_info = np.loadtxt(ts_file[0], delimiter=',')
        print "Reading timestamps from csv file"
    else:
        ts_info = None
        print "Extracting timestamps from file names"

    files = os.listdir(frame_path)
    files = [f for f in files if f.endswith('.jpg')]
    files.sort()

    images = None
    _memmap_file = None
    shape = None
    timestamps = None

    read_img = True
    _memmap_file = op.join(frame_path, 'frames.memmap')
    if use_memmap and op.exists(_memmap_file) and not force_reload:
        read_img = False

    counter = 0
    for i, f in enumerate(tqdm(files)):

        try:

            if read_img or i == 0:

                img = ndimage.imread(op.join(frame_path, f),
                                     flatten=True)

                img = img.astype(np.uint8)
                if img.shape == (960, 1280):
                    img = img[::subsample, ::subsample]

            if images is None:

                n_files = len(files)
                shape = (n_files, img.shape[0], img.shape[1])

                if use_memmap and read_img:
                    images = np.memmap(_memmap_file,
                                       dtype=np.uint8,
                                       mode='w+',
                                       shape=shape)
                else:
                    images = np.zeros(shape, np.uint8)

                timestamps = np.zeros((n_files,))

            if read_img:
                images[i, :, :] = img

            if ts_info is None:
                ts = int(op.splitext(f)[0].split('_')[-1])
            else:
                ts = ts_info[counter, 3]
            timestamps[i] = ts - start_index

            counter += 1

            if counter >= max_num_frames:
                break

        except BaseException:
            print("Couldn't read frame {}".format(f))
            traceback.print_exc()

    if use_memmap:
        # flush data
        del images

        images = np.memmap(_memmap_file,
                           dtype=np.uint8,
                           mode='r',
                           shape=shape)
        memmap_file = _memmap_file

    return {'images': images,
            'timestamps': np.asarray(timestamps),
            'memmap_file': memmap_file}


class VideoDataHandler():

    def __init__(self, frame_path,
                 start_index=0,
                 subsample=2,
                 max_num_frames=np.Inf,
                 use_memmap=True,
                 force_reload=False,
                 assigned_groups=None,
                 groups=None):

        self.frame_path = frame_path
        self.start_index = start_index
        self.subsample = subsample
        self.max_num_frames = max_num_frames
        self.use_memmap = use_memmap
        self.force_reload = force_reload
        self.assigned_groups = assigned_groups
        self.groups = groups

        self.timestamps = None
        self.start_time = None
        self.valid = None
        self.samplerate = None

        self._fp = None

    def prepare(self):

        ts_file = op.join(self.frame_path, 'timestamps.npz')
        ts_info = np.load(ts_file)
        dd = ts_info['hardware'].item()
        timestamps = dd['timestamps']
        fs = float(dd['samplerate'])
        start_time = dd['start_time']
        valid = ts_info['valid']

        # save frames to memory-mapped file
        memmap_file = op.join(self.frame_path, 'frames.memmap')
        param_file = op.join(self.frame_path, 'frames.params')

        if not op.exists(memmap_file):

            import imageio

            # TODO: do this in a separate function or even create classes to
            # parse different types of inputs
            mp4_file = glob.glob(op.join(self.frame_path, '*.mp4'))[0]

            fp = None
            n_frames = len(timestamps)

            with imageio.get_reader(mp4_file, 'ffmpeg') as reader:

                for i, frame in enumerate(tqdm(reader)):

                    frame = cv2.cvtColor(frame, cv.CV_BGR2GRAY)

                    if self.subsample > 1:
                        frame = frame[::self.subsample, ::self.subsample]

                    if fp is None:
                        shape = (n_frames, frame.shape[0], frame.shape[1])
                        fp = np.memmap(memmap_file,
                                       dtype=np.uint8,
                                       mode='w+',
                                       shape=shape)
                    fp[i, :, :] = frame

                    if i+1 >= n_frames:
                        break

                del fp

            with open(param_file, 'w') as pf:
                pickle.dump(shape, pf)

        else:
            with open(param_file, 'r') as f:
                shape = pickle.load(f)

        self.frame_rate = 1. / np.mean(np.diff(timestamps))
#        files = glob.glob(op.join(self.frame_path, '*.jpg'))
#        files.sort()
#        n_files = int(min(len(files), self.max_num_frames))
#        files = files[:n_files]
#
#        # processor start time and sample rate
#        ts_file = op.join(self.frame_path, 'timestamps.npz')
#        if op.exists(ts_file):
#            tmp = np.load(ts_file)['hardware'].item()
#            start_time = tmp['start_time']
#            fs = float(tmp['samplerate'])
#        else:
#            # load first timestamp from recording directory
#            start_times = np.load(op.join(self.frame_path, '..',
#                                          'start_times.npz'))['Node100'].item()
#            fs = float(start_times['samplerate'])
#            start_time = start_times['timestamp'] / fs
#
#        # load timestamps
#        csv_file = glob.glob(op.join(self.frame_path, '*.csv'))
#
#        if len(csv_file) > 0:
#            print "Reading timestamps from csv file"
#            ts_info = np.loadtxt(csv_file[0], delimiter=',')
#            timestamps = ts_info[:, 3] / fs - start_time
#
#        else:
#            print "Extracting timestamps from file names"
#            timestamps = np.zeros((n_files,))
#            for i, f in enumerate(files):
#                ts = int(op.splitext(f)[0].split('_')[-1])
#                timestamps[i] = ts / fs - start_time
#
#        timestamps = timestamps[:n_files]
#
#        # check for invalid (0 bytes) files
#        valid = np.ones((len(files),), dtype=np.bool)
#        for i, f in enumerate(files):
#            if op.getsize(f) == 0:
#                valid[i] = False
#
#        timestamps = timestamps[valid]
#        n_valid = np.sum(valid)
#
#        # save frames to memory-mapped file
#        memmap_file = op.join(self.frame_path, 'frames.memmap')
#        param_file = op.join(self.frame_path, 'frames.params')
#
#        if not op.exists(memmap_file):
#
#            fp = None
#            cnt = 0
#            for i, f in enumerate(tqdm(files)):
#
#                if valid[i]:
#
#                    img = ndimage.imread(f, flatten=True).astype(np.uint8)
#
#                    if self.subsample > 1:
#                        img = img[::self.subsample, ::self.subsample]
#
#                    shape = (n_valid, img.shape[0], img.shape[1])
#
#                    if fp is None:
#                        fp = np.memmap(memmap_file,
#                                       dtype=np.uint8,
#                                       mode='w+',
#                                       shape=shape)
#                    fp[cnt, :, :] = img
#                    cnt += 1
#
#            del fp
#
#            with open(param_file, 'w') as pf:
#                pickle.dump(shape, pf)
#
#        else:
#            with open(param_file, 'r') as f:
#                shape = pickle.load(f)

        self.memmap_file = memmap_file
        self.param_file = param_file

        self._fp = np.memmap(memmap_file,
                             dtype=np.uint8,
                             mode='r',
                             shape=shape)
        self.timestamps = timestamps
        self.samplerate = fs
        self.start_time = start_time
        self.valid = valid

    def close(self):

        if self._fp is not None:

            del self._fp
            self._fp = None

    def remove(self):

        for f in [self.memmap_file,
                  self.param_file]:

            if op.exists(f):
                print("Removing file: {}".format(f))
                os.remove(f)

    def __del__(self):

        self.close()

    def get_data(self, index):

        return self._fp[index, :, :], self.timestamps[index]


class AccelerometerDataHandler():

    def __init__(self, rec_path, samplerate=100.):

        from scipy import signal

        self.rec_path = rec_path
        self.samplerate = float(samplerate)

        tmp = np.load(op.join(rec_path, 'aux_channels.npz'))
        D = tmp['data']
        fs = tmp['samplerate']

        f_lower = 1.
        f_upper = .4 * samplerate
#        Wn = f_upper / fs * 2
        Wn = (f_lower / fs * 2, f_upper / fs * 2)
        b, a = signal.butter(2, Wn, btype='bandpass',
                             analog=False, output='ba')
        DD = signal.filtfilt(b, a, D, axis=0)

        n = int(round(fs / samplerate))
        A = np.sum(np.abs(DD[::n, :]), axis=1)
        ts = np.arange(A.shape[0]) / self.samplerate

        self.data = A
        self.timestamps = ts


class LfpDataHandler():

    def __init__(self, rec_path, samplerate=50.):

        from lindenlab.io import database as lldb
        from lindenlab.util import filter_data, compute_envelope

        self.rec_path = rec_path
        self.samplerate = float(samplerate)

        D, fs = lldb.load_lfp_data(rec_path, ignore_dead_channels=True)

        D = filter_data(D, fs, f_lower=150, filt_type='highpass')
        D = filter_data(compute_envelope(np.mean(D, axis=1)), fs,
                        f_upper=25., filt_type='lowpass')

        subsample = int(round(fs / samplerate))
        D = D[::subsample]
        ts = np.arange(D.shape[0]) / (fs / subsample)

        self.data = D
        self.timestamps = ts


class ClassifiedDataHandler():

    def __init__(self, rec_path):

        self.data = None
        self.timestamps = None

        self.file_path = op.join(rec_path, 'classified_behaviour.npz')

        if op.exists(self.file_path):

            tmp = np.load(self.file_path)
            self.timestamps = tmp['timestamps']
            self.data = tmp['pred_labels']


class MousecamDataHandler():

    def __init__(self, rec_path, subsample=2):

        self.rec_path = rec_path
        self.subsample = subsample

        self._fp = None
        self.video_file = None
        self.data = None

        self.load_data()

    def load_data(self):

        import imageio

        video_files = glob.glob(op.join(self.rec_path, '*eye*.h264'))

        if len(video_files) > 0:

            video_file = video_files[0]

            params = np.load(op.splitext(video_file)[0] + '.npz')
            timestamps = params['timestamps']

            memmap_file = op.splitext(video_file)[0] + '.memmap'
            n_frames = len(timestamps)
            sub = self.subsample
            if not op.exists(memmap_file):

                # open reader
                reader = imageio.get_reader(video_file, 'ffmpeg')

                # process frames
                fp = None
                size = None
                for i in tqdm(range(n_frames)):

                    frame = cv2.cvtColor(reader.get_data(i), cv.CV_BGR2GRAY)

                    if i == 0:
                        size = (n_frames, frame.shape[0]/sub,
                                frame.shape[1]/sub)
                        fp = np.memmap(memmap_file, dtype='uint8',
                                       mode='w+', shape=size)

                    fp[i, :, :] = frame[::sub,
                                        ::sub]

                # make sure to flush file
                del fp

            else:
                reader = imageio.get_reader(video_file, 'ffmpeg')
                frame = cv2.cvtColor(reader.get_data(0), cv.CV_BGR2GRAY)
                size = (n_frames, frame.shape[0]/sub, frame.shape[1]/sub)

            self.memmap_file = memmap_file
            self.timestamps = timestamps
            self._fp = np.memmap(memmap_file, dtype='uint8',
                                 mode='r', shape=size)

        else:
            self.data = None
            self.timestamps = None
            self.memmap_file = None
            self._fp = None

    def is_valid(self):

        return self._fp is not None

    def get_data(self, ts, tol=.05):

        ind = np.where(np.abs(self.timestamps - ts) <= tol)[0]

        if len(ind) > 0:
            return self._fp[int(np.round(np.mean(ind))), :, :]
        else:
            return None

    def close(self):

        pass

    def remove(self):

        if self.memmap_file is not None and op.exists(self.memmap_file):
            os.remove(self.memmap_file)


class PlaybackHandler(QtCore.QObject):

    # playback control states
    NOTHING = 0
    RUNNING = 1
    PAUSING = 2
    EXITING = 3

    finished = QtCore.pyqtSignal()
    updated = QtCore.pyqtSignal(int)

    def __init__(self, n_frames, frame_rate=10.):

        super(PlaybackHandler, self).__init__()

        self.n_frames = n_frames
        self.frame_rate = frame_rate
        self.status = self.NOTHING
        self.current_frame = 0
        self.mutex = QtCore.QMutex()

    @QtCore.pyqtSlot()
    def process(self):

        n_frames = self.n_frames

        while True:

            self.mutex.lock()
            status = self.status
            frame_rate = self.frame_rate
            self.mutex.unlock()

            if status == self.RUNNING:

                if self.current_frame < n_frames:

                    self.updated.emit(self.current_frame)
                    self.current_frame = min(self.current_frame + 1,
                                             n_frames - 1)

                time.sleep(1./frame_rate)

            else:

                if status == self.EXITING:
                    break
                else:
                    time.sleep(0.5)

        self.current_frame = 0

        self.finished.emit()

    def set_status(self, s):

        self.mutex.lock()
        self.status = s
        self.mutex.unlock()

    def get_status(self):

        self.mutex.lock()
        s = self.status
        self.mutex.unlock()

        return s

    def set_current_frame(self, x):

        self.mutex.lock()
        self.current_frame = x
        self.mutex.unlock()

    def set_frame_rate(self, x):

        self.mutex.lock()
        self.frame_rate = x
        self.mutex.unlock()


class MPLWidget(qw.QWidget):

    labels_changed = QtCore.pyqtSignal(np.ndarray, np.ndarray)

    def __init__(self, parent,
                 timestamps=None,
                 n_groups=10,
                 group_min=0,
                 width=5, height=8, dpi=100,
                 static_data=None,
                 overlay_data=None):

        super(MPLWidget, self).__init__()

        self.parent = parent

        self.timestamps = timestamps
        self.n_groups = n_groups
        self.group_min = group_min
        self.static_data = static_data
        self.overlay_data = overlay_data

        self.fig = Figure((width, height), dpi=dpi)
        self.ax = self.fig.add_subplot(111)
        self.ax.hold(True)

        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.setParent(self)
        self.canvas.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.canvas.setFocus()

        self.toolbar = NavigationToolbar(self.canvas, self)
        self.zoom_active = False
        for child in self.toolbar.children():
            if isinstance(child, qw.QToolButton) and child.text() == 'Zoom':
                child.toggled.connect(self.zoom_button_toggled)

        self.init_figure()

        self.canvas.mpl_connect('button_press_event', self.button_pressed)
        self.canvas.mpl_connect('button_release_event', self.button_released)
        self.canvas.mpl_connect('motion_notify_event', self.mouse_moved)

        vbox = qw.QVBoxLayout()
        vbox.addWidget(self.toolbar)
        vbox.addWidget(self.canvas)
        self.setLayout(vbox)

    def init_figure(self):

        ax = self.ax
        x = self.timestamps

        y = np.zeros_like(x)
        self.lines = {}
        for i in range(self.n_groups):
            self.lines[i] = ax.plot(x, y, 'ko-')[0]

        self.pos_line = ax.plot([0, 0], [0, self.n_groups],
                                '-', color=3*[.5])[0]

        if self.overlay_data is not None and \
                self.overlay_data.data is not None:
            ax.plot(self.overlay_data.timestamps,
                    self.overlay_data.data, '-', color=3*[.74], lw=.5)

        y_min = -1
        if self.static_data is not None:

            if not isinstance(self.static_data, list):
                static_data = [self.static_data]
            else:
                static_data = self.static_data

            colors = ['r', 'b', 'o']
            for i, dd in enumerate(static_data):
                x = dd.timestamps
                y = dd.data
                ax.plot(x, -2-i*2 + y / (4*y.std()), '-', color=colors[i],
                        alpha=.25)
                y_min -= 1.5

        ax.set_xlim(x.min()-1, x.max()+1)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Assigned group')
        ax.set_ylim(y_min, self.n_groups)

        ax.set_yticks(np.arange(self.group_min, self.n_groups))
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))

        # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # Only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')

        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.xaxis.label.set_fontsize(10)
        ax.xaxis.label.set_fontname('Arial')
        ax.yaxis.label.set_fontsize(10)
        ax.xaxis.label.set_fontname('Arial')
        for at in ax.texts:
            at.set_fontsize(10)
            at.set_fontname('Arial')

        self.fig.canvas.draw()

    def plot_labels(self, y, pos=-1):

        x = self.timestamps

        for i in range(self.n_groups):

            yi = np.ma.masked_where(y != i, y)
            self.lines[i].set_data(x, yi)
            self.ax.draw_artist(self.lines[i])

        if pos >= 0:

            self.pos_line.set_data(2*[x[pos]], [0, self.n_groups])
            self.ax.draw_artist(self.pos_line)

#        self.fig.canvas.blit(self.ax.bbox)
        self.fig.canvas.draw()

    def update_plot(self):

        for i in range(self.n_groups):
            self.ax.draw_artist(self.lines[i])

        self.ax.draw_artist(self.pos_line)

#        self.fig.canvas.blit(self.ax.bbox)
        self.fig.canvas.draw()

    def update_figure(self):

        self.fig.canvas.draw()

    def button_pressed(self, event):

        self.x_start = event.xdata
        self.y_start = event.ydata

    def zoom_button_toggled(self, event):

        self.zoom_active = event

    def button_released(self, event):

        if not self.zoom_active:

            x1 = self.x_start
            y1 = self.y_start
            x2 = event.xdata
            y2 = event.ydata

            if x1 is not None and x2 is not None and x1 != x2:

                if x1 > x2:
                    x1, x2 = x2, x1
                    y1, y2 = y2, y1

                slope = (y2 - y1) / (x2 - x1)
                intercept = y1 - slope*x1

                ts = self.timestamps
                ind = np.where(np.logical_and(ts >= x1, ts <= x2))[0]
                yy = ts[ind] * slope + intercept

                labels = np.asarray(np.round(yy), np.int)

                self.labels_changed.emit(ind, labels)

    def mouse_moved(self, event):

        pass


class AnnotatorWidget(qw.QWidget):

    # behavioral state groups
    _groups = {-1: 'Invalid',
               0: 'Unclassified',
               1: 'Active exploration',
               2: 'Quiet wakefulness',
               3: 'Grooming',
               4: 'Rearing',
               5: 'Eating',
               6: 'Sleeping'}

    """
    Dahwale et al. bioRxiv 2015:
        grooming, eating, active exploration, task engagement, quiet
        wakefulness, rapid eye movement, slow wave sleep

    Venkrataman et al. J Neurophysiol 2010:
        Grooming, Resting, Eating, Rearing
    """

    # assigned modes
    MODE_PLAYING = 0
    MODE_ASSIGNING = 1

    def __init__(self, rec_path,
                 max_num_frames=np.Inf, use_memmap=True,
                 force_reload=False,
                 keep_memmap_files=False,
                 win_name="annotation window",
                 show_accel_data=False):

        qw.QWidget.__init__(self)

        self.rec_path = rec_path
        self.max_num_frames = max_num_frames
        self.use_memmap = use_memmap
        self.force_reload = force_reload
        self.win_name = win_name
        self.keep_memmap_files = keep_memmap_files

        self.current_frame = 0
        self.current_group = 0
        self.frame_rate = 10
        self.mode = self.MODE_PLAYING
        self.slider_is_moving = False

        self.thread = None

        if show_accel_data:
            # useful for debugging purposes
            self.accel_data = AccelerometerDataHandler(rec_path)
        else:
            self.accel_data = None

        self.lfp_data = LfpDataHandler(rec_path)
        self.mousecam_data = MousecamDataHandler(rec_path)
        self.classified_data = ClassifiedDataHandler(rec_path)

        self.data = None
        self.read_data()

        self.setWindowTitle('Video Annotator')
        mainbox = qw.QVBoxLayout(self)

        # control buttons
        mainbox.addWidget(qw.QLabel("<b>Controls</b>", self))
        self.buttons = collections.OrderedDict()
        self.buttons['start'] = qw.QPushButton('Play', self)
        self.buttons['start'].clicked.connect(self.start)

        self.buttons['stop'] = qw.QPushButton('Stop', self)
        self.buttons['stop'].clicked.connect(self.stop)

        self.buttons['reset'] = qw.QPushButton('Reset', self)
        self.buttons['reset'].clicked.connect(self.reset)

        self.buttons['save'] = qw.QPushButton('Save', self)
        self.buttons['save'].clicked.connect(self.save_results)

        self.buttons['exit'] = qw.QPushButton('Exit', self)
        self.buttons['exit'].clicked.connect(self.close_and_exit)

        hbox = qw.QHBoxLayout(self)
        [hbox.addWidget(self.buttons[k]) for k in self.buttons]
        mainbox.addLayout(hbox)

        mainbox.addWidget(self.HLine())

        # fps slider
        mainbox.addWidget(qw.QLabel("<b>Frame rate</b>", self))
        self.fps_slider = qw.QSlider(QtCore.Qt.Horizontal)
        self.fps_slider.setMinimum(1)
        self.fps_slider.setMaximum(60)
        self.fps_slider.setValue(self.frame_rate)
        self.fps_slider.setTickPosition(qw.QSlider.TicksBelow)
        self.fps_slider.setTickInterval(10)
        self.fps_slider.sliderMoved.connect(self.fps_slider_changed)
        mainbox.addWidget(self.fps_slider)

        mainbox.addWidget(self.HLine())

        # position slider
        mainbox.addWidget(qw.QLabel("<b>Position</b>", self))
        self.pos_slider = qw.QSlider(QtCore.Qt.Horizontal)
        self.pos_slider.setMinimum(0)
        self.pos_slider.setMaximum(self.get_frame_count()-1)
        self.pos_slider.setValue(0)
        self.pos_slider.setTickPosition(qw.QSlider.TicksBelow)
        self.pos_slider.setTickInterval(int(self.get_frame_count() / 25.))
        self.pos_slider.valueChanged.connect(self.pos_slider_changed)
        self.pos_slider.sliderMoved.connect(self.pos_slider_moved)
        self.pos_slider.sliderPressed.connect(self.pos_slider_pressed)
        self.pos_slider.sliderReleased.connect(self.pos_slider_released)
        self.pos_slider.setTracking(True)
        mainbox.addWidget(self.pos_slider)

        mainbox.addWidget(self.HLine())

        # state buttons
        mainbox.addWidget(qw.QLabel("<b>Assign state</b>", self))
        grid = self.create_button_grid()
        mainbox.addLayout(grid)

        # show current state
        self.state_label = qw.QLabel("<b>Current state: Unclassified</b>",
                                     self)
        mainbox.addWidget(self.state_label)

        mainbox.addWidget(self.HLine())

        # behavioral state plot
        static_data = [self.lfp_data]
        if self.accel_data is not None:
            static_data.append(self.accel_data)

        self.plot_widget = MPLWidget(self,
                                     timestamps=self.data.timestamps,
                                     n_groups=len(self.get_groups()),
                                     group_min=min(self.get_groups()),
                                     width=5, height=4, dpi=100,
                                     static_data=static_data,
                                     overlay_data=self.classified_data)
        self.plot_widget.labels_changed.connect(self.labels_changed)
        self.plot_widget.update_figure()
        mainbox.addWidget(self.plot_widget)

        if show_accel_data:
            # accelerometer signal widget
            self.aux_widget = MPLWidget(self,
                                        timestamps=self.data.timestamps,
                                        n_groups=len(self.get_groups()),
                                        width=5, height=4, dpi=100)

        self.plot_widget.labels_changed.connect(self.labels_changed)
        self.plot_widget.update_figure()
        mainbox.addWidget(self.plot_widget)

        mainbox.addStretch()
        self.setLayout(mainbox)
        self.setGeometry(100, 100, 500, 1000)

        try:
            cv2.namedWindow(win_name,
                            cv2.WINDOW_OPENGL | cv2.WND_PROP_ASPECT_RATIO)

            if self.mousecam_data.is_valid():
                cv2.namedWindow("mousecam",
                                cv2.WINDOW_OPENGL | cv2.WND_PROP_ASPECT_RATIO)

            print("Using window with OpenGL support")

        except BaseException:
            cv2.namedWindow(win_name,
                            cv2.WINDOW_NORMAL | cv2.WND_PROP_ASPECT_RATIO)

            if self.mousecam_data.is_valid():
                cv2.namedWindow("mousecam",
                                cv2.WINDOW_NORMAL | cv2.WND_PROP_ASPECT_RATIO)

            print("Using window without OpenGL support")

        self.start_thread()

    def __del__(self):

        self.cleanup()

    def cleanup(self):

        if self.thread is not None:
            self.handler.set_status(PlaybackHandler.EXITING)
            self.thread.terminate()
            self.thread.wait()
            self.thread = None

        self.data.close()
        self.mousecam_data.close()

        if not self.keep_memmap_files:

            self.data.remove()
            self.mousecam_data.remove()

    def closeEvent(self, event):

        self.cleanup()
        event.accept()

    def HLine(self):

        hline = qw.QFrame()
        hline.setFrameShape(qw.QFrame.HLine)
        hline.setFrameShadow(qw.QFrame.Sunken)

        return hline

    def resizeEvent(self, event):

        self.plot_widget.update_figure()

    def get_frame_count(self):

        if self.data is None:
            return -1
        else:
            return len(self.data.timestamps)

    def fps_slider_changed(self):

        fps = self.fps_slider.value()
        self.frame_rate = fps
        if self.handler is not None:
            self.handler.set_frame_rate(fps)

    def pos_slider_changed(self):

        pos = self.pos_slider.value()

        if self.mode == self.MODE_PLAYING:
            self.current_group = self.data.assigned_groups[pos]
        else:
            self.data.assigned_groups[pos] = self.current_group

            self.update_state_label()

    def pos_slider_moved(self):

        pos = self.pos_slider.value()
        self.update_gui(pos, update_slider=False)

        if self.handler is not None:
            self.handler.set_current_frame(pos)

    def pos_slider_pressed(self):

        self.mode = self.MODE_PLAYING

        if self.handler is not None:
            self.handler.set_status(PlaybackHandler.NOTHING)
            self.slider_is_moving = True

    def pos_slider_released(self):

        self.mode = self.MODE_PLAYING
        self.slider_is_moving = False

    def button_clicked(self, index):

        self.current_group = index
        self.update_state_label()
        self.mode = self.MODE_ASSIGNING

    def labels_changed(self, ind, labels):

        if len(ind) > 0:

            self.data.assigned_groups[ind] = labels

            if self.handler is not None and \
                    self.handler.get_status() != PlaybackHandler.RUNNING:

                self.update_gui(self.handler.current_frame)

            self.plot_widget.update_figure()

    def shortcut_pressed(self, name):

        pass

    def update_state_label(self):

        name = self.get_current_group_name()
        self.state_label.setText('<b>Current state: {}</b>'.format(name))

    def create_button_grid(self, buttons_per_row=3):

        grid = qw.QGridLayout()
        grid.setSpacing(10)

        groups = self.get_groups()
        group_names = self.get_group_names()
        n_groups = len(group_names)

        n_rows = int(np.ceil(n_groups / float(buttons_per_row)))
        y_offset = 0

        for i in range(n_rows):
            for j in range(buttons_per_row):

                index = i * buttons_per_row + j
                if index < n_groups:

                    gid = groups[index]
                    name = group_names[index]
                    label = '%d &%s' % (gid, name)

                    button = qw.QPushButton(label)
                    button.clicked.connect(partial(self.button_clicked,
                                                   groups[index]))
                    grid.addWidget(button, y_offset+i, j)

                    cb = partial(self.shortcut_pressed, name)
                    qw.QShortcut(QtGui.QKeySequence(name[0]),
                                 self, cb,
                                 context=QtCore.Qt.WidgetShortcut)
                    button.setFocus()

                else:
                    grid.addWidget(qw.QLabel(''), y_offset+i, j)

            y_offset += n_rows

        return grid

    def get_groups(self):

        return sorted(self._groups.keys())

    def get_group_names(self):

        return [self._groups[k] for k in self.get_groups()]

    def get_current_group_name(self):

        return self._groups[self.current_group]

    def read_data(self):

        frame_path = op.join(self.rec_path, 'frames')

        self.data = VideoDataHandler(frame_path,
                                     max_num_frames=self.max_num_frames,
                                     use_memmap=self.use_memmap,
                                     force_reload=self.force_reload,
                                     groups=self._groups)
        self.data.prepare()

        annot_path = op.join(self.rec_path, 'video_annotations.npz')
        if op.exists(annot_path):

            print("Reading existing annotations from {}".format(annot_path))
            annot = dict(**np.load(annot_path))

            data_updated = False

            if len(self.data.timestamps) != len(annot['timestamps']):
                # safeguard for compatibility with annotations created using
                # a previous version of this widget; classify all invalid
                # frames as -1 (= 'Invalid') and update the other variables
                N = self.data.timestamps.shape[0]
                groups = annot['assigned_groups']
                new_groups = -1 * np.ones((N,), dtype=groups.dtype)
                new_groups[self.data.valid] = groups
                annot['assigned_groups'] = new_groups
                annot['timestamps'] = self.data.timestamps
                annot['groups'] = self._groups

                data_updated = True

            if 'valid' not in annot:
                annot['valid'] = self.data.valid
                data_updated = True

            if data_updated:
                # rewrite updated annotations
                print "Updating video annotations file to new format"
                np.savez(annot_path, **annot)

            if np.all(self.data.timestamps == annot['timestamps']):
                labels = annot['assigned_groups']
                print "Done!"
            else:
                raise ValueError("Timestamps in video and annotations "
                                 "file differ")

        else:
            labels = np.zeros((self.data.timestamps.shape[0],),
                              dtype=np.int8)

        self.data.assigned_groups = labels

    def start_thread(self):

        self.thread = QtCore.QThread(objectName='PlaybackThread')
        self.handler = PlaybackHandler(self.get_frame_count(),
                                       frame_rate=self.frame_rate)
        self.handler.set_status(PlaybackHandler.NOTHING)
        self.handler.moveToThread(self.thread)
        self.handler.finished.connect(self.thread.quit)
        self.handler.updated.connect(self.update_gui)
        self.thread.started.connect(self.handler.process)
        self.thread.start()

    def update_gui(self, x, update_slider=True):

        mat, ts = self.data.get_data(x)
        cv2.imshow(self.win_name, mat / 255.)

        if self.mousecam_data.is_valid():

            frame = self.mousecam_data.get_data(ts)

            if frame is not None:
                cv2.imshow('mousecam', frame)

        cv2.waitKey(1)

        if self.mode == self.MODE_ASSIGNING:
            self.data.assigned_groups[x] = self.current_group

        if update_slider:
            self.pos_slider.setValue(x)

        self.plot_widget.plot_labels(self.data.assigned_groups, pos=x)

    def start(self):

        if self.handler is not None:
            self.handler.set_status(PlaybackHandler.RUNNING)

    def stop(self):

        if self.handler is not None:
            self.handler.set_status(PlaybackHandler.NOTHING)

    def reset(self):

        if self.handler is not None:

            self.handler.set_status(PlaybackHandler.NOTHING)
            self.handler.set_current_frame(0)
            self.mode = self.MODE_PLAYING
            self.update_gui(0, update_slider=True)

    def save_results(self):

        self.stop()
        time.sleep(0.25)

        result_file = op.join(self.rec_path, 'video_annotations.npz')
        print("Saving data to file: {}".format(result_file))
        np.savez(result_file,
                 timestamps=self.data.timestamps,
                 assigned_groups=self.data.assigned_groups,
                 groups=self._groups,
                 valid=self.data.valid)

    def close_and_exit(self):

        self.cleanup()
        self.close()


@click.command()
@click.argument('rec_path', type=click.Path(exists=True))
@click.option('--frames', '-F', default=np.Inf,
              help="The maximum number of frames. Default: Inf")
@click.option('--nomemmap', '-n', is_flag=True,
              help="Don't use memmap files as temporary frame storage")
@click.option('--force', '-f', is_flag=True,
              help="Force rewriting of memmap files that already exist")
@click.option('--keep', '-k', is_flag=True,
              help="Keep temporary memmap files after existing")
def cli(rec_path=None, frames=np.Inf, nomemmap=False, force=False, keep=False):

    print("Annotating recording: {}".format(rec_path))

    app = qw.QApplication([])
    w = AnnotatorWidget(rec_path,
                        max_num_frames=frames,
                        use_memmap=not nomemmap,
                        force_reload=force,
                        keep_memmap_files=keep)
    w.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    cli()
