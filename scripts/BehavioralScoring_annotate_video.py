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
"""

from __future__ import print_function

from six import string_types
import os
import os.path as op
import sys
import glob
import numpy as np
from scipy import ndimage
from scipy import signal
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

print("Using opencv version:", CV_VERSION)


def read_frames(frame_path, start_index=0, subsample=2,
                max_num_frames=np.Inf,
                use_memmap=True,
                force_reload=False):

    ts_file = glob.glob(op.join(frame_path, '*.csv'))
    if len(ts_file) > 0:
        ts_info = np.loadtxt(ts_file[0], delimiter=',')
        print("Reading timestamps from csv file")
    else:
        ts_info = None
        print("Extracting timestamps from file names")

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

    def __init__(self, video_file,
                 timestamps=None,
                 annotations=None,
                 fps=30.,
                 subsample=2,
                 max_num_frames=np.Inf,
                 use_memmap=True,
                 force_reload=False,
                 output_format='csv'):

        vid_format = op.splitext(video_file)[1]
        if vid_format == '.h264' and timestamps is None:
            raise ValueError('h264 format requires timestamps')

        self.video_file = video_file
        self.timestamps = timestamps
        self.annotations = annotations
        self.fps = fps
        self.subsample = subsample
        self.max_num_frames = max_num_frames
        self.use_memmap = use_memmap
        self.force_reload = force_reload
        self.output_format = output_format

        self._fp = None

    def prepare(self):

        video_path = op.split(self.video_file)[0]
        filename = op.splitext(op.split(self.video_file)[1])[0]

        # check video timestamps
        ts = self.timestamps
        if isinstance(ts, string_types) and op.exists(ts):
            self.timestamp_file = ts
            ext = op.splitext(ts)[1]
            if ext == '.txt':
                ts = np.loadtxt(ts)
            elif ext == '.csv':
                ts = np.loadtxt(ts, delimiter=',')
            elif ext == '.npy':
                ts = np.load(ts)
            else:
                raise ValueError('invalid timestamp file extension: ' + ext)

        else:
            self.timestamp_file = op.join(video_path,
                                          filename + '_timestamps.' +
                                          self.output_format)

        # check annotations
        annot = self.annotations
        if isinstance(annot, string_types) and op.exists(annot):
            annot_file = annot
            ext = op.splitext(annot)[1]
            if ext == '.txt':
                annot = np.loadtxt(annot)
            elif ext == '.csv':
                annot = np.loadtxt(annot, delimiter=',')
            elif ext == '.npy':
                annot = np.load(annot)
            else:
                raise ValueError('invalid annotation file extension: ' + ext)

            self.annotations = annot
            self.annotation_file = annot_file
        else:
            self.annotation_file = op.join(video_path,
                                           filename + '_annotations.' +
                                           self.output_format)
            if annot is None:
                self.annotations = np.NaN * np.zeros((ts.shape[0]),
                                                     dtype=np.int)

        # save frames to memory-mapped file
        memmap_file = op.join(video_path, filename + '.memmap')
        param_file = op.join(video_path, filename + 'memmap.params')

        if not op.exists(memmap_file) or self.force_reload:

            import imageio

            fp = None

            frame_cnt = 0
            _fps = None
            with imageio.get_reader(self.video_file, 'ffmpeg') as reader:

                vid_format = op.splitext(self.video_file)[1]
                if vid_format != '.h264':
                    meta = reader.get_meta_data()
                    fps = meta['fps']
                    n_frames = meta['nframes']
                else:
                    n_frames = len(ts)

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
                    frame_cnt += 1

                    if i+1 >= self.max_num_frames:
                        break

                del fp  # -> flush

                if _fps is not None:
                    fps = _fps
                else:
                    fps = self.fps

                if ts is None:
                    ts = np.arange(frame_cnt) / float(fps)

            with open(param_file, 'w') as pf:
                pickle.dump({'shape': shape,
                             'ts': ts,
                             'fps': self.fps}, pf)

        else:
            with open(param_file, 'r') as f:
                params = pickle.load(f)
                shape = params['shape']
                ts = params['ts']
                fps = params['fps']

        self.memmap_file = memmap_file
        self.param_file = param_file

        self._fp = np.memmap(memmap_file,
                             dtype=np.uint8,
                             mode='r',
                             shape=shape)
        self.timestamps = ts
        self.fps = fps

    def save(self):

        print("Saving annotations to file:", self.annotation_file)
        print("Saving timestamps to file:", self.timestamp_file)

        if self.output_format == 'csv':
            np.savetxt(self.annotation_file, self.annotations, fmt='%d')
            np.savetxt(self.timestamp_file, self.timestamps)

        elif self.output_format == 'npy':
            np.save(self.annotation_file, self.annotations)
            np.save(self.timestamp_file, self.timestamps)

    def close(self):

        if self._fp is not None:

            del self._fp
            self._fp = None

    def cleanup(self):

        for f in [self.memmap_file,
                  self.param_file]:

            if op.exists(f):
                print("Removing file: {}".format(f))
                os.remove(f)

    def __del__(self):

        self.close()

    def get_data(self, index):

        return self._fp[index, :, :], self.timestamps[index]


class LfpDataHandler():
    """simple class to hold LFP data

        Assumed data format:
            - numpy npz file with variables:
            - data: array with LFP data [num_channels x observations]
            - samplerate: sampling frequency (in Hz)

        If there are multiple LFP channels the mean across all channels will
        be used with subsequent lowpass filtering (and optionally subsampling)
    """

    def __init__(self, lfp_file, samplerate=50., f_lowpass=10):

        assert op.exists(lfp_file)

        self.lfp_file = lfp_file

        try:
            data = np.load(lfp_file)
            D = data['data']
            fs = float(data['samplerate'])

            self.samplerate = fs
            if D.ndim > 1:
                D = np.mean(D, axis=0)

            Wn = f_lowpass / fs * 2.
            b, a = signal.butter(4, Wn, btype='lowpass', analog=False,
                                 output='ba')
            D = signal.filtfilt(b, a, np.abs(D), axis=0)

            subsample = int(round(fs / samplerate))
            D = D[::subsample]
            ts = np.arange(D.shape[0]) / (fs / subsample)

            self.data = D
            self.timestamps = ts
            self.samplerate = fs / subsample

        except BaseException:
            print("Could not load LFP data from file:", lfp_file)
            print("Reason:")
            traceback.print_exc()
            self.data = None
            self.samplerate = None

    def is_valid(self):
        return self.data is not None


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
                 n_states=10,
                 state_min=0,
                 width=5, height=8, dpi=100,
                 static_data=None,
                 overlay_data=None):

        super(MPLWidget, self).__init__()

        self.parent = parent

        self.timestamps = timestamps
        self.n_states = n_states
        self.state_min = state_min
        self.static_data = static_data
        self.overlay_data = overlay_data

        self.fig = Figure((width, height), dpi=dpi)
        self.ax = self.fig.add_subplot(111)

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
        for i in range(self.n_states):
            self.lines[i] = ax.plot(x, y, 'ko-')[0]

        self.pos_line = ax.plot([0, 0], [0, self.n_states],
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
        ax.set_ylabel('Assigned state')
        ax.set_ylim(y_min, self.n_states)

        ax.set_yticks(np.arange(self.state_min, self.n_states))
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

        for i in range(self.n_states):

            yi = np.ma.masked_where(y != i, y)
            self.lines[i].set_data(x, yi)
            self.ax.draw_artist(self.lines[i])

        if pos >= 0:

            self.pos_line.set_data(2*[x[pos]], [0, self.n_states])
            self.ax.draw_artist(self.pos_line)

        self.fig.canvas.draw()

    def update_plot(self):

        for i in range(self.n_states):
            self.ax.draw_artist(self.lines[i])

        self.ax.draw_artist(self.pos_line)
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

    # default behavioral states
    _default_states = {-1: 'Invalid',
                       0: 'Unclassified',
                       1: 'Active exploration',
                       2: 'Quiet wakefulness',
                       3: 'Grooming',
                       4: 'Rearing',
                       5: 'Eating',
                       6: 'Sleeping'}

    """
    Dahwale et al. eLife 2017;6:e27702
        grooming, eating, active exploration, task engagement, quiet
        wakefulness, rapid eye movement, slow wave sleep

    Venkatraman et al. J Neurophysiol 104(1):569-75
        Grooming, Resting, Eating, Rearing
    """

    # assigned modes
    MODE_PLAYING = 0
    MODE_ASSIGNING = 1

    def __init__(self, video_handler, lfp_handler=None,
                 states=None, win_name="annotation window"):

        qw.QWidget.__init__(self)

        self.video_handler = video_handler
        self.lfp_handler = lfp_handler
        self.win_name = win_name  # for opencv window

        if states is None:
            states = self._default_states
        self.states = states

        self.current_frame = 0
        self.current_state = 0
        self.frame_rate = 10
        self.mode = self.MODE_PLAYING
        self.slider_is_moving = False

        self.thread = None

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
        static_data = None

        if lfp_handler is not None:
            static_data.append(lfp_handler)

        self.plot_widget = MPLWidget(self,
                                     timestamps=video_handler.timestamps,
                                     n_states=len(self.get_states()),
                                     state_min=min(self.get_states()),
                                     width=5, height=4, dpi=100,
                                     static_data=static_data)
        self.plot_widget.labels_changed.connect(self.labels_changed)
        self.plot_widget.update_figure()
        mainbox.addWidget(self.plot_widget)

        self.plot_widget.labels_changed.connect(self.labels_changed)
        self.plot_widget.update_figure()
        mainbox.addWidget(self.plot_widget)

        mainbox.addStretch()
        self.setLayout(mainbox)
        self.setGeometry(100, 100, 500, 1000)

        try:
            cv2.namedWindow(win_name,
                            cv2.WINDOW_OPENGL | cv2.WND_PROP_ASPECT_RATIO)

            print("Using window with OpenGL support")

        except BaseException:
            cv2.namedWindow(win_name,
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

        if self.video_handler is None:
            return -1
        else:
            return len(self.video_handler.timestamps)

    def fps_slider_changed(self):

        fps = self.fps_slider.value()
        self.frame_rate = fps
        if self.handler is not None:
            self.handler.set_frame_rate(fps)

    def pos_slider_changed(self):

        pos = self.pos_slider.value()

        if self.mode == self.MODE_PLAYING:
            self.current_state = self.video_handler.annotations[pos]
        else:
            self.video_handler.annotations[pos] = self.current_state

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

        self.current_state = index
        self.update_state_label()
        self.mode = self.MODE_ASSIGNING

    def labels_changed(self, ind, labels):

        if len(ind) > 0:

            self.video_handler.annotations[ind] = labels

            if self.handler is not None and \
                    self.handler.get_status() != PlaybackHandler.RUNNING:

                self.update_gui(self.handler.current_frame)

            self.plot_widget.update_figure()

    def shortcut_pressed(self, name):

        pass

    def update_state_label(self):

        name = self.get_current_state_name()
        self.state_label.setText('<b>Current state: {}</b>'.format(name))

    def create_button_grid(self, buttons_per_row=3):

        grid = qw.QGridLayout()
        grid.setSpacing(10)

        states = self.get_states()
        state_names = self.get_state_names()
        n_states = len(state_names)

        n_rows = int(np.ceil(n_states / float(buttons_per_row)))
        y_offset = 0

        for i in range(n_rows):
            for j in range(buttons_per_row):

                index = i * buttons_per_row + j
                if index < n_states:

                    gid = states[index]
                    name = state_names[index]
                    label = '%d &%s' % (gid, name)

                    button = qw.QPushButton(label)
                    button.clicked.connect(partial(self.button_clicked,
                                                   states[index]))
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

    def get_states(self):

        return sorted(self.states.keys())

    def get_state_names(self):

        return [self.states[k] for k in self.get_states()]

    def get_current_state_name(self):

        return self.states[self.current_state]

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

        mat, ts = self.video_handler.get_data(x)
        cv2.imshow(self.win_name, mat / 255.)

        cv2.waitKey(1)

        if self.mode == self.MODE_ASSIGNING:
            self.video_handler.annotations[x] = self.current_state

        if update_slider:
            self.pos_slider.setValue(x)

        self.plot_widget.plot_labels(self.video_handler.annotations, pos=x)

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
        self.video_handler.save()

    def close_and_exit(self):

        self.cleanup()
        self.close()


@click.command()
@click.argument('video_file', type=click.Path(exists=True))
@click.option('--timestamp-file', '-t', default=None,
              help='A numpy (npy) or text file with video timestamps '
                   '(one entry per row)')
@click.option('--annotation-file', '-a', default=None,
              help='File with video annotations (one entry per video frame)')
@click.option('--fps', '-r', default=30.,
              help="frame rate of video file (required for h264 files)")
@click.option('--max-frames', '-n', default=np.Inf,
              help="The maximum number of frames. Default: Inf")
@click.option('--subsample', '-s', default=2,
              help="Subsample video data for faster processing")
@click.option('--force', '-f', is_flag=True,
              help="Force rewriting of memmap files if already exist")
@click.option('--keep', '-k', is_flag=True,
              help="Keep temporary memmap files after existing")
@click.option('--lfp-file', '-l', default=None,
              help='Numpy (npz) file with LFP data ("data" and "samplerate")')
@click.option('--format', '-F', default='csv',
              help='Output file format. Can be either "csv" or "npy". '
                   'Default: csv')
@click.option('--state-file', '-S', default=None,
              help='csv or npy file with behavioral states.'
                   'For csv format: the first column contains state label '
                   '(e.g. grooming) and the decond column name of the state. '
                   'For npy format: a dict where keys are state labels and '
                   'values represent state names. If no state file is '
                   'supplied default states will be used.')
def cli(video_file=None, timestamp_file=None, annotation_file=None,
        fps=None, max_frames=np.Inf, subsample=None, force=False, keep=False,
        lfp_file=None, format=None, state_file=None):

    print("Annotating video:", video_file)

    # wrap data into container
    video_handler = VideoDataHandler(video_file,
                                     fps=fps,
                                     timestamps=timestamp_file,
                                     annotations=annotation_file,
                                     max_num_frames=max_frames,
                                     force_reload=force,
                                     subsample=subsample,
                                     output_format=format)
    video_handler.prepare()

    if lfp_file is not None:
        lfp_handler = LfpDataHandler(lfp_file)
    else:
        lfp_handler = None

    if state_file is None:
        states = None
    else:
        assert op.exists(state_file), "state file not a valid file!"

        ext = op.splitext(state_file)
        if ext == '.csv':
            states = {}
            for line in np.genfromtxt(state_file, dtype=None, delimiter=','):
                states[int(line[0])] = str(line[1])

        elif ext == 'npz':
            states = np.load(state_file).item()

    app = qw.QApplication([])
    w = AnnotatorWidget(video_handler, lfp_handler, states=states)
    w.show()
    exit_status = app.exec_()

    if not keep:
        video_handler.cleanup()

    sys.exit(exit_status)


if __name__ == '__main__':
    cli()
