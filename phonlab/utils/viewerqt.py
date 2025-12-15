__all__=['ViewerQt']

import sys
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
from matplotlib.widgets import Button
from matplotlib.patches import Rectangle
from matplotlib.collections import LineCollection
import subprocess
from .signal import loadsig
from phonlab import prep_audio
from phonlab import sgram
from time import time
import matplotlib.style as mplstyle
mplstyle.use('fast')

from PySide6 import QtCore, QtWidgets
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

def df2lines(df):
    lines = np.concat([df.t1.to_numpy(), [df.t2.iloc[-1]]])
    return lines

def df2texobj(df):
    return zip(
        df[['t1','t2']].mean(axis=1),
        df.iloc[:, 2]
    )

def make_linecollection(lines):
    lpts = [[(x,0), (x,1)] for x in lines]
    return LineCollection(lpts)

class ThrottledCanvas(FigureCanvas):
    """
    Adds:
     - request_draw() to coalesce draw calls to about 60 FPS via a QTimer.
     - simple blitting helpers for fast temporary-artist updates.
    """

    def __init__(self, fig, parent=None):
        super().__init__(fig)
        if parent is not None:
            self.setParent(parent)

        # throttled draw setup
        self._pending_draw = False
        self._draw_timer = QtCore.QTimer()
        self._draw_timer.setInterval(16)  # ~60 FPS
        self._draw_timer.timeout.connect(self._perform_draw)

        # blit caches
        self._blit_available = True
        self._background = None
        self._blit_artist = None

    def request_draw(self):
        """Coalesce multiple draw() calls into one (runs the timer)."""
        if not self._pending_draw:
            self._pending_draw = True
            # start (timer will call _perform_draw)
            self._draw_timer.start()

    def _perform_draw(self):
        """Actual draw invoked by the timer."""
        self._draw_timer.stop()
        self._pending_draw = False
        try:
            self.draw()
            # update cached background if blitting is used elsewhere
            if self._blit_artist is None:
                self._background = None
        except Exception:
            # fallback: attempt direct draw (rare)
            try:
                FigureCanvas.draw(self)
            except Exception:
                pass

    # ---- blit helpers (best-effort) ----
    def enable_blit_for_artist(self, artist):
        """Call after a full draw() to cache a background for blitting of `artist`."""
        try:
            # copy_from_bbox requires a renderer from the Agg backend
            self._blit_artist = artist
            self._background = self.copy_from_bbox(self.axes.bbox)
            self._blit_available = True
        except Exception:
            self._blit_available = False
            self._background = None
            self._blit_artist = None

    def blit_draw_artist(self, artist):
        """
        Fast update of single artist via blitting.
        If unsupported, raises Exception to let caller fall back.
        """
        if not self._blit_available or self._background is None:
            raise RuntimeError("Blit not available")

        # restore background, draw artist, and blit
        try:
            self.restore_region(self._background)
            artist.set_animated(True)
            self.axes.draw_artist(artist)
            self.blit(self.axes.bbox)
            artist.set_animated(False)
        except Exception as e:
            raise

    def clear_blit_cache(self):
        self._background = None
        self._blit_artist = None
        self._blit_available = False


class ViewerQt:
    def __init__(self, fn, df):
        self.start_x = None
        self.n_rows = 3
        lines = df2lines(df)
        self.lines = lines

        self.fig, self.axs = plt.subplots(nrows=self.n_rows, ncols=1)
        for i in range(1, self.n_rows):
            self.axs[i].sharex(self.axs[0])

        self.fn = fn
        wavdata, fs = loadsig(fn, chansel=[0])  # taking just the left channel, with 'chansel'
        decim = 10
        x = sp.signal.decimate(wavdata, decim)
        self.axs[0].plot(np.arange(0, x.size)/(fs/decim), x, c='k')
        self.axs[0].set_xlim([0, wavdata.size/fs])
        sgram(wavdata, fs, tf=8000, ax=self.axs[1])
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0, hspace=0)

        cid_keypress = self.fig.canvas.mpl_connect('key_press_event', self.on_keypress)
        cid_press = self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        cid_motion = self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        cid_release = self.fig.canvas.mpl_connect('button_release_event', self.on_release)

        l, b, w, h =  self.axs[self.n_rows-1].get_position().extents
        print(l, b, w, h)
        self.axprev = self.fig.add_axes([l, b, 0, 0.05])
        self.axcur = self.fig.add_axes([l, b, w-l,  0.05])
        self.axnext = self.fig.add_axes([l, b, 0, 0.05])
        self.bnext = Button(self.axnext, '')
        self.bnext.on_clicked(self.play_segment)
        self.bcur = Button(self.axcur, '')
        self.bcur.on_clicked(self.play_segment)
        self.bprev = Button(self.axprev, '')
        self.bprev.on_clicked(self.play_segment)

        self.tier_kwargs = dict(
            alpha=1,
            c="b",
            ls="-",
        )
        self.position_kwargs = dict(
            alpha=0.7,
            c="r",
            ls="--",
        )
        self.line_kwargs = dict(
            color="b",
            ls="-",
            lw=4,
        )
        self.span_kwargs = dict(
            alpha=0.3,
            ls="-",
        )
        self.seg_span_kwargs = dict(
            alpha=0.3,
            fill='g',
            ls="-",
        )
        self.seg_kwargs = dict(
            fill=False,
            edgecolor="b",
            lw=2,
            ls="-",
        )

        self.line_axs = list(range(len(self.axs)))
        self.cursor_lines = [
            a.axvline(1, 0, 1, **self.position_kwargs)
            for i, a in enumerate(self.axs)
            if i in self.line_axs
        ]
        self.set_active_line(0, visible=False)
        # collection of handles for selection span on axes
        self.current_span = [
            a.axvspan(1, 2, **self.span_kwargs)
            for i, a in enumerate(self.axs)
        ]
        # selected tier segment
        self.seg_span = self.axs[2].axvspan(-2,-1, **self.seg_span_kwargs)
        self.set_active_span(0,1,visible=False)
        self.set_active_tier_segment(-1)
        # format lims for tier axes
        self.axs[2].get_yaxis().set_visible(False)
        self.axs[2].get_yaxis().set_ticks([])
        # 
        # for t1 in self.lines:
        #     # self.axs[2].axvspan(t1, t2, -0.1, 1.02, **self.seg_kwargs)
        #     self.axs[2].axvline(t1, -0.1, 1.02, **self.line_kwargs)
        self.tier_lines = make_linecollection(self.lines)
        self.axs[2].add_collection(self.tier_lines)
        for t, txt in df2texobj(df):
            self.axs[2].text(t, 0.5, txt, ha="center")


        plt.show()

    ## methods for updating visuals when user interacts with figure

    def play_segment(self, event):
        xlims = self.axs[0].get_xlim()
        s1 = self.current_span[0].get_x()
        s2 = s1+self.current_span[0].get_width()
        if event.inaxes == self.axprev:
            print("prev")
            start_time = xlims[0]
            end_time = s1
        elif event.inaxes == self.axcur:
            print("cur")
            start_time = s1
            end_time = s2
        elif event.inaxes == self.axnext:
            print("next")
            start_time = s2
            end_time = xlims[1]
        else:
            return
        # print(f"{start_time=}")
        # print(f"{end_time=}")
        # this has problems because of seek position?
        subprocess.run(["ffplay", "-loglevel", "quiet", "-ss", f"{start_time}", "-t", f"{end_time}", "-nodisp", f"-autoexit", f"{self.fn}"])

    def set_active_tier_boundary(self, ind):
        """Click on a tier boundary to move it.
        """
        # set all segments to width=1
        lw = np.ones(len(self.lines), dtype=int) * 1
        # set selected segment style
        if ind >= 0:
            lw[ind] = 4
        self.tier_lines.set(linewidths=lw)

    def set_active_tier_segment(self, ind):
        """Click in a tier segment to higlight it.
        """
        # clear selection line
        # show tier selection
        if ind >= 0:
            self.seg_span.set_visible(True)
            self.seg_span.set_x(self.lines[ind])
            self.seg_span.set_width(self.lines[ind+1] - self.lines[ind])
        else:
            self.seg_span.set_visible(False)

    def set_active_span(self, l, w, visible=True):
        """Update active span across all axes.
        """
        # draw spans
        for a in self.current_span:
            a.set_visible(visible)
            a.set_x(l)
            a.set_width(w)

    def set_active_line(self, p1, visible=True):
        """Click on a signal to get value.
        Also allows adding/splitting segments.
        """
        # clear selection line
        for cur_line in self.cursor_lines:
            cur_line.set(visible=visible)
            cur_line.set_xdata([p1])

    def resize_play_buttons(self, s1, s2):
        l, b, w, h =  self.axs[self.n_rows-1].get_position().extents
        xsz = np.diff(self.axs[self.n_rows-1].get_xlim())[0]
        p1 =  ((s1/xsz)* (w-l))
        p2 =  ((s2/xsz)* (w-l))
        w3 = (xsz-s2)/xsz * (w-l)

        self.axprev.set_position([l, b, p1, .05])
        self.axcur.set_position([l+p1, b, p2-p1, .05])
        self.axnext.set_position([l+p2, b, w3, .05])

    def get_ind_under_point(self, event):
        """
        Return the index of the point closest to the event position or *None*
        if no point is within ``self.epsilon`` to the event position.
        """
        self.epsilon = 5  # in n_pixels, make dependant on zoom level?

        eventxt = self.tier_lines.get_transform().inverted().transform((event.x,event.y))
        xy = [(x,0) for x in self.lines]
        xyt = self.tier_lines.get_transform().transform(xy)  # to display coords
        d = np.array([x[0] for x in xyt]) - event.x
        ind = int(abs(d).argmin())
        if abs(d[ind]) < self.epsilon:
            return ind
        else:
            ind =  np.searchsorted([x[0] for x in xyt], event.x)
            return (ind-1, ind)

    ## event callbacks

    def on_keypress(self, event):
        print('press', event.key)
        sys.stdout.flush()
        if event.key == 'x':
            pass
            # visible = xl.get_visible()
            # xl.set_visible(not visible)
            # fig.canvas.draw()
        elif event.key == "ctrl+1":
            print(f"adding sement on tier 1 at {self.start_x=}")
            pos = np.searchsorted(self.lines, self.start_x)
            self.lines = np.insert(self.lines, pos, self.start_x)
            self.tier_lines.set_segments(lines)
            self.set_tier_labels(labels)

    def on_press(self, event):
        """Record the starting x-coordinate on button press."""
        # is click on existing:
        # tier axes
        if event.inaxes == self.axs[2]:
            ind = self.get_ind_under_point(event)
            # span boundary
            if isinstance(ind, int):
                p1 = self.lines[ind]
                self.set_active_tier_boundary(ind)
                self.set_active_line(event.xdata)
                self.set_active_tier_segment(-1)
                self.set_active_span(0, 0, False)
                self.resize_play_buttons(p1, p1)
            # tier span
            else:
                p1, p2 = self.lines[ind[0]], self.lines[ind[1]]
                # setspanhere
                self.set_active_tier_boundary(-1)
                self.set_active_tier_segment(ind[0])
                self.set_active_span(p1, p2-p1)
                self.resize_play_buttons(p1, p2)
        # sgram or signal
        elif event.inaxes == self.axs[0] or event.inaxes == self.axs[1]: 
            # print("setting line", event.xdata)
            # clear span
            self.set_active_tier_boundary(-1)
            self.set_active_span(0, 0, False)
            # draw selection line
            self.set_active_line(event.xdata)
            self.start_x = event.xdata
            self.resize_play_buttons(event.xdata, event.xdata)
        # for a in self.current_span:
        #     a.set_x(event.xdata)
        #     a.set_width(0)
        self.fig.canvas.draw_idle() # Redraw the canvas efficiently

    def on_motion(self, event):
        """Dynamically update the axvspan as the mouse moves (dragging)."""
        # if:
        # moving span
        # moving edge/point
        if self.start_x is None or event.inaxes != self.axs[0]:
            return

        end_x = event.xdata
        # Draw the new temporary vertical span
        x = min(self.start_x, end_x)
        w = abs(self.start_x - end_x)
        self.set_active_span(x, w)
        s1 = self.current_span[0].get_x()
        s2 = s1+self.current_span[0].get_width()
        self.resize_play_buttons(s1, s2)
        self.set_active_tier_segment(-1)
        # self.fig.canvas.draw_idle() # Redraw the canvas efficiently
        self.fig.canvas.draw_idle() # Redraw the canvas efficiently

    def on_release(self, event):
        """Finalize the axvspan on button release."""
        if self.start_x is None or event.inaxes != self.axs[0]: 
            return
        # self.selection_span.get_xy()[i, 0] = new_xmin if i in [0, 3] else new_xmax
        # s1 = self.current_span.get_x()
        # s2 = self.current_span.get_width()
        # p = self.current_span.get_xy()
        # print(f"{s1=} {s2=}")
        # print(f"{p=}")
        # The final span is left on the plot by the last on_motion call
        self.start_x = None

#
#

class WaveformCanvas(ThrottledCanvas):
    def __init__(self, parent=None, sync: TempLineSync = None):
        fig = Figure()
        super().__init__(fig, parent=parent)
        self.axes = fig.add_subplot(111)
        self.axes.set_autoscale_on(False)
        self.wave_x = None
        self.wave_y = None

        # temp line artist
        self._temp_line = None

        # callbacks
        self.time_selected_callback = None
        self.zoom_callback = None

        # sync object
        self.sync = sync

        # connect events
        self.mpl_connect("motion_notify_event", self._on_motion)
        self.mpl_connect("button_press_event", self._on_click)
        self.mpl_connect("scroll_event", self._on_scroll)

    def plot_waveform(self, data):
        self.wav_time = np.arange(data.shape[0])
        self.wav_vals = data
        self.ax_wav.clear()
        self.ax_wav.plot(self.wav_time, self.wav_vals, linewidth=0.8, antialiased=False)
        # fix y-limits to avoid autoscale cost
        self.ax_wav.set_ylim(np.min(self.wav_time), np.max(self.wav_time))
        # TODO: just use self._xlim?
        self.ax_wav.set_xlim([0, data.size/fs])
        # full redraw now
        self.draw()
        # allow blit caching for temp line after full draw
        self.request_blit_setup()

    def plot_sgram(self, data):
        # TODO: just plot directly?
        sgram(self.wav_vals, self.fs, tf=8000, ax=self.ax_sgram)
        self.draw()
        # allow blit caching for temp line after full draw
        self.request_blit_setup()

    def request_blit_setup(self):
        """After a full draw, try to create a blit cache for the temp line."""
        # attempt to create a temp line if none exists so blit can cache
        if self._temp_line is None:
            self._temp_line = self.axes.axvline(
                0, color="g", linestyle="--", linewidth=1, antialiased=False
            )
            # leave it invisible until used
            self._temp_line.set_visible(False)
            self.draw()
        # set up blit cache
        try:
            self.enable_blit_for_artist(self._temp_line)
        except Exception:
            # if blit setup fails, we'll fall back to throttled full draws
            self.clear_blit_cache()

    def set_temp_line(self, x):
        """Safe, full-redraw setter (fallback)."""
        if self._temp_line is None:
            self._temp_line = self.axes.axvline(
                x, color="g", linestyle="--", linewidth=1, antialiased=False
            )
        else:
            self._temp_line.set_visible(True)
            self._temp_line.set_xdata([x, x])
        # request a (throttled) redraw
        self.request_draw()

    def clear_temp_line(self):
        if self._temp_line:
            self._temp_line.set_visible(False)
            self.request_draw()

    def set_temp_line_blit(self, x):
        """
        Try to blit-update the temp line. If blit not available, fall back to set_temp_line.
        """
        if self._temp_line is None:
            # fallback: create the line and draw
            self.set_temp_line(x)
            return

        if not self._blit_available or self._background is None:
            # fallback to full redraw
            self.set_temp_line(x)
            return

        # use blit path
        self._temp_line.set_visible(True)
        self._temp_line.set_xdata([x, x])
        try:
            self.blit_draw_artist(self._temp_line)
        except Exception:
            # on any blit failure, disable blit cache and fallback
            self.clear_blit_cache()
            self.set_temp_line(x)

    # Event handlers
    def _on_motion(self, event):
        if event.inaxes != self.axes or event.xdata is None:
            return
        x = event.xdata
        # update locally and sync others
        try:
            self.set_temp_line_blit(x)
        except Exception:
            self.set_temp_line(x)

        if self.sync:
            self.sync.update_all(x, sender=self)

    def _on_click(self, event):
        if event.inaxes != self.axes or event.xdata is None:
            return
        idx = int(round(event.xdata))
        if self.time_selected_callback:
            self.time_selected_callback(idx)

    def _on_scroll(self, event):
        if self.zoom_callback:
            self.zoom_callback(event.xdata, event.step)




#
#

class MainWindow(QtWidgets.QWidget):
    def __init__(self, window_size=200):
        super().__init__()
        self.setWindowTitle("Waveform Viewer (PySide6)")
        self.window_size = window_size
        self.data_length = 0
        self._last_clicked_time = None

        # temp-line synchronizer
        self.sync = TempLineSync()


        v = QtWidgets.QVBoxLayout(self)
        # canvases
        self.canvas = WaveformCanvas(parent=self, sync=self.sync)
        # allow canvases to be registered
        self.sync.register(self.canvas)
        v.addWidget(self.canvas)
        self.scrollbar = QtWidgets.QScrollBar(QtCore.Qt.Horizontal)
        v.addWidget(self.scrollbar)

        # self.canvas.plot_waveform()
        # self.canvas.plot_sgram()

        # connect callbacks
        self.canvas.time_selected_callback = self._on_time_selected
        self.wave_canvas.zoom_callback = self._on_zoom

        # --- Scroll throttling (coalesce rapid events) ---
        self.pending_scroll = False
        self.last_scroll_value = None
        self._scroll_timer = QtCore.QTimer()
        self._scroll_timer.setInterval(16)  # 16ms ~60 FPS
        self._scroll_timer.timeout.connect(self._apply_scroll_update)
        # reduce spam: disable native tracking (some platforms emit many events while dragging)
        self.scrollbar.setTracking(False)
        # connect both sliderMoved (user drag) and valueChanged (keyboard/arrow/click)
        self.scrollbar.sliderMoved.connect(self._queue_scroll_update)
        self.scrollbar.valueChanged.connect(self._queue_scroll_update)

        # key focus
        self.setFocusPolicy(QtCore.Qt.StrongFocus)

    # def load_waveform(self, data):
    def init_scrollbar(self, data):
        """Load waveform (1D array)."""
        max_pos = max(0, self.data_length - self.window_size)
        self.scrollbar.setRange(0, max_pos)
        self.scrollbar.setPageStep(self.window_size)
        self.scrollbar.setSingleStep(1)
        self.scrollbar.setEnabled(True)
        # set initial x-limits
        self._set_xlim(0, min(self.window_size, self.data_length - 1))

        # Once plotted, give canvases a chance to setup blit caches
        self.canvas.request_blit_setup()

    def _set_xlim(self, xmin, xmax):
        """Apply same x-limits to both canvases (integers expected)."""
        xmin = max(0, int(xmin))
        xmax = max(xmin + 1, int(xmax))
        self.canvas.axes.set_xlim(xmin, xmax)
        self.canvas.request_draw()

        # update blit background caches if using blit
        try:
            self.canvas.request_blit_setup()
        except Exception:
            pass

    # key press: add a line at last clicked time
    def keyPressEvent(self, event):
        if self._last_clicked_time is not None and self.data_length > 0:
            x = max(0, min(self._last_clicked_time, self.data_length - 1))
            self.line_canvas.add_line(x)

    def _on_time_selected(self, t):
        self._last_clicked_time = t

    # ----------------------------------------
    # Scroll throttling (coalescing) methods
    # ----------------------------------------
    @QtCore.Slot(int)
    def _queue_scroll_update(self, value):
        """Queue a scroll update; actual apply runs at ~60 FPS."""
        self.last_scroll_value = int(value)
        if not self.pending_scroll:
            self.pending_scroll = True
            self._scroll_timer.start()

    @QtCore.Slot()
    def _apply_scroll_update(self):
        """Called by timer to apply the latest queued scroll value."""
        self._scroll_timer.stop()
        self.pending_scroll = False
        if self.last_scroll_value is None:
            return
        val = int(self.last_scroll_value)
        left = val
        right = left + self.window_size
        right = min(right, self.data_length - 1) if self.data_length > 0 else right
        self._set_xlim(left, right)

    # ----------------------------------------
    # Zooming (mouse wheel) centered on cursor
    # ----------------------------------------
    def _on_zoom(self, center_x, wheel_step):
        if center_x is None:
            return
        # wheel_step > 0 means scroll up (zoom in) in matplotlib default
        factor = 0.12  # zoom fraction per wheel step
        sign = -1 if wheel_step > 0 else 1  # wheel_step >0 => zoom in => reduce window
        delta = int(max(1, self.window_size * factor)) * sign
        new_size = self.window_size + delta
        new_size = max(5, min(new_size, self.data_length))
        self.window_size = new_size

        # re-center on cursor x
        left = int(center_x - new_size / 2)
        left = max(0, min(left, max(0, self.data_length - new_size)))

        self.scrollbar.setRange(0, max(0, self.data_length - new_size))
        self.scrollbar.setValue(left)
        self._set_xlim(left, left + new_size)

    # convenience
    def set_window_size(self, sz):
        self.window_size = int(sz)
        if self.data_length > 0:
            self.scrollbar.setRange(0, max(0, self.data_length - self.window_size))
            self.scrollbar.setValue(0)
            self._set_xlim(0, min(self.window_size, self.data_length - 1))




# def main():
#     app = QtWidgets.QApplication(sys.argv)
#     fn = sys.argv[1]
#     tiers = sys.argv[2]
#     print(f"{fn=}")
#     print(f"{tiers=}")
#
#     # w = MainWindow(window_size=200)
#     #
#     # # example data: sine + noise
#     # t = np.linspace(0, 10, 2000)
#     # data = np.sin(2 * np.pi * 3 * t) + 0.15 * np.random.randn(len(t))
#     #
#     # w.init_data(data, fs)
#     # w.plot_waveform(data)
#     # w.plot_sgram(data)
#     # # w.load_waveform(data)
#     # w.init_scrollbar(data)
#     # w.resize(1000, 700)
#     #
#     # w.show()
#     sys.exit(app.exec())
#
#
# if __name__ == "__main__":
#     main()

