#!/usr/bin/env python3

import sys
import numpy as np
import matplotlib
import phonlab as phon

# Fast raster backend (important for WSL2)
matplotlib.use("Agg")

from PySide6 import QtCore, QtWidgets
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

# ------------------------------------------------------------
# Single canvas with two shared-x axes
# ------------------------------------------------------------
class WaveformCanvas(FigureCanvas):
    def __init__(self, n_tiers=1, parent=None):
        self.fig = Figure()
        super().__init__(self.fig)
        self.setParent(parent)

        self.axs = self.init_axes(n_tiers)
        # self.ax_wave = self.fig.add_subplot(211)
        # self.ax_lines = self.fig.add_subplot(212, sharex=self.ax_wave)
        self.ax_wave = self.axs[0]
        self.ax_sgram = self.axs[1]
        self.axs_tiers = self.axs[2:-1]
        self.ax_button = self.axs[-1]

        for ax in self.axs:
            ax.set_autoscale_on(False)
        # TODO: should work here, but needed in plot_waveform?
        for ax in self.axs_tiers:
            ax.set_yticks([])
            ax.set_xticks([])
        self.ax_button.set_yticks([])

        self.wave_data = None
        self.boundary_lines = []

        self.temp_wave = None
        self.temp_lines = None
        self.selected_line = None
        self.last_clicked_x = None

        self.span_start = None
        self.span_end = None
        self.span_patch = None

        self.zoom_callback = None
        self.time_selected_callback = None

        self.mpl_connect("motion_notify_event", self.on_motion)
        self.mpl_connect("button_press_event", self.on_click)
        self.mpl_connect("button_release_event", self.on_release)
        self.mpl_connect("scroll_event", self.on_scroll)

    def init_axes(self, n_tiers):
        waveform_h, sgram_h, tier_h, button_h = 1, 1, 0.3, 0.1
        n_rows = 3 + n_tiers
        height_ratios = [waveform_h, sgram_h] + [tier_h]*n_tiers + [button_h]

        # TODO: remove hspace
        gs = GridSpec(n_rows, 1, figure=self.fig, height_ratios=height_ratios)
        axs = [self.fig.add_subplot(gs[x]) for x in range(n_rows)]
        for i in range(1, n_rows):
            axs[i].sharex(axs[0])
        return axs


    # --------------------------------------------------------
    # Plotting
    # --------------------------------------------------------
    def plot_waveform(self, data, fs):
        self.wave_data = data
        x = np.arange(len(data)) / fs

        self.ax_wave.clear()
        for ax in self.axs_tiers:
            ax.clear()
            ax.set_yticks([])
            ax.set_xticks([])
            ax.set_ylim(0, 1)

        self.ax_wave.plot(x, data, linewidth=0.8, antialiased=False)
        self.ax_wave.set_ylim(np.min(data), np.max(data))

        self.boundary_lines.clear()
        self.temp_wave = None
        self.temp_lines = None

        self.draw_idle()

    def plot_tier(self, ax, df):
        for data in self.

    def commit_span_as_boundaries(self):
        if self.span_start is None or self.span_end is None:
            return

        left, right = sorted((self.span_start, self.span_end))
        self.add_boundary(left)
        self.add_boundary(right)
        self.clear_span()

    # --------------------------------------------------------
    # Temporary cursor line
    # --------------------------------------------------------
    def set_temp_line(self, x):
        if self.temp_wave is None:
            self.temp_wave = self.ax_wave.axvline(
                x, color="g", linestyle="--", linewidth=1, antialiased=False
            )
            self.temp_lines = self.axs_tiers.axvline(
                x, color="g", linestyle="--", linewidth=1, antialiased=False
            )
        else:
            self.temp_wave.set_xdata([x, x])
            self.temp_lines.set_xdata([x, x])

        self.temp_wave.set_visible(True)
        self.temp_lines.set_visible(True)
        self.draw_idle()

    def clear_temp_line(self):
        if self.temp_wave:
            self.temp_wave.set_visible(False)
            self.temp_lines.set_visible(False)
            self.draw_idle()

    #
    #
    #

    def update_span(self, x0, x1):
        left, right = sorted((x0, x1))

        if self.span_patch is None:
            self.span_patch = self.ax_wave.axvspan(
                left, right, color="tab:blue", alpha=0.25
            )
        else:
            self.span_patch.set_xy([left, 0])
            self.span_patch.set_width(right - left)

        self.draw_idle()

    def clear_span(self):
        if self.span_patch is not None:
            self.span_patch.remove()
            self.span_patch = None
            self.span_start = None
            self.span_end = None
            self.draw_idle()

    # --------------------------------------------------------
    # Boundary lines
    # --------------------------------------------------------
    def add_boundary(self, x):
        ln = self.axs_tiers.axvline(x, color="r", linewidth=1, antialiased=False)
        self.boundary_lines.append(ln)
        self.clear_temp_line()
        self.draw_idle()

    # --------------------------------------------------------
    # Event handlers
    # --------------------------------------------------------
    def on_motion(self, event):
        if event.inaxes not in (self.ax_wave, self.axs_tiers):
            return
        if event.xdata is None:
            return

        x = event.xdata

        # Update temporary cursor
        self.set_temp_line(x)

        # Update span if dragging on waveform
        if self.span_start is not None and event.inaxes == self.ax_wave:
            self.span_end = x
            self.update_span(self.span_start, self.span_end)

        # Drag existing boundary line
        if self.selected_line and event.inaxes == self.axs_tiers:
            self.selected_line.set_xdata([x, x])
            self.draw_idle()

    def on_click(self, event):
        if event.inaxes == self.ax_wave and event.xdata is not None:
            self.span_start = event.xdata
            self.span_end = event.xdata
            self.update_span(self.span_start, self.span_end)

        elif event.inaxes == self.axs_tiers and event.xdata is not None:
            for ln in self.boundary_lines:
                if abs(event.xdata - ln.get_xdata()[0]) < 0.5:
                    self.selected_line = ln
                    return

        self.last_clicked_x = (
            int(round(event.xdata)) if event.xdata is not None else None
        )

    def on_release(self, event):
        self.selected_line = None

        if self.span_start is not None and event.inaxes == self.ax_wave:
            self.span_end = event.xdata
            self.span_start = None

    def on_scroll(self, event):
        if self.zoom_callback and event.xdata is not None:
            self.zoom_callback(event.xdata, event.step)


# ------------------------------------------------------------
# Main window
# ------------------------------------------------------------
class MainWindow(QtWidgets.QWidget):
    def __init__(self, window_size=200, n_tiers=1):
        super().__init__()
        self.setWindowTitle("Waveform Viewer (Shared X)")
        self.window_size = window_size
        self.data_length = 0

        layout = QtWidgets.QVBoxLayout(self)

        self.canvas = WaveformCanvas(n_tiers=n_tiers)
        layout.addWidget(self.canvas)

        self.textbox = QtWidgets.QLineEdit()
        layout.addWidget(self.textbox)

        self.scrollbar = QtWidgets.QScrollBar(QtCore.Qt.Horizontal)
        layout.addWidget(self.scrollbar)

        self.canvas.time_selected_callback = self.on_time_selected
        self.canvas.zoom_callback = self.on_zoom

        self.pending_scroll = False
        self.last_scroll_value = None
        self.scroll_timer = QtCore.QTimer()
        self.scroll_timer.setInterval(16)
        self.scroll_timer.timeout.connect(self.apply_scroll)

        self.scrollbar.setTracking(False)
        self.scrollbar.sliderMoved.connect(self.queue_scroll)
        self.scrollbar.valueChanged.connect(self.queue_scroll)
        self.canvas.setFocus()

        # self.setFocusPolicy(QtCore.Qt.StrongFocus)

    def plt(self):
        self.show()
        return "foo"

    # --------------------------------------------------------
    # Data loading
    # --------------------------------------------------------
    def init_app(self, data, fs, dfs):
        self.fs = fs
        self.signal_length = data.size/fs
        # print(f"{self.signal_length=}")
        # self.canvas.plot_waveform(data, self.fs, dfs)
        self.canvas.plot_waveform(data, fs)
        for i, df in enumerate(dfs):
            self.canvas.plot_tier(i, df)
        self.data_length = len(data)

        max_pos = max(0, self.data_length - self.window_size)
        page_step = 500
        max_pos = 2000 - page_step
        # self.scrollbar.setRange(0, max_pos)
        self.scrollbar.setMinimum(0)
        self.scrollbar.setMaximum(max_pos)
        self.scrollbar.setPageStep(page_step)
        self.scrollbar.setValue(0)

        self.set_xlim(0)

    # --------------------------------------------------------
    # X-range handling (single shared-x call)
    # --------------------------------------------------------
    def set_xlim(self, left):
        left = left / 1000
        left = max(0, min(left, self.data_length - self.window_size))
        right = left + self.window_size
        right = right/1000
        print(left, right)
        self.canvas.ax_wave.set_xlim(left, right)
        self.canvas.draw_idle()

    # --------------------------------------------------------
    # Scroll throttling
    # --------------------------------------------------------
    @QtCore.Slot(int)
    def queue_scroll(self, value):
        self.last_scroll_value = value
        if not self.pending_scroll:
            self.pending_scroll = True
            self.scroll_timer.start()

    @QtCore.Slot()
    def apply_scroll(self):
        self.scroll_timer.stop()
        self.pending_scroll = False
        if self.last_scroll_value is not None:
            self.set_xlim(self.last_scroll_value)

    # --------------------------------------------------------
    # Zoom
    # --------------------------------------------------------
    def on_zoom(self, center_x, step):
        factor = 0.9 if step > 0 else 1.1
        print(f"{self.window_size=}")
        new_size = int(self.window_size * factor)
        # new_size = max(self.window_size, min(new_size, self.data_length))
        if new_size > self.signal_length:
            new_size = int(self.signal_length*1000)
        if new_size < 0.001:
            new_size = 0.001
        print(f"{new_size=}")
        self.window_size = new_size

        left = int(center_x - new_size / 2)
        # left = max(0.01, min(left, self.data_length - new_size))

        self.scrollbar.setRange(0, max(0, new_size))
        self.scrollbar.setValue(left)
        self.set_xlim(left)
        # breakpoint()

    # --------------------------------------------------------
    # Boundary creation
    # --------------------------------------------------------
    def on_time_selected(self, x):
        self.canvas.last_clicked_x = x

    def keyPressEvent(self, event):
        canvas = self.canvas
        key = event.key()

        # print(f"{event=}")
        if key == QtCore.Qt.Key_Minus:
            center_x = np.mean(self.canvas.ax_wave.get_xlim())
            self.on_zoom(center_x, step=-1)
            return
        elif key == QtCore.Qt.Key_Equal:
            center_x = np.mean(self.canvas.ax_wave.get_xlim())
            self.on_zoom(center_x, step=1)
            return
        # Priority: commit span if it exists
        if canvas.span_start is not None and canvas.span_end is not None:
            canvas.commit_span_as_boundaries()
            return

        # Fallback: single-point boundary
        if canvas.last_clicked_x is not None:
            canvas.add_boundary(canvas.last_clicked_x)


# ------------------------------------------------------------
# Run example
# ------------------------------------------------------------
def view():
    if not QtWidgets.QApplication.instance():
        app = QtWidgets.QApplication(sys.argv)
    else:
        app = QtWidgets.QApplication.instance()
    w = MainWindow(window_size=200)

    t = np.linspace(0, 10, 2000)
    data = np.sin(2 * np.pi * 3 * t) + 0.15 * np.random.randn(len(t))

    w.load_waveform(data, 10)
    w.resize(1000, 700)
    w.show()
    sys.exit(app.exec())
    app.exec()
    return "foo"


class Tier:
    def __init__(self, df):
        self.name = "MyName"
        self.df = df

    def __repr__(self):
        return f"Tier: {self.df.columns[-1]} with {self.df.shape[0]} intervals"


class Sound:
    def __init__(self, fn, tg=True):
        tfn = fn.replace(".wav", ".TextGrid")
        wav, fs = phon.loadsig(fn)
        dfs = phon.tg_to_df(tfn)[:2]
        self.wav = wav
        self.fs = fs
        self.tiers = [Tier(df) for df in dfs]
        print(self.tiers)

    def __repr__(self):
        return f"Sound ({self.wav.size/self.fs}s) with {len(self.tiers)} tiers"

    def plot(self):
        if not QtWidgets.QApplication.instance():
            app = QtWidgets.QApplication(sys.argv)
        else:
            app = QtWidgets.QApplication.instance()
        # w = MainWindow(window_size=self.wav.size / self.fs, n_tiers=2)
        ws = self.wav.size/self.fs*1000
        w = MainWindow(window_size=ws, n_tiers=2)

        # t = np.linspace(0, 10, 2000)
        # data = np.sin(2 * np.pi * 3 * t) + 0.15 * np.random.randn(len(t))
        self.tiers_ = ["foo"]

        w.init_app(self.wav, self.fs, self.tiers)
        # w.resize(1000, 700)
        w.show()
        sys.exit(app.exec())
        app.close()

fn = "phonlab/data/example_audio/im_twelve.wav"
# s = Sound(fn, tg=True)
# s.plot()
# if __name__ == "__main__":
#     main()
