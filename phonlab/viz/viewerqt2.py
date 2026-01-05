#!/usr/bin/env python3

import sys
import numpy as np
import matplotlib
import phonlab as phon
from .canvas import WaveformCanvas

# Fast raster backend (important for WSL2)
matplotlib.use("Agg")

from PySide6 import QtCore, QtWidgets
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec


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

        self.textbox = QtWidgets.QLineEdit()
        layout.addWidget(self.textbox)

        self.canvas = WaveformCanvas(n_tiers=n_tiers)
        layout.addWidget(self.canvas)

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
        bin_width = 0.04
        bin_width = 0.008
        self.canvas.plot_sgram(data, fs, bin_width)
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
        if "t2" in df.columns:
            self.tier_type = "segment"
        else:
            self.tier_type = "point"

    def __repr__(self):
        return f"Tier: {self.df.columns[-1]} with {self.df.shape[0]} intervals"


class Sound:
    def __init__(self, fn, tg=True):
        tfn = fn.replace(".wav", ".TextGrid")
        wav, fs = phon.loadsig(fn)
        dfs = phon.tg_to_df(tfn)  # [:2]
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
        w = MainWindow(window_size=ws, n_tiers=len(self.tiers))

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
