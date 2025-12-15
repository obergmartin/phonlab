#!/usr/bin/env python3

import sys
import numpy as np
import matplotlib

# Fast raster backend (important for WSL2)
matplotlib.use("Agg")

from PySide6 import QtCore, QtWidgets
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


# ------------------------------------------------------------
# Single canvas with two shared-x axes
# ------------------------------------------------------------
class WaveformCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure()
        super().__init__(self.fig)
        self.setParent(parent)

        self.ax_wave = self.fig.add_subplot(211)
        self.ax_lines = self.fig.add_subplot(212, sharex=self.ax_wave)

        self.ax_wave.set_autoscale_on(False)
        self.ax_lines.set_autoscale_on(False)
        self.ax_lines.set_yticks([])

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

    # --------------------------------------------------------
    # Plotting
    # --------------------------------------------------------
    def plot_waveform(self, data):
        self.wave_data = data
        x = np.arange(len(data))

        self.ax_wave.clear()
        self.ax_lines.clear()
        self.ax_lines.set_yticks([])

        self.ax_wave.plot(x, data, linewidth=0.8, antialiased=False)
        self.ax_wave.set_ylim(np.min(data), np.max(data))
        self.ax_lines.set_ylim(0, 1)

        self.boundary_lines.clear()
        self.temp_wave = None
        self.temp_lines = None

        self.draw_idle()

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
            self.temp_lines = self.ax_lines.axvline(
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
        ln = self.ax_lines.axvline(x, color="r", linewidth=1, antialiased=False)
        self.boundary_lines.append(ln)
        self.clear_temp_line()
        self.draw_idle()

    # --------------------------------------------------------
    # Event handlers
    # --------------------------------------------------------
    def on_motion(self, event):
        if event.inaxes not in (self.ax_wave, self.ax_lines):
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
        if self.selected_line and event.inaxes == self.ax_lines:
            self.selected_line.set_xdata([x, x])
            self.draw_idle()

    def on_click(self, event):
        if event.inaxes == self.ax_wave and event.xdata is not None:
            self.span_start = event.xdata
            self.span_end = event.xdata
            self.update_span(self.span_start, self.span_end)

        elif event.inaxes == self.ax_lines and event.xdata is not None:
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
    def __init__(self, window_size=200):
        super().__init__()
        self.setWindowTitle("Waveform Viewer (Shared X)")
        self.window_size = window_size
        self.data_length = 0

        layout = QtWidgets.QVBoxLayout(self)

        self.canvas = WaveformCanvas(self)
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

        self.setFocusPolicy(QtCore.Qt.StrongFocus)

    # --------------------------------------------------------
    # Data loading
    # --------------------------------------------------------
    def load_waveform(self, data):
        self.canvas.plot_waveform(data)
        self.data_length = len(data)

        max_pos = max(0, self.data_length - self.window_size)
        self.scrollbar.setRange(0, max_pos)
        self.scrollbar.setPageStep(self.window_size)
        self.scrollbar.setValue(0)

        self.set_xlim(0)

    # --------------------------------------------------------
    # X-range handling (single shared-x call)
    # --------------------------------------------------------
    def set_xlim(self, left):
        left = max(0, min(left, self.data_length - self.window_size))
        right = left + self.window_size
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
        factor = 0.85 if step > 0 else 1.2
        new_size = int(self.window_size * factor)
        new_size = max(10, min(new_size, self.data_length))
        self.window_size = new_size

        left = int(center_x - new_size / 2)
        left = max(0, min(left, self.data_length - new_size))

        self.scrollbar.setRange(0, max(0, self.data_length - new_size))
        self.scrollbar.setValue(left)
        self.set_xlim(left)

    # --------------------------------------------------------
    # Boundary creation
    # --------------------------------------------------------
    def on_time_selected(self, x):
        self.canvas.last_clicked_x = x

    def keyPressEvent(self, event):
        canvas = self.canvas

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
def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow(window_size=200)

    t = np.linspace(0, 10, 2000)
    data = np.sin(2 * np.pi * 3 * t) + 0.15 * np.random.randn(len(t))

    w.load_waveform(data)
    w.resize(1000, 700)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
