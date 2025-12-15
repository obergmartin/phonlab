#!/usr/bin/env python3
"""
Waveform viewer (PySide6) with:
 - two stacked Matplotlib canvases (waveform + lines)
 - synchronized temporary cursor line following the mouse
 - click -> select time; press any key -> add boundary line
 - draggable boundary lines
 - mouse-wheel zoom centered at cursor
 - horizontal scrollbar controlling x-limits
 - throttled redraws and scroll coalescing for smooth performance (WSL2-friendly)
"""

import sys
import numpy as np
import matplotlib

# Use Agg renderer (fast raster backend); FigureCanvasQTAgg will still integrate with Qt.
matplotlib.use("Agg")

from PySide6 import QtCore, QtWidgets
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


# -------------------------------------------------------------
# Utility: Shared temp-line synchronizer
# -------------------------------------------------------------
class TempLineSync:
    """Register canvases that should share a single temporary cursor x-position."""

    def __init__(self):
        self._canvases = []

    def register(self, canvas):
        self._canvases.append(canvas)

    def update_all(self, x, sender=None):
        for c in self._canvases:
            if c is not sender:
                try:
                    c.set_temp_line_blit(x)
                except Exception:
                    # fallback
                    c.set_temp_line(x)

    def clear_all(self):
        for c in self._canvases:
            try:
                c.clear_temp_line_blit()
            except Exception:
                c.clear_temp_line()


# -------------------------------------------------------------
# Base Matplotlib canvas with throttled drawing + blit helpers
# -------------------------------------------------------------
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


# -------------------------------------------------------------
# Waveform canvas (top)
# -------------------------------------------------------------
class WaveformCanvas(ThrottledCanvas):
    def __init__(self, parent=None, sync: TempLineSync = None):
        fig = Figure()
        super().__init__(fig, parent=parent)
        self.axes = fig.add_subplot(211)
        self.axes.set_autoscale_on(False)
        self.x_data = None
        self.y_data = None

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
        data = np.asarray(data)
        self.x_data = np.arange(data.shape[0])
        self.y_data = data
        self.axes.clear()
        self.axes.plot(self.x_data, self.y_data, linewidth=0.8, antialiased=False)
        # fix y-limits to avoid autoscale cost
        self.axes.set_ylim(np.min(self.y_data), np.max(self.y_data))
        # full redraw now
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
        # if dragging line
        # if dragging span
        # else moving templine
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


# -------------------------------------------------------------
# Lines canvas (bottom) - contains draggable red lines
# -------------------------------------------------------------
class LinesCanvas(ThrottledCanvas):
    def __init__(self, parent=None, sync: TempLineSync = None):
        fig = Figure()
        super().__init__(fig, parent=parent)
        self.axes = fig.add_subplot(111)
        self.axes.set_autoscale_on(False)

        self.lines = []  # list of Line2D objects (vertical red lines)
        self._temp_line = None
        self._selected_line = None

        self.zoom_callback = None
        self.sync = sync

        # connect events
        self.mpl_connect("motion_notify_event", self._on_motion)
        self.mpl_connect("button_press_event", self._on_click)
        self.mpl_connect("motion_notify_event", self._on_drag)
        self.mpl_connect("button_release_event", self._on_release)
        self.mpl_connect("scroll_event", self._on_scroll)

    def add_line(self, x):
        ln = self.axes.axvline(x, color="r", linewidth=1, antialiased=False)
        self.lines.append(ln)
        # When adding a permanent line, clear any temp lines
        if self.sync:
            self.sync.clear_all()
        self.request_draw()

    def set_temp_line(self, x):
        if self._temp_line is None:
            self._temp_line = self.axes.axvline(
                x, color="g", linestyle="--", linewidth=1, antialiased=False
            )
        else:
            self._temp_line.set_visible(True)
            self._temp_line.set_xdata([x, x])
        self.request_draw()

    def clear_temp_line(self):
        if self._temp_line:
            self._temp_line.set_visible(False)
            self.request_draw()

    def set_temp_line_blit(self, x):
        if self._temp_line is None:
            self._temp_line = self.axes.axvline(
                x, color="g", linestyle="--", linewidth=1, antialiased=False
            )
            self.draw()
            # try caching for blit if possible
            try:
                self.enable_blit_for_artist(self._temp_line)
            except Exception:
                self.clear_blit_cache()
                self.set_temp_line(x)
            return

        if not self._blit_available or self._background is None:
            self.set_temp_line(x)
            return

        self._temp_line.set_visible(True)
        self._temp_line.set_xdata([x, x])
        try:
            self.blit_draw_artist(self._temp_line)
        except Exception:
            self.clear_blit_cache()
            self.set_temp_line(x)

    # Event handlers
    def _on_motion(self, event):
        # show temporary line only when not dragging a real line
        if (
            self._selected_line is None
            and event.inaxes == self.axes
            and event.xdata is not None
        ):
            x = event.xdata
            try:
                self.set_temp_line_blit(x)
            except Exception:
                self.set_temp_line(x)
            if self.sync:
                self.sync.update_all(x, sender=self)

    def _on_click(self, event):
        if event.inaxes != self.axes or event.xdata is None:
            return
        # if in plot axes, start span
        # detect if clicking near an existing line to start drag
        for ln in self.lines:
            if abs(event.xdata - ln.get_xdata()[0]) < 0.6:
                self._selected_line = ln
                return
        # else, nothing special (new lines are added via main window keypress)

    def _on_drag(self, event):
        if (
            self._selected_line
            and event.inaxes == self.axes
            and event.xdata is not None
        ):
            x = event.xdata
            self._selected_line.set_xdata([x, x])
            # fast update via blit if possible
            try:
                # We can try to blit the selected red line, but it's not set up as animated.
                # Simpler: update temp-line and blit that, and let full redraw happen at throttled rate.
                self.request_draw()
            except Exception:
                self.request_draw()
            # also sync the temp-line position
            if self.sync:
                self.sync.update_all(x, sender=self)

    def _on_release(self, event):
        self._selected_line = None

    def _on_scroll(self, event):
        if self.zoom_callback:
            self.zoom_callback(event.xdata, event.step)

    def set_xlim(self, xmin, xmax):
        self.axes.set_xlim(xmin, xmax)
        self.request_draw()


# -------------------------------------------------------------
# Main window: orchestrates both canvases + scrollbar + keyboard
# -------------------------------------------------------------
class MainWindow(QtWidgets.QWidget):
    def __init__(self, window_size=200):
        super().__init__()
        self.setWindowTitle("Waveform Viewer (PySide6)")
        self.window_size = window_size
        self.data_length = 0
        self._last_clicked_time = None

        # temp-line synchronizer
        self.sync = TempLineSync()

        # layout
        v = QtWidgets.QVBoxLayout(self)

        # canvases
        self.wave_canvas = WaveformCanvas(parent=self, sync=self.sync)
        self.line_canvas = LinesCanvas(parent=self, sync=self.sync)

        # allow canvases to be registered
        self.sync.register(self.wave_canvas)
        self.sync.register(self.line_canvas)

        v.addWidget(self.wave_canvas)
        v.addWidget(self.line_canvas)

        # scrollbar
        self.scrollbar = QtWidgets.QScrollBar(QtCore.Qt.Horizontal)
        v.addWidget(self.scrollbar)

        # connect callbacks
        self.wave_canvas.time_selected_callback = self._on_time_selected
        self.wave_canvas.zoom_callback = self._on_zoom
        self.line_canvas.zoom_callback = self._on_zoom

        # --- Scroll throttling (coalesce rapid events) ---
        self.pending_scroll = False
        self.last_scroll_value = None
        self._scroll_timer = QtCore.QTimer()
        self._scroll_timer.setInterval(16)  # ~60 FPS
        self._scroll_timer.timeout.connect(self._apply_scroll_update)

        # reduce spam: disable native tracking (some platforms emit many events while dragging)
        self.scrollbar.setTracking(False)

        # connect both sliderMoved (user drag) and valueChanged (keyboard/arrow/click)
        self.scrollbar.sliderMoved.connect(self._queue_scroll_update)
        self.scrollbar.valueChanged.connect(self._queue_scroll_update)

        # key focus
        self.setFocusPolicy(QtCore.Qt.StrongFocus)

    def load_waveform(self, data):
        """Load waveform (1D array)."""
        self.wave_canvas.plot_waveform(data)
        self.data_length = len(data)
        if self.data_length <= 0:
            self.scrollbar.setEnabled(False)
            return

        max_pos = max(0, self.data_length - self.window_size)
        self.scrollbar.setRange(0, max_pos)
        self.scrollbar.setPageStep(self.window_size)
        self.scrollbar.setSingleStep(1)
        self.scrollbar.setEnabled(True)
        # set initial x-limits
        self._set_xlim(0, min(self.window_size, self.data_length - 1))

        # Once plotted, give canvases a chance to setup blit caches
        self.wave_canvas.request_blit_setup()
        # also setup for lines canvas (it will create a temp line)
        self.line_canvas.set_temp_line(0)
        self.line_canvas.request_draw()

    def _set_xlim(self, xmin, xmax):
        """Apply same x-limits to both canvases (integers expected)."""
        xmin = max(0, int(xmin))
        xmax = max(xmin + 1, int(xmax))
        self.wave_canvas.axes.set_xlim(xmin, xmax)
        self.wave_canvas.request_draw()
        self.line_canvas.set_xlim(xmin, xmax)

        # update blit background caches if using blit
        try:
            self.wave_canvas.request_blit_setup()
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


# -------------------------------------------------------------
# Run as script (example)
# -------------------------------------------------------------
def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow(window_size=200)

    # example data: sine + noise
    t = np.linspace(0, 10, 2000)
    data = np.sin(2 * np.pi * 3 * t) + 0.15 * np.random.randn(len(t))

    w.load_waveform(data)
    w.resize(1000, 700)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
