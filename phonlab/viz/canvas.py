import numpy as np
import pandas as pd
from PySide6 import QtCore, QtWidgets
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.cm import get_cmap

from phonlab import compute_sgram

def df2lines(df: pd.DataFrame):
    if "t2" in df.columns:
        lines = np.concat([df.t1.to_numpy(), [df.t2.iloc[-1]]])
    elif "t1" in df.columns:
        lines = df.t1.to_numpy()
    else:
        print("not a valid df")
        return
    return lines

def df2texobj(df):
    if "t2" in df.columns:
        return list(zip(
            df[['t1','t2']].mean(axis=1),
            df.iloc[:, 2]
        ))
    elif "t1" in df.columns:
        return list(zip(
            df['t1'],
            df.iloc[:, 1]
        ))

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

        gs = GridSpec(n_rows, 1, figure=self.fig, height_ratios=height_ratios, hspace=0,
                      left=0.1, right=0.9, bottom=0.01, top=0.95)
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
        self.tier_lines = []
        self.ax_sgram.set_ylim(0,8000)

        self.draw_idle()

    def plot_sgram(self, data, fs, wb, min_prop=0.2):
        f, ts, Sxx = compute_sgram(data, fs, wb)
        vmin = np.min(Sxx) + (np.max(Sxx)-np.min(Sxx))*min_prop
        extent = (min(ts), max(ts), min(f), max(f))  # get the time and frequency values for indices.
        cmap = get_cmap("Grays")
        im = self.ax_sgram.imshow(
            Sxx, aspect='auto', interpolation='nearest', cmap=cmap, vmin=vmin,
            extent=extent, origin='lower'
        )
        self.ax_sgram.grid(which='major', axis='y', linestyle=':')




    def plot_tier(self, ax_ind, tier):
        # put this in Tier class??
        if tier.tier_type == "segment":
            ls = "-"
        elif tier.tier_type == "point":
            # ls = "--"
            # TODO: these numbers should depend on figsize??
            ls = (0, (5, 15, 5, 0))
        # tier lines
        tier_lines = df2lines(tier.df)
        cur_ax = self.axs_tiers[ax_ind]
        cur_tier_lines = []
        for data in tier_lines:
            h = cur_ax.axvline(x=data, ls=ls)
            cur_tier_lines.append(h)

        # add segment labels
        tier_labels = df2texobj(tier.df)
        for t, txt in tier_labels:
            cur_ax.text(t, 0.5, txt, ha="center")

    #
    #

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

        if self.temp_lines is None:
            self.temp_lines = self.axs_tiers[0].axvline(
                x, color="g", linestyle="--", linewidth=1, antialiased=False
            )
        else:
            self.temp_lines.set_xdata([x, x])

        if self.temp_wave is None:
            self.temp_wave = self.ax_wave.axvline(
                x, color="g", linestyle="--", linewidth=1, antialiased=False
            )
        else:
            self.temp_wave.set_xdata([x, x])

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

