import matplotlib as mpl
import matplotlib.collections as mcollections
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, Button
# from mpl_toolkits.axes_grid1.inset_locator import InsetPosition
import numpy as np
from collections import namedtuple


class Sound:
    def __init__(self, x, fs, tg=None):
        self.x = x
        self.fs = fs
        self.tg = tg
        self.times = np.linspace(0, x.shape[0] * fs, x.shape[0])
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

    def add_tier_line(self, ax, val):
        # ind = np.searchsorted(self.edges[ax], val)
        # self.edges[ax] = np.insert(self.edges[ax], ind, val)
        self.edges[ax] = np.append(self.edges[ax], val)
        # tier_linesy.append([0, 1])
        self.tier_marks.append(
            self.axs[ax + self.ax_off].axvline(val, 0, 1, **self.tier_kwargs)
        )

    def add_tier_text(self, ax, x, val):
        lab_starts = np.array([i.get_position()[0] for i in self.labs[ax]])
        ind = np.searchsorted(lab_starts, x)
        cur_text = mpl.text.Text(x, 0.5, text=val, zorder=1)
        self.labs[ax].insert(ind, cur_text)
        self.axs[ax + self.ax_off].add_artist(cur_text)

    def plot(self):
        """Generate Praat-style Sound and TextGrid visulization.

        Axes:
            0: TextBox input
            1: Waveform
            2: Spectrogram, etc.
            3 - 3+N: TextGrid tiers
            -1: Time Axis navigation
        """

        def get_gridspec(fig, n_tiers):
            n_plts = n_tiers + 4
            gs_top = plt.GridSpec(n_plts, 1, top=1.0, hspace=0.05, bottom=0.75)
            gs_sigs = plt.GridSpec(2, 1, top=0.9, bottom=0.4, hspace=0.05)
            gs_tiers = plt.GridSpec(n_tiers, 1, top=0.4, hspace=0, bottom=0.1)
            gs_bottom = plt.GridSpec(1, 1, top=0.1, hspace=0, bottom=0.0)

            axs = [fig.add_subplot(gs_top[0, :])]
            axs1 = [fig.add_subplot(gs_sigs[i, :]) for i in range(2)]
            axs2 = [fig.add_subplot(gs_tiers[i, :]) for i in range(n_tiers)]
            axs3 = [fig.add_subplot(gs_bottom[0, :])]
            # print(f"{len(axs)=} {len(axs1)=} {len(axs2)=} {len(axs3)=}   ")
            # PlotLayout = namedtuple("PlotLayout", ["textbox", "wavform", "analyses", "tiers", "navigation"])
            # self.layout = PlotLayout(axs[0], axs1[0], axs1[1], axs2, axs3)
            axs.extend(axs1)
            axs.extend(axs2)
            axs.extend(axs3)
            return axs

        def make_ui():

            # ip = InsetPosition(self.axs[-1], [0.4, 0.66, 0.4, 0.33])
            ip = [0.4, 0.66, 0.4, 0.33]
            # self.button1_ax = self.axs[-1]
            # self.button1_ax.set_axes_locator(ip)
            # self.button1 = Button(self.button1_ax, "button1")
            # cur_rect = mpl.patches.Rectangle(
            #     (0, 0), 1, 1, fc='y', visible=False)
            # self.selection_rect.append(cur_rect)
            # self.axs[t+self.ax_off].add_artist(cur_rect)

        def onclick(event):
            self.cur_position = event.xdata
            ax = event.inaxes

            # update selection rectangle
            if ax := event.inaxes:
                clicked_axis = self.axs[0].figure.axes.index(ax)
                # print(f"{clicked_axis=}")
            if clicked_axis in self.line_axs:
                for cur_line in self.cursor_lines:
                    cur_line.set_xdata([event.xdata])
                # print(ax._children)
                for i, a in enumerate(self.selection_rect):
                    if clicked_axis == 1:
                        continue
                    if clicked_axis in self.tier_axes:
                        self.selected_tier = self.axs[0].figure.axes.index(ax)
                    indmin = np.searchsorted(self.edges[i], event.xdata)
                    xmin, xmax = self.edges[i][indmin - 1 : indmin + 1]
                    self.selection_rect[i].set_x(xmin)
                    self.selection_rect[i].set_width(xmax - xmin)

                    self.button1_ax.figure.subplots_adjust(left=xmin, right=xmax)
                    # TODO: should get text from artist object and eliminate
                    # the class object containing labels
                    tier_n = self.selected_tier - self.ax_off
                    cur_txt = self.labs[tier_n][indmin - 2].get_text()
                    # print(cur_txt)
                    self.text_box.set_val(cur_txt)
                    a.set_visible(i + self.ax_off == self.selected_tier)

            fig.canvas.draw()
            fig.canvas.flush_events()

        # number of rows for the figure.

        # first axis number for tiers
        self.ax_off = 3
        nr = len(self.tg) + self.ax_off
        plt.ion()
        fig = plt.figure()
        self.axs = get_gridspec(fig, len(self.tg))
        # the signal axis
        # self.axs = [plt.subplot(nr-1, 1, 1)]
        self.axs[1].plot(self.times, self.x)
        self.axs[2].plot([1, 2, 1])
        for i in range(2, 2 + len(self.tg)):
            self.axs[i].sharex(self.axs[1])
        # axes for texbox input
        # self.ax_textbox = fig.add_axes([.1, .9, .3, .08])
        self.ax_textbox = self.axs[0]
        # self.axs.append(self.ax_textbox)
        self.text_box = TextBox(self.ax_textbox, "Text:")
        make_ui()

        # each axis needs a selection rectangle
        self.tier_axes = range(self.ax_off, nr)
        self.selected_tier = self.ax_off
        self.selection_rect = []
        # add tier axes
        # self.axs.extend([
        #     plt.subplot(nr, 1, i+1, sharex=self.axs[0]) for i in range(self.ax_off, nr)
        # ])
        for i in range(self.ax_off, self.ax_off + len(self.tg)):
            self.axs[i].set_ylim(0, 1)
            self.axs[i].tick_params(
                bottom=False,
                labelbottom=False,
                left=False,
                labelleft=False,
            )

        # add data to tier axes
        self.labs = [[] for i in range(nr)]
        self.tier_marks = [[] for i in range(nr)]
        self.edges = [np.array([0]) for i in range(nr)]
        for t, cur_tier in enumerate(self.tg):
            cur_rect = mpl.patches.Rectangle((0, 0), 1, 1, fc="y", visible=False)
            self.selection_rect.append(cur_rect)
            self.axs[t + self.ax_off].add_artist(cur_rect)
            self.add_tier_line(t, 0)
            for i, interval in enumerate(cur_tier):
                # self.edges[t] = np.append(self.edges[t], interval.xmax)
                self.add_tier_line(t, interval.xmax)
                self.add_tier_text(t, interval.xmin, interval.text)
                # self.labs[t].append(interval.text)

        # Last click position
        self.line_axs = list(range(len(self.axs)))
        self.line_axs.pop()
        self.line_axs.pop(0)
        self.cursor_lines = [
            a.axvline(0, 0, 1, **self.position_kwargs)
            for i, a in enumerate(self.axs)
            if i in self.line_axs
        ]
        cid = fig.canvas.mpl_connect("button_press_event", onclick)

        # for a in self.axs:
        #     print(a._children)
        plt.show(block=True)


Interval = namedtuple("Interval", ["xmin", "xmax", "text"])
tg = [
    [Interval(0, 1.1, "test11"), Interval(1.1, 2.2, "test12")],
    [Interval(0, 1.2, "test21"), Interval(1.2, 2.2, "test22")],
    [Interval(0, 1.3, "test11"), Interval(1.3, 2.2, "test12")],
    [Interval(0, 1.4, "test21"), Interval(1.4, 2.2, "test22")],
]
s = Sound(np.sin(range(220)), 0.01, tg)
s.plot()

# vim: ts=4
