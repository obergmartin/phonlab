import sys
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
from matplotlib.widgets import Button
from matplotlib.patches import Rectangle
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec
import subprocess
from .signal import loadsig
from phonlab import prep_audio
from phonlab import sgram
from time import time
import sounddevice as sd
import matplotlib.style as mplstyle
mplstyle.use('fast')

viz_settings = dict(
    tier_kwargs = dict(
        alpha=1,
        c="b",
        ls="-",
    ),
    position_kwargs = dict(
        alpha=0.7,
        c="r",
        ls="--",
    ),
    line_kwargs = dict(
        color="b",
        ls="-",
        lw=4,
    ),
    span_kwargs = dict(
        alpha=0.3,
        ls="-",
    ),
    seg_span_kwargs = dict(
        alpha=0.3,
        fill='g',
        ls="-",
    ),
    seg_kwargs = dict(
        fill=False,
        edgecolor="b",
        lw=2,
        ls="-",
    ),
)


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
    line_collection =  LineCollection(lpts)
    line_collection.set_picker(True)
    return line_collection


class Viewer:
    def __init__(self, fn, df):
        self.start_x = None
        assert isinstance(df, list), "df must be list(dfs)"
        self.n_rows = 3 + len(df)
        print(f"{self.n_rows=}")
        self.lines = [df2lines(x) for x in df]

        # self.fig, self.axs = plt.subplots(nrows=self.n_rows, ncols=1)

        self.fig = plt.figure()
        height_ratios = [1, 1] + [.3]*len(df) + [0.1]
        gs = GridSpec(self.n_rows, 1, figure=self.fig, height_ratios=height_ratios)
        self.axs = [self.fig.add_subplot(gs[x]) for x in range(self.n_rows)]
        for i in range(1, self.n_rows):
            self.axs[i].sharex(self.axs[0])

        # tier_axs
        self.tier_axs = self.axs[2:2+len(df)]
        self.button_axs = self.axs[-1]

        # load data
        self.fn = fn
        self.wavdata, self.fs = loadsig(fn, chansel=[0])  # taking just the left channel, with 'chansel'
        decim = 10
        x = sp.signal.decimate(self.wavdata, decim)
        self.axs[0].plot(np.arange(0, x.size)/(self.fs/decim), x, c='k')
        self.axs[0].set_xlim([0, self.wavdata.size/self.fs])
        sgram(self.wavdata, self.fs, tf=8000, ax=self.axs[1])
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0, hspace=0)

        self.fig.canvas.mpl_connect('key_press_event', self.on_keypress)
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('pick_event', self.on_pick)

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

        self.viz_settings = viz_settings

        # set up indicator line across axes
        self.line_axs = list(range(len(self.axs)))
        self.cursor_lines = [
            a.axvline(1, 0, 1, **self.viz_settings["position_kwargs"])
            for i, a in enumerate(self.axs)
            if i in self.line_axs
        ]
        self.set_active_line(0, visible=False)
        # collection of handles for selection span on axes
        self.current_span = [
            a.axvspan(1, 2, **self.viz_settings["span_kwargs"])
            for i, a in enumerate(self.axs)
        ]

        # selected tier segment
        #
        self.seg_span = self.axs[2].axvspan(-2,-1, **self.viz_settings["seg_span_kwargs"])
        self.set_active_span(0,1,visible=False)
        for i in range(len(df)):
            self.set_active_tier_segment(i, -1)
        # format lims for tier axes
        self.axs[2].get_yaxis().set_visible(False)
        self.axs[2].get_yaxis().set_ticks([])
        # 
        # for t1 in self.lines:
        #     # self.axs[2].axvspan(t1, t2, -0.1, 1.02, **self.seg_kwargs)
        #     self.axs[2].axvline(t1, -0.1, 1.02, **self.line_kwargs)

        # tier lines
        self.active_tier_boundary = None
        self.tier_lines = [make_linecollection(x) for x in self.lines]
        self.tier_labels = [df2texobj(x) for x in df]
        for i, (cur_lines, cur_labels) in enumerate(zip(self.tier_lines, self.tier_labels)):
            self.axs[i+2].add_collection(cur_lines)
            for t, txt in cur_labels:
                self.axs[i+2].text(t, 0.5, txt, ha="center")

        plt.show()

    ## methods for updating visuals when user interacts with figure

    def play_segment(self, event):
        xlims = self.axs[0].get_xlim()
        s1 = self.current_span[0].get_x()
        s2 = s1+self.current_span[0].get_width()
        if event.inaxes == self.axprev:
            start_time = xlims[0]
            end_time = s1
        elif event.inaxes == self.axcur:
            start_time = s1
            end_time = s2
        elif event.inaxes == self.axnext:
            start_time = s2
            end_time = xlims[1]
        else:
            return
        # tier lines
        # TODO: pad signal to ensure everything gets played?
        # TODO: is it worthwhile trying to cache previous play buffer?
        t1, t2 = int(start_time*self.fs), int(end_time*self.fs)
        myarray = self.wavdata[t1:t2]
        sd.play(myarray, self.fs)

    def set_active_tier_boundary(self, tier_ind, ind):
        """Click on a tier boundary to move it.
        """
        # set all segments to width=1
        lw = np.ones(len(self.lines[tier_ind]), dtype=int) * 1
        # set selected segment style
        if ind >= 0:
            lw[ind] = 4
            self.active_tier_boundary = self.tier_lines[tier_ind]
            self.active_tier_boundary.set(linewidths=lw)
        else:
            # pass ind == -1 to clear active segment
            self.active_tier_boundary = None

    def set_active_tier_segment(self, tier_ind, ind):
        """Click in a tier segment to higlight it.
        """
        # clear selection line
        # show tier selection
        if ind >= 0:
            self.seg_span.set_visible(True)
            self.seg_span.set_x(self.lines[tier_ind][ind])
            self.seg_span.set_width(self.lines[tier_ind][ind+1] - self.lines[tier_ind][ind])
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

    def get_ind_under_point(self, event, tier_ind):
        """
        Return the index of the point closest to the event position or *None*
        if no point is within ``self.epsilon`` to the event position.
        """
        self.epsilon = 5  # in n_pixels, make dependant on zoom level?

        eventxt = self.tier_lines[tier_ind].get_transform().inverted().transform((event.x,event.y))
        xy = [(x,0) for x in self.lines[tier_ind]]
        xyt = self.tier_lines[tier_ind].get_transform().transform(xy)  # to display coords
        d = np.array([x[0] for x in xyt]) - event.x
        ind = int(abs(d).argmin())
        if abs(d[ind]) < self.epsilon:
            # print(f"found segment", ind)
            return ind
        else:
            ind =  np.searchsorted([x[0] for x in xyt], event.x)
            # print(f"found span", (ind-1, ind))
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
            pos = np.searchsorted(self.lines[0], self.start_x)
            self.lines[0] = np.insert(self.lines[0], pos, self.start_x)
            self.tier_lines[0].set_segments(self.lines[0])
            self.set_tier_labels(labels)

    def on_pick(self, event):
        for i, cur_tier in enumerate(self.tier_axs):
            if event.inaxes == cur_tier:
                tier_ind = i
                self.cur_tier = i
        print(f"{event.artist=}")
        print(f"{event.ind=}")
        print(f"{self.tier_lines[self.cur_tier][event.ind]=}")

    def on_press(self, event):
        """Record the starting x-coordinate on button press."""
        # is click on existing:
        # tier axes
        for i, cur_tier in enumerate(self.tier_axs):
            if event.inaxes == cur_tier:
                tier_ind = i
                self.cur_tier = i
        if event.inaxes in self.tier_axs:
            ind = self.get_ind_under_point(event, tier_ind)
            # span boundary
            if isinstance(ind, int):
                self.dragging = "segment"
                self.active_tier_boundary = self.tier_lines[tier_ind]
                p1 = self.lines[tier_ind][ind]
                self.set_active_tier_boundary(tier_ind, ind)
                self.set_active_line(event.xdata)
                self.set_active_tier_segment(tier_ind, -1)
                self.set_active_span(0, 0, False)
                self.resize_play_buttons(p1, p1)
            # tier span
            else:
                self.dragging = None
                p1, p2 = self.lines[tier_ind][ind[0]], self.lines[tier_ind][ind[1]]
                # setspanhere
                self.set_active_tier_boundary(tier_ind, -1)
                self.set_active_tier_segment(tier_ind, ind[0])
                self.set_active_span(p1, p2-p1)
                self.resize_play_buttons(p1, p2)
        # sgram or signal
        elif event.inaxes == self.axs[0] or event.inaxes == self.axs[1]:
            self.dragging = "span"
            # print("setting line", event.xdata)
            # clear span
            self.set_active_tier_boundary(0, -1)
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
        if self.dragging == "span":
            # Draw the new temporary vertical span
            x = min(self.start_x, end_x)
            w = abs(self.start_x - end_x)
            self.set_active_span(x, w)
            s1 = self.current_span[0].get_x()
            s2 = s1+self.current_span[0].get_width()
            self.resize_play_buttons(s1, s2)
            # TODO: need to clear all tiers?
            # self.set_active_tier_segment(-1)
            # self.fig.canvas.draw_idle() # Redraw the canvas efficiently
            self.fig.canvas.draw_idle() # Redraw the canvas efficiently
        elif self.dragging == "segment":

            self.active_tier_boundary.set_xdata([x, x])
            pass

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


