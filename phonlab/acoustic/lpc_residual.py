__all__=['lpcresidual']

import numpy as np
from librosa import util, lpc
from scipy.signal import windows, fftconvolve
from tensorflow.signal import overlap_and_add

from ..utils.prep_audio_ import prep_audio

def lpcresidual(y, fs, target_fs=16000, o = 18, l=0.04, s=0.005):    
    """Compute the residual signal which results from filtering the input array **y** using LPC inverse filtering.  
This signal is useful in voice quality and periodicity routines.

The LPC order is equal to (target_fs/1000) + 2, which is by default is 18.

Parameters
==========
    y : ndarray
        A one-dimensional array of audio samples
    fs : int
        Sampling rate of **y**
    target_fs: int, default = 16000
        Algorithms from the covarep library of voice analysis routines require target_fs=16000
    o: integer, default = 18
        The "order" of the LPC analysis.  The number of coefficients to use in the LPC analysis.  
        The default value is that recommended by Drugman, Kane, and Gobl (2013) for voice quality
        analysis (fs/1000 + 2), with the caveat that a smaller number may be more appropriate for 
        voices with higher fundamental frequency.
    l: float, default = 0.04
        The duration of the LPC analysis window, 40 milliseconds
    s: float, default = 0.005
        The interval between successive frames in the LPC analysis, 5 milliseconds

Returns
=======
    lpc_residual : ndarray
        A one-dimensional array -- the residual derived by inverse filtering the input 
        audio signal. It has the same number of samples as the input **y** array.
    fs : int
        The sampling rate of **lpc_residual**.  It will be the same as **target_fs**, which
        by default is 16000 Hz.

    
    """
    x, fs = prep_audio(y, fs, target_fs=target_fs, pre = 0, quiet=True)  # resample
    
    frame_length = int(fs * l) # number of samples in a frame
    step = int(fs * s)  # number of samples between frames, hop

    frames = util.frame(x, frame_length=frame_length, hop_length=step,axis=0)   # view as frames
    frames = np.multiply(frames,windows.hamming(frame_length))   # apply a Hamming window to each frame

    A = lpc(frames, order=o)  # get lpc coefficients
    inv = fftconvolve(frames,A,mode="same",axes=1) # inverse filter, 
    inv = inv * np.sum(np.square(frames))/np.sum(np.square(inv))
    
    lpc_resid = overlap_and_add(inv,step)  # put frames back together into waveform with overlap and add
    lpc_resid = lpc_resid/np.max(np.fabs(lpc_resid))

    # pad the lpc_residual to be the same length as the input
    npad = len(x)-len(lpc_resid)

    lpc_resid = np.pad(lpc_resid,(0,npad),mode='edge')  ## repeat the last sample npad times
    
    return lpc_resid,fs
