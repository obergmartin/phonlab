__all__=['get_rms','get_f0_sift','get_f0_srh','get_f0_ac','get_f0_acd']

import numpy as np
from scipy.signal import windows, find_peaks, spectrogram, peak_prominences, fftconvolve
from scipy import fft
from librosa import feature, util, lpc
from pandas import DataFrame
from ..utils.prep_audio_ import prep_audio


def get_rms(y, fs, scale=False):
    """Measure the time-varying root mean square (RMS) amplitude of the signal in **y**.

    Parameters
    ==========
        y : ndarray
            A one-dimensional array of audio samples
        fs : int
            Sampling rate of **x**
        scale : boolean, default = False
            optionally scale the rms amplitude to maximum peak.

    Returns
    =======
        df: pandas DataFrame
            There are two columns in the returned frame - sec, rms.

    """

    # constants and global variables
    frame_length_sec = 0.04
    step_sec = 0.005

    frame_length = int(fs * frame_length_sec) 
    half_frame = frame_length//2
    step = int(fs * step_sec)  # number of samples between frames

    rms = feature.rms(y=y,frame_length=frame_length, hop_length=step,center=False)
    if scale:
        rms = 20*np.log10(rms[0]/np.max(rms[0]))
    else:
        rms = 20*np.log10(rms[0])

    nb = rms.shape[0]  # the number of frames
    sec = (np.array(range(nb)) * step + half_frame).astype(int)/fs

    return DataFrame({'sec': sec, 'rms':rms})

def get_f0_sift(y, fs, f0_range = [63,400], pre = 0.94):
    """Track the fundamental frequency of voicing (f0), using a time-domain method.

    The method in this function is an implementation of John Markel's (1972) simplified inverse filter tracking algorithm (SIFT) which is also used in track_formants().  LPC coefficients are calculated for each frame and the audio signal is inverse filtered with these, resulting in a quasi glottal waveform. Then autocorrelation is used to estimate the fundamental frequency.  Probability of voicing is given from a logistic regression formula using `rms` and `c` trained to predict the voicing state as determined by EGG data using the function `phonlab.egg_to_oq()` over the 10 speakers in the ASC corpus of Mandarin speech. The prediction of the EGG voicing decision was about 85% correct.

    Parameters
    ==========
        y : ndarray
            A one-dimensional array of audio samples
        fs : int
            Sampling rate of **x**
        f0_range : list of two integers, default = [63,400]
            The lowest and highest values to consider in pitch tracking.

    Returns
    =======
        df: pandas DataFrame  
            measurements at 5 msec intervals.

    Note
    ====
    The columns in the returned dataframe are for each frame of audio:
        * sec - time at the midpoint of each frame
        * f0 - estimate of the fundamental frequency
        * rms - peak normalized rms amplitude in the band from 0 to 8000 Hz
        * c - value of the peak autocorrelation found in the frame
        * probv - estimated probability of voicing
        * voiced - a boolean, true if probv>0.5

    References
    ==========
    Markel, J. (1972) The SIFT algorithm for fundamental frequency estimation. `IEEE Transactions on Audio and Electroacoustics`, 20(5), 367 - 377

    Example
    =======

    .. code-block:: Python
    
         x,fs = phon.loadsig("sf3_cln.wav",chansel=[0])
         f0df = get_f0(x, fs, f0_range= [63,400])
        
         ret = phon.sgram(x,fs,cmap='Blues') # draw the spectrogram from the array of samples
         ax1 = ret[0]  # the first item returned, is the matplotlib axes of the spectrogram
         ax2 = ax1.twinx()
         ax2.plot(f0df.sec,f0df.f0, 'go')  

    .. figure:: images/get_f0.png
       :scale: 90 %
       :alt: a 'bluescale' spectrogram with red dots marking the f0
       :align: center

       Marking the f0 found by `phon.get_f0()`

   """
    # constants and global variables
    frame_length_sec = 0.04
    step_sec = 0.005
    
    x, fs = prep_audio(y, fs, target_fs=16000, pre = 0, quiet=True)  # no preemphasis, for RMS calc

    frame_length = int(fs * frame_length_sec) 
    half_frame = frame_length//2
    step = int(fs * step_sec)  # number of samples between frames

    rms = feature.rms(y=x,frame_length=frame_length, hop_length=step,center=False)
    rms = 20*np.log10(rms[0]/np.max(rms[0]))

    if (pre > 0): x = np.append(x[0], x[1:] - pre * x[:-1])  # now apply pre-emphasis

    w = windows.hamming(frame_length)
    frames = util.frame(x, frame_length=frame_length, hop_length=step,axis=0)    
    frames = np.multiply(frames,w)   # apply a Hamming window to each frame, for lpc

    nb = frames.shape[0]  # the number of frames (or blocks) in the LPC analysis
    f0 = np.empty(nb)
    c = np.empty((nb))

    sec = np.array([(i * (step/fs)+(frame_length_sec/2)) for i in range(nb)])  
    A = lpc(frames, order=32,axis=-1)  # get LPC coefs, can use a largish order
    xi = fftconvolve(frames,A,mode="same",axes=1) # inverse filter

    th = fs//f0_range[1]
    tl = fs//f0_range[0]
    
    for i in range(nb): 
        cormat = np.correlate(xi[i], xi[i], mode='full') # autocorrelation 
        ac = cormat[cormat.size//2:] # the autocorrelation is in the last half of the result
        idx = np.argmax(ac[th:tl]) + th # index of peak correlation (in range lowest to highest)
        f0[i] = 1/(idx/fs)      # converted to Hz
        if (ac[0]<= 0) | (ac[idx] <= 0):  # avoid srt of a negative number
            c[i] = 0
        else:
            c[i] = np.sqrt(ac[idx]) / np.sqrt(ac[0])

    odds = np.exp(-0.3 + (0.13*rms[:nb]) + (6.46*c[:nb]))  # logistic formula, trained on ASC corpus
    probv = odds / (1 + odds)
    voiced = probv > 0.5

    return DataFrame({'sec': sec, 'f0':f0, 'rms':rms, 'c':c, 'probv': probv, 'voiced':voiced})
    

def get_f0_srh(y, fs, f0_range = [60,400]):
    """Track the fundamental frequency of voicing (f0), using a frequency domain method.

This function is an implementation of Drugman and Alwan's (2011) "Summation of 
Residual Harmonics" (SRH) method of pitch tracking.  The signal is downsampled to 
10 kHz, and inverse filtered with LPC analysis to remove the influence of vowel 
formants. Then harmonics are found in the spectrum of the residual signal.
Drugman and Alwan found that this technique provides an estimate of F0 that is 
robust when the audio signal is corrupted by noise. 

Probability of voicing is given from a logistic regression formula using `rms` and `srh` trained to predict the voicing state as determined by EGG data using the function `phonlab.egg_to_oq()` over the 10 speakers in the ASC corpus of Mandarin speech. The prediction of the EGG voicing decision was about 75% correct.

Parameters
==========
    y : string or ndarray
        A one-dimensional array of audio samples
    fs : int
        Sampling rate of **x**
    f0_range : list of two integers, default = [63,400]
        The lowest and highest values to consider in pitch tracking. This algorithm is quite sensitive to the values given in this setting.

Returns
=======
    df : pandas DataFrame  
        measurements at 5 msec intervals.

Note
====
The columns in the returned dataframe are for each frame of audio:
    * sec - time at the midpoint of each frame
    * f0 - estimate of the fundamental frequency
    * rms - peak normalized rms amplitude in the band from 0 to 5 kHz
    * srh - value of SRH (normalized sum of the residual harmonics)
    * probv - estimated probability of voicing
    * voiced - a boolean, true if probv>0.5

References
==========

Drugman, Thomas & Alwan, Abeer (2011) Joint robust voicing detection 
and pitch estimation based on residual harmonics. ISCA (Florence, Italy) pp. 1973ff

    
    """
    frame_length_sec = 0.04
    step_sec = 0.005
    fs_khz = 10
    
    x, fs = prep_audio(y, fs, target_fs=fs_khz*1000, pre = 0.96, quiet=True)  
    frame_length = int(fs * frame_length_sec) 
    half_frame = frame_length//2
    step = int(fs * step_sec)  # number of samples between frames

    rms = feature.rms(y=x,frame_length=frame_length, hop_length=step,center=False)
    rms = 20*np.log10(rms[0]/np.max(rms[0]))

    w = windows.hamming(frame_length)
    frames = util.frame(x, frame_length=frame_length, hop_length=step,axis=0)    
    frames = np.multiply(frames,w)   # apply a Hamming window to each frame, for lpc

    nb = frames.shape[0]  # the number of frames (or blocks) in the LPC analysis
    sec = (np.array(range(nb)) * step + half_frame).astype(int)/fs

    A = lpc(frames, order=int(fs_khz*2+2))  # get lpc coefficients
    filtered_frames = fftconvolve(frames,A,mode="same",axes=1) # inverse filter
    Sxx = np.abs(np.fft.rfft(filtered_frames,2**14))           # spectrum of inverse
                     
    f0 = np.empty(nb)
    c = np.empty((nb))

    for i in range(nb): 
        S = Sxx[i]
        T = len(S)/fs
        srh_max = 0
        max_harmonic = 6
        for f in range(f0_range[0], f0_range[1]): 
            fT = int(f*T)  # test this as frequency of H1
            h = S[fT]
            for k in range(2,max_harmonic):
                h += S[fT*k] - S[int(fT*(k-0.5))]
            srh = h/(max_harmonic-1)
            if srh > srh_max:
                srh_max = srh
                f0[i] = f        
        c[i] = srh_max

    odds = np.exp(2.75 + (0.12*rms) + (0.4*c))  # logistic formula, trained on ASC corpus
    probv = odds / (1 + odds)
    voiced = probv > 0.5

    return DataFrame({'sec': sec, 'f0':f0, 'rms':rms, 'srh':c, 'probv': probv, 'voiced':voiced})

def get_f0_ac(y, fs, f0_range = [60,400]):
    """Track the fundamental frequency of voicing (f0), using a time domain method.

    This function implements a simple autocorrelation method of pitch tracking with no filtering prior to calculating the autocorrelation. Probability of voicing is given from a logistic regression formula using `rms` and `c` trained to predict the voicing state as determined by EGG data using the function `phonlab.egg_to_oq()` over the 10 speakers in the ASC corpus of Mandarin speech. The prediction of the EGG voicing decision was about 88% correct.

    Parameters
    ==========
        y : ndarray
            A one-dimensional array of audio samples
        fs : int
            Sampling rate of **x**
        f0_range : list of two integers, default = [63,400]
            The lowest and highest values to consider in pitch tracking.

    Returns
    =======
        df: pandas DataFrame  
            measurements at 5 msec intervals.

    Note
    ====
    The columns in the returned dataframe are for each frame of audio:
        * sec - time at the midpoint of each frame
        * f0 - estimate of the fundamental frequency
        * rms - peak normalized rms amplitude in the band from 0 to fs/2
        * c - value of the peak autocorrelation found in the frame
        * probv - estimated probability of voicing
        * voiced - a boolean, true if probv>0.5
    """

    # constants and global variables
    frame_length_sec = 1/f0_range[0]
    step_sec = 0.005

    # no preemphasis, up-sample
    x, fs = prep_audio(y, fs, target_fs = 24000, pre = 0, quiet=True)  

    frame_length = int(fs * frame_length_sec) 
    half_frame = frame_length//2
    step = int(fs * step_sec)  # number of samples between frames

    rms = feature.rms(y=x,frame_length=frame_length, hop_length=step,center=False)
    rms = 20*np.log10(rms[0]/np.max(rms[0]))

    frames = util.frame(x, frame_length=frame_length, hop_length=step,axis=0)    

    nb = frames.shape[0]  # the number of frames (or blocks) in the LPC analysis

    sec = (np.array(range(nb)) * step + half_frame).astype(int)/fs

    f0 = np.empty(nb)
    c = np.empty((nb))

    th = fs//f0_range[1]
    tl = fs//f0_range[0]
    
    for i in range(nb): 
        cormat = np.correlate(frames[i], frames[i], mode='full') # autocorrelation 
        ac = cormat[cormat.size//2:] # the autocorrelation is in the last half of the result
        idx = np.argmax(ac[th:tl]) + th # index of peak correlation (in range lowest to highest)
        f0[i] = 1/(idx/fs)      # converted to Hz
        if (ac[0]<= 0) | (ac[idx] <= 0):
            c[i] = 0
        else:
            c[i] = np.sqrt(ac[idx]) / np.sqrt(ac[0])

    odds = np.exp(0.48 + (0.19*rms) + (5.44*c))  # logistic formula, trained on ASC corpus
    probv = odds / (1 + odds)
    voiced = probv > 0.5

    return DataFrame({'sec': sec, 'f0':f0, 'rms':rms, 'c':c, 'probv': probv, 'voiced':voiced})

def f0_from_harmonics(f_p,i,h,nh):  
    ''' Assign harmonic numbers to the peaks in f_p -- this function is used in get_f0_acd
    
        f_p: an array of peak frequencies
        i: the starting peak to look at (0,n)
        h: the starting harmonic number to assign to this peak (1,n-1)
    '''
    Np = len(f_p)  # number of peaks
    m = np.zeros(Np)
    f0 = []
    m[i] = h
    f0 = np.append(f0, f_p[i]/h)  # f0 if peak i is harmonic h
    thresh = 0.055 * f0[0]  # 5.5% of the f0 value
    ex = 0  # number of harmonics over h=11

    for j in range(i+1,Np):  # step through the spectral peaks
        lowest_deviation = 1000
        best_f0 = np.nan
        for k in range(h+1,nh+1):  # step through harmonics
            test_f0 = f_p[j]/k
            deviation = abs(test_f0 - f0[0])
            if deviation < lowest_deviation: # pick the best harmonic number for this peak
                lowest_deviation = deviation
                best_f0 = test_f0
                best_k = k
        if lowest_deviation < thresh:  # close enough to be a harmonic
            m[j] = best_k
            f0 = np.append(f0,best_f0)
            if (h>11): ex = ex + 1
            h=h+1
    C = ((h-1) + (Np - ex))/ np.count_nonzero(m)

    mean_f0 = np.average(f0,weights=np.arange(len(f0))+1)
    return C,mean_f0 
    
def get_f0_acd(y, fs, prom=14, f0_range=[60,400], min_height = 0.6, test_time=-1):
    """Track the fundamental frequency of voicing, using a frequency domain method.

This function implements the 'approximate common denominator" algorithm proposed by Aliik, Mihkla and Ross (1984), which was an improvement on the method proposed by Duifuis, Willems and Sluyter (1982).  The algorithm finds candidate harmonic peaks in the spectrum, and chooses a value of f0 that best predicts the harmonic pattern.  One feature of this method is that it reports a voice quality measure (the difference in the amplitudes of harmonic 1 and harmonic 2).

Probability of voicing is given from a logistic regression formula using `rms` and Duifuis et al.'s `c` parameter trained to predict the voicing state as determined by EGG data using the function `phonlab.egg_to_oq()` over the 10 speakers in the ASC corpus of Mandarin speech. The prediction of the EGG voicing decision was about 86% correct.
Aliik et al. (1984) used a cutoff of C<3.5 as a voicing threshold.

Parameters
==========
    y : ndarray
        A one-dimensional array of audio samples
    fs : int
        the sampling rate of the audio in **y**.
    prom : numeric, default = 14 dB
        In deciding whether a peak in the spectrum is a possible harmonic, this prominence value is passed to scipy.find_peaks().  A larger value means that the spectral peak must be more prominent to be considered as a possible harmonic peak, and thus the algorithm is less likely to report pitch values when the parameter is given a high value.  In general, 20 is a high value, and 3 is low.
    f0_range : a list of two integers, default=[60,400]
        The lowest and highest values to consider in pitch tracking. The algorithm is not particularly sensitive to this parameter, but it can be useful in avoiding pitch-halving or pitch-doubling.
    min_height: numeric, default = 0.6
        As a proportion of the range between the lowest amplitude in the spectrum and the highest, only peaks above `min_height` will be considered to be harmonics. The value that is passed to find_peaks() is: `amplitude_min + min_height*(amplitude_range)`. 

Returns
=======
    df: pandas DataFrame  
        measurements at 5 msec intervals.

Note
====
    The columns in the returned dataframe are for each frame of audio:
        * sec - time at the midpoint of each frame
        * f0 - estimate of the fundamental frequency
        * rms - rms amplitude in a low frequency band from 0 to 1200 Hz
        * h1h2 - the difference in the amplitudes of the first two harmonics (H1 - H2) in dB
        * probv - estimated probability of voicing
        * voiced - a boolean, true if probv>0.5

Example
=======

.. code-block:: Python
    
    example_file = importlib.resources.files('phonlab') / 'data' / 'example_audio' / 'stereo.wav'

    x,fs = phon.loadsig(example_file, chansel=[0])
    f0df = phon.get_f0_acd(x,fs,prom=18)

    snd = parselmouth.Sound(str(example_file)).extract_left_channel()  # create a Praat Sound object
    pitch = snd.to_pitch()  # create a Praat pitch object
    f0df2 = phon.pitch_to_df(pitch)  # convert it into a Pandas dataframe

    ret = phon.sgram(x,fs,cmap='Grays') # draw a spectrogram of the sound

    f0_range = [60,400]

    ax1 = ret[0]  # get the plot axis
    ax2 = ax1.twinx()  # and twin it for plotting f0
    ax2.plot(f0df2.sec,f0df2.f0, color='chartreuse',marker="s",linestyle = "none")
    ax2.plot(f0df.sec,f0df.f0, color='dodgerblue',marker="d",linestyle = "none")  
    ax2.set_ylim(f0_range)
    ax2.set_ylabel("F0 (Hz)", size=14)
    for item in ax2.get_yticklabels(): item.set_fontsize(14)

.. figure:: images/acd_pitch_trace.png
    :scale: 33 %
    :alt: a spectrogram with a pitch trace calculated by get_f0_acd
    :align: center

    Comparing the f0 found by `phon.get_f0_acd()` plotted with blue diamonds, and the f0 
    values found by `parselmouth` `to_Pitch()`, plotted with chartreuse dots.

    """
    nh = 6  # maximum number of harmonics to consider
    down_fs = nh*400  # down sample frequency
    x, fs = prep_audio(y, fs, target_fs = down_fs, pre=0,quiet=True)  # no preemphasis
    
    step_sec = 0.005  # a new pitch estimate every 5 ms
    N = 1024    # FFT size

    frame_len = int(fs*0.04)  # 40 ms frame
    step = int(fs*step_sec)  # stride between frames
    noverlap = frame_len - step   # points of overlap between successive frames

    while (frame_len > N): N = N * 2  # increase fft size if needed
    w = windows.hamming(frame_len)
    f,ts,Sxx = spectrogram(x,fs=fs,noverlap = noverlap, window=w, nperseg = frame_len, 
                              nfft = N, scaling = 'spectrum', mode = 'magnitude', detrend = 'linear')
    rms = 20 * np.log10(np.sqrt(np.divide(np.sum(np.square(Sxx),axis=0),len(f)))) 
    Sxx = 20 * np.log10(Sxx)

    nb = len(ts)  # the number of frames in the spectrogram
    f0 = np.full(nb,np.nan)  # array filled with nan
    h1h2 = np.full(nb,np.nan)        # array filled with nan
    c = np.full(nb,5.0)      # default value of c is 5.0
        
    min_dist = int(f0_range[0]/(fs/N)) # min distance btw harmonics
    max_dist = int(f0_range[1]/(fs/N))
    dist = int((min_dist + max_dist)/2)

    ## temp 
    if test_time>0:
        i_test = np.argmin(np.fabs(test_time-ts)) # the ts that is closest to this
    else: 
        i_test = -1
    ## temp
    
    for idx in range(nb):
        spec = Sxx[:,idx]
        height = np.min(spec) + min_height * np.abs(np.max(spec)-np.min(spec))  # required height of a peak
        peaks,props = find_peaks(spec, height = height, prominence=prom, distance = min_dist, wlen=dist)

        if len(peaks)>2:  # we did find some harmonics?
            for p in range(3):  # for each of the first three spectral peaks
                for h in range(1,5): # treat it as one of the first four harmonics
                    if (h==p*2): break
                    C,_f0 = f0_from_harmonics(f[peaks],p,h,nh)
                    
                    if idx==i_test: 
                        print(f'_f0: {_f0:0.1f}, peak: {p}, harmonic: {h}, C: {C:0.2f}')
                        
                    if (f0_range[0] < _f0) & (_f0 < f0_range[1]) & (C < c[idx]):
                        c[idx] = C      
                        i_f0 = np.argmin(np.fabs(_f0 - f)) # f index that is closest to f0
                        i_2f0 = np.argmin(np.fabs((2 * _f0) - f)) # closest to 2f0 for (h1h2)
                        h1h2[idx] = spec[i_f0] - spec[i_2f0]
                        f0[idx] = _f0

        if idx==i_test:  # show diagnostic info, only at a target frame
            if len(peaks)<nh: # the highest harmonic number to consider
                n = len(peaks)
            else:
                n = nh
            plt.plot(f,spec)
            plt.vlines(f[peaks[0:n]],np.min(spec),np.max(spec))
            plt.axhline(height)
            print("number of peaks: ",len(peaks), "start time: ", ts[0])
            print(f'min_dist = {min_dist}, max_dist = {max_dist}, down_fs={fs}, len(f)={len(f)}, N={N}')
            print(f"median difference between adjacent peaks {np.median(np.ediff1d(f[peaks[0:n]])):0.2f}")
            print(f"frequency of the lowest peak {f[peaks[0]]:0.2f}")
            print(f'mean prominence: {np.mean(props["prominences"][0:n]):0.3f} mean peak: {np.mean(props["peak_heights"][0:n]):0.3f}')
            print(f"height = {height:0.2f},max={np.max(spec):0.2f}, min={np.min(spec):0.2f}, c={best_c}")
            print(f"time = {test_time}, f0 = {f0[idx]:0.2f}, h1h2 = {h1h2[idx]:0.2f}")

    odds = np.exp(8.65 + (0.16*rms) - (0.83*c))  # logistic formula, trained on ASC corpus
    probv = odds / (1 + odds)
    voiced = probv > 0.5

    return DataFrame({'sec': ts, 'f0':f0, 'rms':rms, 'h1h2':h1h2, 'c':c, 'probv': probv, 'voiced':voiced})

