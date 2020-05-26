# py_walk.py - library for gait recognition
#
# Sungcheol Kim @ IBM
#
# Version 1.0.0 - 2020/03/23
# Version 1.1.0 - 2020/04/05 - add panda DataFrame

import numpy as np
import pandas as pd
import os
import csv
from tqdm.notebook import tqdm

import matplotlib.pyplot as plt

from bokeh.plotting import figure, show, gridplot
from bokeh.io import export_png
from bokeh.models import Span

import wfdb

def view_signal(sources, fields, channels=[], start_idx=0, end_idx=-1, filename='temp.png') -> None:
    """view time series data

    Parameters
    ----------
    sources : 2D numpy array
        time series data with 3 acceleration, 3 angular velocity
    fields : dictionary
        'fs', 'sig_name' information
    channels : list, optional
        list for signal channels
        0-2 for acceleration,
        3-5 for angular velocity, by default []
    filename : str, optional
        save for figure, by default 'temp.png'
    """

    if len(channels) == 0:
        channels = range(sources.shape[1])

    fig, axs = plt.subplots(ncols=1, nrows=len(channels), figsize=(20,4*len(channels)), sharex='col')

    if len(channels) == 1:
        axs = [axs]

    for j, i in enumerate(channels):
        source = sources[start_idx:end_idx, i]
        _plot_single(source, fields, i, axs[j])
    fig.tight_layout()

    plt.show()


def _plot_single(source, fields, idx, axe, step_pos=None):
    if len(source) > 5000:
        source = sources[start_idx:start_idx+5000, idx]
    t = np.linspace(0, len(source)/fields['fs'], num=len(source))

    axe.plot(t, source, color='navy')
    axe.set_xlabel('Time')
    axe.set_ylabel(fields['sig_name'][idx])

    if len(step_pos) > 0:
        axe.plot(t[step_pos], source[step_pos], 'o', color='red', alpha=0.5, markersize=12)

    return axe


def _detect_bout(source, window=10, threshold=0.08, bout_dis=80,
        bout_dur=300, show_flag=False, debug_flag=False) -> list:
    """detect high standard deviation session for bout activity

    Parameters
    ----------
    source : 1D data
        time series data for bout detection
    window : int, optional
        window size for standard deviation, by default 10
    threshold : float, optional
        std threshold for bout activity, by default 0.08
    bout_dis : int, optional
        inter bout sessioin distance, by default 80
    bout_dur : int, optional
        minimum bout duration in window size unit, by default 300
    show_flag : bool, optional
        show figure, by default True
    debug_flag : bool, optional
        set debug flag for details, by default False

    Returns
    -------
    list of tuple of two index
        one tuple corresponding to one bout
    """

    # calculate sd for window
    n = len(source)
    n_source = np.reshape(source[:n//window*window], (n//window, window))
    sd_source = np.std(n_source, axis=1)
    windowid = np.arange(len(sd_source))

    boutid = windowid[np.where(sd_source > threshold)]
    if (debug_flag): print(boutid)
    bout_list = []

    if (len(boutid) > 0):
        # detect continous bout (inter distance 100 windows)
        n_boutid = np.zeros(len(boutid)+2)
        n_boutid[0] = -1000
        n_boutid[-1] = boutid[-1] + 1000
        n_boutid[1:-1] = boutid
        ii = [i for i in range(len(n_boutid)-1) if (n_boutid[i+1] - n_boutid[i]) > bout_dis]
        last_window = n_boutid[ii]
        ii = [i for i in range(1, len(n_boutid)) if (n_boutid[i] - n_boutid[i-1]) > bout_dis]
        first_window = n_boutid[ii]

        for i in range(len(first_window)-1):
            if (last_window[i+1] - first_window[i] > bout_dur):
                bout_list.append((first_window[i], last_window[i+1]))
        if (debug_flag): print(bout_list)

    # show in time series
    if show_flag and (n < 5000):
        f = figure(width=950, height=200, y_range=[min(sd_source), max(sd_source)],
                   title='standard deviation in window size {}, interdistance {}'.format(window, window*bout_dis))
        f.line(windowid, sd_source, color='navy')
        f.circle(boutid, sd_source[boutid], size=7, color='red', alpha=0.5)
        for i in range(len(bout_list)):
            bouts_start = Span(location=bout_list[i][0], dimension='height', line_color='green',
                            line_dash='dashed', line_width=1.5)
            f.add_layout(bouts_start)
            bouts_stop = Span(location=bout_list[i][1], dimension='height', line_color='blue',
                            line_dash='dashed', line_width=1.5)
            f.add_layout(bouts_stop)

        show(f)

    for i in range(len(bout_list)):
        bout_list[i] = (bout_list[i][0]*window, bout_list[i][1]*window)

    return bout_list


def _get_bout(source, bout_list, pos=0) -> list:
    """return time series data in one bout

    Parameters
    ----------
    source : 1D numpy array
        time series data
    bout_list : list of bout information
        outcome of detect bout
    pos : int, optional
        index position in bout list, by default 0
    window : int, optional
        window size in bout detection, by default 10

    Returns
    -------
    1D numpy array
        time series data in bout
    """

    if (pos <0) or (pos > len(bout_list)-1):
        print('out of range')
        exit(0)

    start_idx = int(bout_list[pos][0])
    end_idx = int(bout_list[pos][1])

    return source[start_idx:end_idx]


def view_bout(sources, fields, bout_list, pos=0, channels=[], step_pos=[]) -> None:
    """view signals by bout unit

    Parameters
    ----------
    source : 1D numpy array
        time series data
    fields : dictionary
        meta data
    bout_list : list of tuples
        bout information list
    pos : int, by default 0
        position in bout list
    channels : list, optional
        list of channels, by default [] for all
    step_pos : list, optional
        list of detected step positions, by default []
    """

    if len(channels) == 0:
        channels = range(sources.shape[1])

    fig, axs = plt.subplots(ncols=1, nrows=len(channels), figsize=(20,4*len(channels)), sharex='col')

    if len(channels) == 1:
        axs = [axs]

    for j, i in enumerate(channels):
        source = _get_bout(sources[:, i], bout_list, pos=pos)
        _plot_single(source, fields, i, axs[j], step_pos=step_pos)

    fig.tight_layout()

    plt.show()


def smooth(x,window_len=11,window='hanning') -> list:
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")


    window_len = int(window_len)
    if window_len < 3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    s = np.r_[x[window_len-1:0:-1], x, x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w = np.ones(window_len,'d')
    else:
        w = eval('np.'+window+'(window_len)')

    y = np.convolve(w/w.sum(),s,mode='valid')
    return y


def _count_steps(source, threshold=0.6, th_dur=0.6, window_len=15, initial=30) -> list:
    """detect index positions in time series data corresponding to the starting point of step

    Parameters
    ----------
    source : 1D numpy array
        time series data
    threshold : float, optional
        this determine the threshold ratio between highest value in a step and begining of new step, by default 0.6
    th_dur : float, optional
        this determine the range of search for local minimum in terms of step duration, by default 0.6
    window_len : int, optional
        window size for smoothing, by default 15
    initial : int, optional
        initial size of typical step duration, by default 30

    Returns
    -------
    list
        array of index for step starting position
    """

    source = np.array(source).flatten()

    s_source = smooth(source, window_len=window_len)

    # check local minimum by slope
    k = 0
    min_sig = s_source[k]
    i = int(window_len/2)

    step_pos = []
    step_dur = [initial]

    while i < len(source):
        local_max = np.max(source[k:i])
        th = (local_max - min_sig)*threshold
        #print(i, th, max_sig-source[i])
        if (s_source[i] - min_sig) <= th:
            dur = int(np.mean(step_dur)*th_dur)
            i0 = 0 if (i-dur) < 0 else i-dur
            i1 = len(s_source) if (i+dur) > len(s_source) else i+dur
            local_min = np.min(s_source[i0:i1])
            if s_source[i] == local_min:
                step_dur.append(i - k + 1)
                k = i
                step_pos.append(k-int(window_len/2))
                min_sig = s_source[i]
        else:
            if s_source[i] < min_sig:
                min_sig = s_source[i]

        i = i + 1

    return np.array(step_pos)


def cal_stepinfo(signals, fields, step_poslist, show_flag=True) -> pd.DataFrame:
    """collect step information as duration, amplitude, balance

    Parameters
    ----------
    signals : full numpy array
        time series of full 6D data
    step_poslist : list
        step position list
    show_flag : logic, optional
        flag for showing figure

    Returns
    -------
    duration, amplitude, balance
        for each step, the duration in seconds, range of acceleration, total sum of yaw velocity
    """

    df = pd.DataFrame()
    df['duration'] = np.zeros(len(step_poslist)-1)
    df['ap_ptp'] = df.duration
    df['v_ptp'] = df.duration
    df['ml_ptp'] = df.duration
    df['balance'] = df.ap_ptp

    for i in range(len(step_poslist)-1):
        i0 = step_poslist[i]
        i1 = step_poslist[i+1]+1

        df.duration.iloc[i] = (i1 - i0)/fields['fs']
        df.ap_ptp.iloc[i] = np.ptp(signals[i0:i1, 2])      # ap-acc 2
        df.ml_ptp.iloc[i] = np.ptp(signals[i0:i1, 1])      # ml-acc 1
        df.v_ptp.iloc[i] = np.ptp(signals[i0:i1, 0])       # v-acc 0

        length = (i1-i0)//2
        w = 100 if length > 100 else 1
        i0 = i0-length if i0 - length - w>= 0 else 0
        i1 = i1-length
        #df.balance.iloc[i] = sum(subtract_movingaverage(signals[i0-w:i1, 3], w=w))        # yaw-velocity 3
        df.balance.iloc[i] = sum(signals[i0:i1, 3])        # yaw-velocity 3

    df['LR'] = ['green' if x > 0 else 'red' for x in df.balance]

    if show_flag:
        plot_stepdb(df)

    return df


def plot_stepdb(df) -> None:
    """plot step distribution

    Parameters
    ----------
    df : DataFrame
        outcome of cal_stepinfo with duration, ap_ptp, ml_ptp, balance, LR
    """

    f_amp = figure(width=300, height=300, title='Step Duration vs Amplitude',
            x_axis_label='Duration [s]', y_axis_label='ptp Magnitude [g]')
    f_amp.circle(df.duration, df.ap_ptp, color=df.LR)
    f_bal = figure(width=300, height=300, title='Balance Check',
            x_axis_label='Duration Right [s]', y_axis_label='Duration Left [s]')
    f_bal.circle(df.duration.loc[df.balance > 0], df.duration.loc[df.balance <=0], color='black')
    f_bal.line([df.duration.min(), df.duration.max()], [df.duration.min(), df.duration.max()])

    fig = gridplot([f_amp, f_bal], ncols=2)

    show(fig)

    export_png(fig, filename='steps_info.png')


def print_stepdb(df) -> pd.DataFrame:
    """create summary DataFrame from step info db

    Parameters
    ----------
    df : DataFrame
        outcome of cal_stepinfo with duration, ap_ptp, ml_ptp, balance, LR

    Returns
    -------
    pd.DataFrame
        summary dataframe for one patient
    """

    res = pd.DataFrame()

    if len(df) == 0: return res

    res['dur_mean'] = [df.duration.mean()]
    res['dur_std'] = [df.duration.std()]
    res['dur_25q'] = [df.duration.quantile(.25)]
    res['dur_50q'] = [df.duration.quantile(.50)]
    res['dur_75q'] = [df.duration.quantile(.75)]
    res['dur_skew'] = [df.duration.skew()]
    res['dur_kurtosis'] = [df.duration.kurtosis()]
    res['dur_mean_l'] = [df.duration.loc[df.LR == 'green'].mean()]
    res['dur_std_l'] = [df.duration.loc[df.LR == 'green'].std()]
    res['dur_mean_r'] = [df.duration.loc[df.LR == 'red'].mean()]
    res['dur_std_r'] = [df.duration.loc[df.LR == 'red'].std()]
    res['ap_mean'] = [df.ap_ptp.mean()]
    res['v_mean'] = [df.v_ptp.mean()]
    res['ml_mean'] = [df.ml_ptp.mean()]

    return res


def _get_parameters(fields, **kwargs):
    """obtain detection parameters from setting file or arguments

    Parameters
    ----------
    fields : dictionary
        meta information about signals

    Returns
    -------
    (bout_threshold, bout_dis, bout_duration, threshold, window_len, th_dur)
        parameters for bout_detection and step_detection
    """

    par_fname = './detect_par.csv'
    par_item = {'id': [fields['tag']],
            'window': [kwargs.get('window', 10)],
            'bout_threshold': [kwargs.get('bout_threshold', 0.06)],
            'bout_dis':[kwargs.get('bout_dis', 15)],
            'bout_dur':[kwargs.get('bout_dur', 150)],
            'threshold':[kwargs.get('threshold', 0.5)],
            'window_len':[kwargs.get('window_len', 15)],
            'th_dur':[kwargs.get('th_dur', 0.6)]}

    # check file
    if not os.path.exists(par_fname):
        print('... create parameter file: {}'.format(par_fname))
        pardb = pd.DataFrame(par_item)
        pardb.to_csv(par_fname, index=False)
    else:
        #print('... read parameter file: {}'.format(par_fname))
        pardb = pd.read_csv(par_fname)
        check = any(pardb.id == fields['tag'])
        update = len(kwargs) != 0
        # check item
        if not check:
            print('... append parameter file: {}'.format(par_fname))
            tmp = pd.DataFrame(par_item)
            pardb = pardb.append(tmp)
            pardb.to_csv(par_fname, index=False)
        elif update:
            for key, value in kwargs.items():
                if key in pardb.columns:
                    print('... update parameter file: {} - {}'.format(key, value))
                    pardb.loc[pardb.id == fields['tag'], key] = value
            pardb.to_csv(par_fname, index=False)

    window = int(pardb.loc[pardb.id == fields['tag'], 'window'])
    bout_threshold = float(pardb.loc[pardb.id == fields['tag'], 'bout_threshold'])
    bout_dis = int(pardb.loc[pardb.id == fields['tag'], 'bout_dis'])
    bout_dur = int(pardb.loc[pardb.id == fields['tag'], 'bout_dur'])
    threshold = float(pardb.loc[pardb.id == fields['tag'], 'threshold'])
    window_len = int(pardb.loc[pardb.id == fields['tag'], 'window_len'])
    th_dur = float(pardb.loc[pardb.id == fields['tag'], 'th_dur'])

    return window, bout_threshold, bout_dis, bout_dur, threshold, window_len, th_dur


def get_gait(signals, fields, idx=2, show_flag=False, filename='dur_amp.png', **kwargs) -> (pd.DataFrame, pd.DataFrame):
    """obtain gait information as duration, amplitutde, balance list

    Parameters
    ----------
    signals : 6D numpy array
        full time series signal
    fiels : dictionary
        meta information on channels of signals
    idx : int, optional
        channel index for step detection, by default 2
        patient identification shown in figure, by default 'CO'
    show_flag : logic, optional
        flag for showing figure, by default False
    filename : str, optional
        file name for saving figure, by default 'dur_amp.png'
    **kwargs : optional options
        for bout detection - bout_threshold, bout_dis, bout_dur
        for step detection - threshold, window_len, th_dur

    Returns
    -------
    step_db, summary_db
        step database for all detected bouts, summary database for one file
    """

    window, bout_threshold, bout_dis, bout_dur, threshold, window_len, th_dur = _get_parameters(fields, **kwargs)

    boutlist = _detect_bout(signals[:, idx], show_flag=show_flag, window=window, threshold=bout_threshold, bout_dis=bout_dis, bout_dur=bout_dur)

    if len(boutlist) == 0:
        print('... detect no bouts. bout_threshold={}. bout_dis={}, bout_dur={}'.format(bout_threshold, bout_dis, bout_dur))
        return pd.DataFrame(), pd.DataFrame()

    stepdb = pd.DataFrame()
    for i in range(len(boutlist)):
        steplist = _count_steps(_get_bout(signals[:, idx], boutlist, pos=i),
                threshold=threshold, th_dur=th_dur, window_len=window_len)
        if show_flag:
            view_bout(signals, fields, boutlist, pos=i, channels=[idx], step_pos=steplist)

        tmp = cal_stepinfo(signals, fields, steplist, show_flag=show_flag)
        stepdb = stepdb.append(tmp)

    if show_flag: plot_stepdb(stepdb)

    summary = print_stepdb(stepdb)
    summary['bout_n'] = len(boutlist)
    summary['step_n_left'] = sum(stepdb.LR == 'green')
    summary['step_n_right'] = sum(stepdb.LR == 'red')
    summary['ID'] = fields['tag']

    return stepdb, summary


def load_ltmm_data(patient_id='co001_base') -> (list, dict):
    """load ltmm data using wfdb from local directory

    Parameters
    ----------
    patient_id : str, optional
        patient data id, by default 'co001_base'

    Returns
    -------
    list, dict
        numpy array with (time series, channel #), dictionary that has meta information such as 'fs', 'sig_len', 'units', 'sig_name'
    """

    base_dir = 'physionet.org/files/ltmm/1.0.0/'
    if patient_id.find('base') > -1:
        filename = base_dir + 'LabWalks/' + patient_id
    else:
        filename = base_dir + patient_id

    sig, fields = wfdb.rdsamp(filename)

    # set fields
    fields['tag'] = patient_id
    fields['filename'] = filename
    sig_name = fields['sig_name']
    sig_name = [ x.split(' ')[-1] for x in sig_name]
    fields['sig_name'] = sig_name
    fields['par_fname'] = filename + '.par'

    # check and read fields
    if not os.path.exists(fields['par_fname']):
        with open(fields['par_fname'],'w') as f:
            w = csv.writer(f)
            w.writerows(fields.items())
    else:
        with open(fields['par_fname']) as f:
            reader = csv.reader(f)
            for row in reader:
                if row[0] not in fields:
                    try:
                        fields[row[0]] = float(row[1])
                    except:
                        fields[row[0]] = row[1]

    return sig, fields


def movingaverage(source, w=100):
    """ substract moving average over w points
    input:
        source
        w
    output:
        background
    """

    # if w=1, return original
    if w == 1:
        return source

    # set total length
    source = np.array(source).flatten()
    ret = np.cumsum(source, dtype=float)
    ret[w:] = ret[w:] - ret[:-w]
    b_source = ret[w-1:]/w

    return b_source

# vim:foldmethod=indent
