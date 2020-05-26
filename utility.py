# utility.py
#
# useful functions for gait analysis

import numpy as np

from nptyping import Array
from numba import njit
from numba.typed import List

from wfdb import processing

def calibration_gravity_parameters(signals:Array) -> Array:
    """calibration_gravity_parameters.

    :param signals:
    :type signals: Array
    :rtype: Array
    """

    n = signals.shape[0]    # row number of 2D array
    A = np.zeros((n, 6))
    row_unit = np.ones((1, n))

    for i in range(n):
        abs_a = np.sqrt(np.sum(signals[i,:]**2))
        A[i, :3] = signals[i, :]*signals[i, :]/abs_a
        A[i, 3:] = signals[i, :]/abs_a

    try:
        tmp = np.linalg.inv(np.matmul(A.T, A))
        res = np.linalg.multi_dot([row_unit, A, tmp]).flatten()
    except np.linalg.LinAlgError:
        print("... not invertable!")
        return [1,1,1,0,0,0]
    else:
        print(np.mean(signals, axis=0))
        print(np.std(signals, axis=0))
        new_signals = np.tile(res[:3], (n, 1))*signals + np.tile(res[3:], (n, 1))
        print(np.mean(new_signals, axis=0))

        return res


@njit(fastmath=True, cache=True)
def check_walk_autocorrelation(signal, detection_window_len:int=400, detection_window_step:int=200, min_auto_corr_score:float=0.75, min_time_lag:int=20, max_time_lag:int=130) -> (List, List):
    """check_walk_autocorrelation.

    :param signal: 1D numpy array signal
    :param detection_window_len: autocorrelation calculation window length
    :type detection_window_len: int
    :param detection_window_step: autocorrelation calculation window step size
    :type detection_window_step: int
    :param min_auto_corr_score: minimum autocorrelation threshold value
    :type min_auto_corr_score: float
    :param min_time_lag: minimum lag time
    :type min_time_lag: int
    :param max_time_lag: maximum lag time
    :type max_time_lag: int
    :rtype: (List, List)
    """

    walk_detection_result = List()
    [ walk_detection_result.append(False) for i in range(len(signal)) ]

    start = 0
    end = start + detection_window_len

    window_corr_list = List()
    while end <= len(signal):
        correlation_scores = []
        for lag in range(min_time_lag, max_time_lag):
            #corr = serial_corr(signal[start:end], lag=lag)
            corr = normalized_auto_corr(signal[start:end], lag=lag)
            correlation_scores.append(corr)

        #print(correlation_scores)
        window_corr_list.append(max(correlation_scores))

        #print(max(correlation_scores), min_auto_corr_score)
        if max(correlation_scores) > min_auto_corr_score:
            walk_detection_result[start:end] = [ True for i in range(start, end) ]

        start += detection_window_step
        end = start + detection_window_len

    return walk_detection_result, window_corr_list


@njit(fastmath=True, cache=True)
def serial_corr(wave: Array[float], lag:int = 1) -> float:
    """serial_corr. calculate numpy based autocorrelation

    :param wave: 1D numpy array for signal wave
    :type wave: Array[float]
    :param lag: lag time
    :type lag: int
    :rtype: float
    """
    n = len(wave)
    y1 = wave[lag:]
    y2 = wave[:n-lag]
    return np.corrcoef(y1, y2)[0, 1]


@njit(fastmath=True, cache=True)
def normalized_auto_corr(wave: Array[float], lag:int = 1) -> float:
    """normalized_auto_corr. calculate normalized autocorrelation

    :param wave: 1D numpy array for signal wave
    :type wave: Array[float]
    :param lag: lag time
    :type lag: int
    :rtype: float
    """
    # check lag and array size
    if len(wave) < 2*lag:
        return 0.

    # normalized autocorrelation
    mu_0 = np.mean(wave[:lag])
    mu_lag = np.mean(wave[lag:2*lag])

    sig_0 = np.std(wave[:lag])
    sig_lag = np.std(wave[lag:2*lag])

    if lag*sig_0*sig_lag == 0:
        return 0.

    return np.sum((wave[:lag] - mu_0)*(wave[lag:2*lag] - mu_lag))/(lag*sig_0*sig_lag)


@njit(fastmath=True, cache=True)
def find_long_period(bool_array: List, min_duration:int, scale:int) -> List:
    """find_long_period. identify long period of motion

    :param bool_array: bool array with check of motion
    :type bool_array: List
    :param min_duration: minimum duration in index unit
    :type min_duration: int
    :param scale: scaling factor for original array
    :type scale: int
    :rtype: List
    """

    s_ind, e_ind = None, None
    index_list = List()
    i = 0
    while i < len(bool_array):
        if bool_array[i]:
            if s_ind is None:
                # beginning of a bout
                s_ind = i
        else:
            if s_ind is not None:
                # end of current bout
                e_ind = i
                if (e_ind - s_ind) >= min_duration:
                    index_list.append([s_ind*scale, e_ind*scale])
                s_ind, e_ind = None, None
        i += 1

    # check last point
    if bool_array[-1]:
        if s_ind is not None:
            # end of current bout
            e_ind = len(bool_array)
            if (e_ind - s_ind) >= min_duration:
                index_list.append([s_ind*scale, e_ind*scale])

    return index_list


@njit(fastmath=True, cache=True)
def SC_method_slope(signal: Array[float], slope_ratio_threshold:float=0.6, search_window_ratio:float=0.6, initial_duration:int=30, direction='down') -> (List, List, List):
    """SC_method_slope. Step count method based on slope cross over counting

    :param signal: 1D numpy array
    :type signal: Array[float]
    :param slope_ratio_threshold: ratio between maximum height within one step and current height from minimum winthin step
    :type slope_ratio_threshold: float
    :param search_window_ratio: window size ratio compared to previous step duration
    :type search_window_ratio: float
    :param initial_duration: initial guess of step duration
    :type initial_duration: int
    :rtype: (List, List)
    """

    # check direction
    if direction == 'down':
        wave = signal
    else:
        wave = -signal

    # prepare initial parameters and list
    step_point, i = 0, 1
    min_sig = wave[step_point]

    step_pos = List()
    slope_list = List()
    ptp_list = List()
    dur_list = List()

    step_dur = np.ones(5)*initial_duration    # history of previous duration
    n_wave = len(wave)
    last_point = n_wave - int(step_dur.mean())

    # start searching by index i
    while i < last_point:
        # calculate local maximum
        local_slope_max = (np.max(wave[step_point:i]) - min_sig)

        # calculate slope
        slope = (wave[i] - min_sig)
        #print(slope, local_slope_max, step_point)

        # cross over
        if slope < local_slope_max*slope_ratio_threshold:
            # find local minimum
            dur = int(step_dur.mean()*search_window_ratio)
            if dur == 0: dur = 1
            i0 = 0 if (i-dur) < 0 else i-dur
            i1 = n_wave-1 if (i+dur) > (n_wave-1) else i+dur
            local_min_index = np.argmin(wave[i0:i1]) + i0
            #print('...', i0, i1, local_min_index)

            # shift index to new step point
            if local_min_index > step_point+search_window_ratio*dur:        # minimum distance between steps should be search_window_ratio * previous duration
                # update duration queue
                step_dur[:-1] = step_dur[1:]
                step_dur[-1] = local_min_index - step_point + 1
                dur_list.append(local_min_index-step_point+1)

                # update index and step_point
                step_point = local_min_index
                step_pos.append(step_point)
                ptp_list.append(local_slope_max)
                min_sig = wave[i]
                i = local_min_index + 1

                if slope != 0:
                    slope_list.append(local_slope_max/slope)
            else:
                i += 1

        # no cross over
        else:
            if wave[i] < min_sig:
                min_sig = wave[i]
            i += 1

    return step_pos, slope_list, ptp_list[1:], dur_list[1:]


def SC_method_localpeak(signal: Array[float], peak_window:int=50, mean_average_window:int=30, direction:str='down') -> (List, List, List):

    hard_peaks = processing.find_local_peaks(signal, peak_window)
    correct_peaks = processing.correct_peaks(signal, hard_peaks, peak_window, mean_average_window, peak_dir=direction)
    correct_peaks = np.unique(correct_peaks)

    correct_peaks = correct_peaks[(correct_peaks > -1) & (correct_peaks < len(signal))]
    if len(correct_peaks) > 1:
        durlist = [ correct_peaks[i+1]-correct_peaks[i] for i in range(len(correct_peaks)-1) ]
        ptplist = [ np.ptp(signal[correct_peaks[i]:correct_peaks[i+1]]) for i in range(len(correct_peaks)-1) ]
    else:
        ptplist = [0]
        durlist = [0]

    return correct_peaks, ptplist, durlist


def SC_method_QRS(signal: Array[float], conf, fs=100) -> (List, List, List):

    xqrs = processing.XQRS(sig=signal, fs=fs, conf=conf)
    xqrs.detect(verbose=False)
    correct_peaks = xqrs.qrs_inds

    if len(correct_peaks) > 1:
        durlist = [ correct_peaks[i+1]-correct_peaks[i] for i in range(len(correct_peaks)-1) ]
        ptplist = [ np.ptp(signal[correct_peaks[i]:correct_peaks[i+1]]) for i in range(len(correct_peaks)-1) ]
    else:
        ptplist = [0]
        durlist = [0]

    return correct_peaks, ptplist, durlist
