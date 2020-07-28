'''
py_gait.py - gait pattern recognizer

author: Sung-Cheol Kim @ IBM

version:
    - 1.0.0 - 2020/04/13
    - 1.0.1 - 2020/04/27 - transfer functions from py_walk
    - 1.0.2 - 2020/05/19 - separate git repository and utility.py

'''

import pandas as pd
import numpy as np
import tqdm
import os
import h5py

from numba.typed import List
from typing import Any, Dict
from nptyping import NDArray

from wfdb import processing
from wfdb import rdsamp

import matplotlib.pyplot as plt
from matplotlib import patches
import seaborn as sns

from utility import calibration_gravity_parameters
from utility import check_walk_autocorrelation
from utility import normalized_auto_corr
from utility import find_long_period
from utility import SC_method_slope
from utility import SC_method_localpeak
from utility import SC_method_QRS

class GAIT:
    """ GAIT - object for storing signal and meta data as fields.
    """
    def __init__(self, pid, group='ltmm', channels=None, update=False, end_time=-1, debug=False) -> None:
        """ __init__

        Parameters
        ----------
        pid : string
            patient id for source file name
        channels : list
            channels in case of multi-channel source
        update : bool
            cached signal is saved in h5 format. update can be used for fresh reading from original source
        end_time : float
            last time point for reading source in minute
        debug : bool
            used for debugging, effective for all other methods
        """

        self._debug = debug

        # read signals
        if group == 'ltmm':
            self.signals, self.fields = self.load_ltmm_data(pid, channels=channels, update=update)
        else:
            self.signals, self.fields = self.load_pain_data(pid, channels=channels, update=update)

        # check end time (min)
        if end_time == -1:
            self.end_index = self.fields['sig_len']
        else:
            self.end_index = int(end_time * 60. * self.fields['fs'])
        self.signals = self.signals[:self.end_index, :]

        self.sum_fname = '.'+pid+'.csv'
        self.seconds = np.linspace(0, self.fields['sig_len']/self.fields['fs'], self.fields['sig_len'])
        print('... load data: {} - length: {:.2f} mins'.format(self.fields['tag'], max(self.seconds)/60))

        self._bout_index_list = []
        self._walk_index_list = []
        self.n_bouts = 0
        self.n_walks = 0

    def _detect_walk_std_th(self, idx=2, **kwargs) -> int:
        """ detect walk based on standard deviation and autocorrelation

        :param idx: channel index in multi-channel source
        :param kwargs: parameters for detection such as "Window" "Min_Duration" "STD_TH"
        """

        # set fields
        for k, i in kwargs.items():
            try:
                self._getfields(k, item=float(i))
            except:
                self._getfields(k, item=i)

        # calculate sd for window
        n = self.end_index
        window = int(self._getfields('Window')*self.fields['fs'])
        min_duration = int(self._getfields('Min_Duration')*self.fields['fs']/window)
        std_threshold = self._getfields('STD_TH')
        print('... [DW] Window: {} Min_Duration: {} STD_TH: {} Channel: {}'.format(window, min_duration, std_threshold, idx))

        # identify windows containing significant motion
        windowed_signal = np.reshape(self.signals[:n//window*window, idx], (n//window, window))
        windowed_signal_std = np.std(windowed_signal, axis=1)
        windowed_signal_bool = windowed_signal_std > std_threshold

        # find long continuous motion
        self._bout_index_list = find_long_period(windowed_signal_bool, min_duration, window)
        self.n_bouts = len(self._bout_index_list)

        if self._debug:
            i = 0
            for s, e in self._bout_index_list:
                print('... [{}] continuous block index: {}, {}/{} std: {}'.format(i, s, e, len(windowed_signal_bool)*window, np.mean(windowed_signal_std[s:e])))
                print(windowed_signal_bool)
                i += 1
        print('... [DW] detect {} bouts - mean STD in window:{:.3f} - duration: {} [sec]'.format(self.n_bouts, np.mean(windowed_signal_std),sum(windowed_signal_bool)*window/self.fields['fs']))

        # assume bout and walk is same
        self._walk_index_list = self._bout_index_list
        self.n_walks = self.n_bouts

        return self.n_bouts

    def _filter_walk_autocorr(self, idx=2, **kwargs) -> int:
        """_filter_walk_autocorr.

        :param idx: index for detection channel
        :param kwargs: parameters for autocorrelation detection such as "NASC_TH", "NASC_Tau_Min", "NASC_Tau_Max"
        :rtype: int number of detected walks
        """

        if self.n_bouts == 0:
            print('... no activities are detected. Run _detect_walk_std_th first')
            return

        # set fields
        for k, i in kwargs.items():
            try:
                self._getfields(k, item=float(i))
            except:
                self._getfields(k, item=i)

        # prepare parameters
        NASC_TH = float(self._getfields('NASC_TH'))
        NASC_Tau_Min = int(self._getfields('NASC_Tau_Min')*self.fields['fs'])
        NASC_Tau_Max = int(self._getfields('NASC_Tau_Max')*self.fields['fs'])
        NASC_Window = int(self._getfields('NASC_Window')*self.fields['fs'])
        NASC_Step = int(self._getfields('NASC_Step')*self.fields['fs'])

        min_duration = int(self._getfields('Min_Duration')*self.fields['fs'])

        print('... [FW] NA Window: {} NA Step: {} TH: {} Tau_Min: {} Tau_Max: {} Min Dur: {} Channel: {}'.format(NASC_Window, NASC_Step, NASC_TH, NASC_Tau_Min, NASC_Tau_Max, min_duration, idx))

        walk_index_list = []
        i = 0
        for s_ind, e_ind in self._bout_index_list:
            signal = self.signals[s_ind:e_ind, idx]
            filtered_window_result, corr_list = check_walk_autocorrelation(signal, detection_window_len=NASC_Window, detection_window_step=NASC_Step, min_auto_corr_score=NASC_TH, min_time_lag=NASC_Tau_Min, max_time_lag=NASC_Tau_Max)

            if self._debug:
                print('... bout [{}] max normalized autocorrelation: {} selected section number - {}/{}'.format(i, np.max(corr_list), sum(filtered_window_result), len(filtered_window_result)))

            # if walk is detected
            if sum(filtered_window_result) > 0:
                # find continuous section
                index_list = find_long_period(filtered_window_result, min_duration, 1)

                # create walk index list
                for s, e in index_list:
                    if self._debug:
                        print('... old index: {}, {} new index: {}, {}'.format(s_ind, e_ind, s_ind+s, s_ind+e))
                        print('... seperate into {} walks'.format(len(index_list)))

                    walk_index_list.append([s+s_ind, e+s_ind])
            i += 1

        self._walk_index_list = walk_index_list
        self.n_walks = len(walk_index_list)

        print('... [FW] filtered {} from {} bouts'.format(self.n_walks, self.n_bouts))

        return self.n_walks

    def _count_steps_slope(self, idx=2, **kwargs) -> int:
        """_count_steps_slope.

        :param idx: index for channel
        :param kwargs: key parameters for slope-based step counting such as 'SC_TH', 'SC_Rw', 'SC_Dur', 'Direction'
        :rtype: int
        """

        # set parameter fields
        for k, i in kwargs.items():
            try:
                self._getfields(k, item=float(i))
            except:
                self._getfields(k, item=i)

        # return if there is no detected walks
        if self.n_walks == 0:
            print('... warning! detect walk first. gait._detect_walk()')
            return

        self.step_point_list = []
        self.step_n_list = []
        self.step_ptp_list = []
        self.step_dur_list = []

        # prepare parameters
        SC_TH = float(self._getfields('SC_TH'))
        SC_Rw = float(self._getfields('SC_Rw'))
        SC_Dur = int(self._getfields('SC_Dur')*self.fields['fs'])
        direction = self._getfields('Direction')

        # run for each walks
        for i in tqdm.tqdm_notebook(range(self.n_walks)):
            signal = self._get_bout(self.signals[:, idx], pos=i, walk=True)
            cp, slist, ptplist, durlist = SC_method_slope(signal, slope_ratio_threshold=SC_TH, search_window_ratio=SC_Rw, initial_duration=SC_Dur, direction=direction)

            if len(cp) == 0:
                self.step_n_list.append(0)
            else:
                self.step_point_list.append(cp)
                self.step_n_list.append(len(cp)-1)
                self.step_ptp_list.append(ptplist)
                self.step_dur_list.append(durlist)

            if self._debug:
                print('... mean slope ratio: {}'.format(np.mean(slist)))
                print('... [{}] detect {} steps'.format(i, len(cp)-1))

        self.total_steps = np.array(self.step_n_list).sum()
        print('... [CS] detect total {} steps with {} bouts on channel {}'.format(self.total_steps, self.n_bouts, idx))

        return self.total_steps

    def _count_steps_localpeaks(self, idx=2, **kwargs) -> int:
        """_count_steps_localpeaks.

        :param idx: index for detecting channel
        :param kwargs: parameters for local peak step counting method such as 'WPD_MAw', 'WPD_Pw', 'Directin'
        :rtype: int
        """

        # set parameter fields
        for k, i in kwargs.items():
            try:
                self._getfields(k, item=float(i))
            except:
                self._getfields(k, item=i)

        # return if there is no detected walks
        if self.n_walks == 0:
            print('... detect walk first! gait._detect_walk()')
            return

        self.step_point_list = []
        self.step_n_list = []
        self.step_ptp_list = []
        self.step_dur_list = []

        # prepare parameters
        MAw = int(self._getfields('WPD_MAw')*self.fields['fs'])   # Mean Average Window
        Pw = int(self._getfields('WPD_Pw')*self.fields['fs'])     # Peak Window
        direction = self._getfields('Direction')

        # run for each walks
        for i in tqdm.tqdm_notebook(range(self.n_walks)):
            signal = self._get_bout(self.signals[:, idx], pos=i, walk=True)
            cp, ptplist, durlist = SC_method_localpeak(signal, peak_window=Pw, mean_average_window=MAw, direction=direction)
            if len(cp) == 0:
                self.step_n_list.append(0)
            else:
                self.step_n_list.append(len(cp)-1)
                self.step_point_list.append(cp)
                self.step_ptp_list.append(ptplist)
                self.step_dur_list.append(durlist)

            if self._debug: print('... [{}] detect {} steps'.format(i, len(cp)-1))

        self.total_steps = np.array(self.step_n_list).sum()
        print('... [CS] detect total {} steps with {} bouts'.format(self.total_steps, self.n_bouts))

        return self.total_steps

    def _count_steps_qrs(self, idx=2, **kwargs) -> int:
        """_count_steps_qrs.

        :param idx: index for detecting channel
        :param kwargs: parameters for QRS step counting such as 'QRS_SD'
        :rtype: int
        """

        # set parameter fields
        for k, i in kwargs.items():
            try:
                self._getfields(k, item=float(i))
            except:
                self._getfields(k, item=i)

        # return if there is no detected walks
        if self.n_walks == 0:
            print('... detect walk first! gait._detect_walk()')
            return

        self.step_point_list = []
        self.step_n_list = []
        self.step_ptp_list = []
        self.step_dur_list = []

        # prepare parameters
        QRS_HR = 60./self._getfields('QRS_SD')      # Initial Step Duration
        conf = processing.XQRS.Conf(hr_init=QRS_HR, hr_max=180, hr_min=60, qrs_width=0.5)

        # run for each walks
        for i in tqdm.tqdm_notebook(range(self.n_walks)):
            signal = self._get_bout(self.signals[:, idx], pos=i, walk=True)
            cp, ptplist, durlist = SC_method_QRS(signal, conf, fs=self.fields['fs'])

            if len(cp) == 0:
                self.step_n_list.append(0)
            else:
                self.step_n_list.append(len(cp)-1)
                self.step_point_list.append(cp)
                self.step_ptp_list.append(ptplist)
                self.step_dur_list.append(durlist)

            if self._debug: print('... [{}] detect {} steps'.format(i, len(cp)-1))

        self.total_steps = np.array(self.step_n_list).sum()
        print('... [CS] detect total {} steps with {} bouts'.format(self.total_steps, self.n_bouts))

        return self.total_steps

    def step_analysis(self, update=False) -> Any:
        """ step_analysis.

        :param update: when update is True, step analysis is redone. otherwise, results are from file
        :type update: bool
        :rtype: DataFrame with "walk_id", "step_id", "duration", "ap_acc_ptp", "v_acc_ptp", "ml_acc_ptp", "yaw_sum"
        """

        if len(self.step_point_list) == 0:
            print('... count steps first! gatit._count_steps()')
            return

        if os.path.exists(self.sum_fname) and not update:
            self.stepdb = pd.read_csv(self.sum_fname)
            return self.stepdb

        # stepdb has all features for each steps
        self.stepdb = pd.DataFrame(np.zeros((self.total_steps, 3)),
                columns=['walk_id', 'step_id', 'duration'])
                #columns=['walk_id', 'step_id', 'duration', 'ap_acc_ptp', 'v_acc_ptp', 'ml_acc_ptp'])
        total_idx = 0

        for i in tqdm.tqdm_notebook(range(self.n_walks)):
            for j in range(self.step_n_list[i]):
                i0 = self.step_point_list[i][j]
                i1 = self.step_point_list[i][j+1]+1

                # calculate features for one step
                self.stepdb.walk_id.iloc[total_idx] = i
                self.stepdb.step_id.iloc[total_idx] = j
                self.stepdb.duration.iloc[total_idx] = (i1 - i0)/self.fields['fs']
                #self.stepdb.v_acc_ptp.iloc[total_idx] = np.ptp(self.signals[i0:i1, 0])
                #self.stepdb.ml_acc_ptp.iloc[total_idx] = np.ptp(self.signals[i0:i1, 1])
                #self.stepdb.ap_acc_ptp.iloc[total_idx] = np.ptp(self.signals[i0:i1, 2])
                #self.stepdb.yaw_sum.iloc[total_idx] = np.sum(self.signals[i0:i1, 3])

                total_idx = total_idx + 1

        self.stepdb.to_csv(self.sum_fname, index=False)

        return self.stepdb

    def walk_analysis(self) -> Any:
        """ walk summary

        :rtype: DataFrame - "walk_id", "stepn", "walk_dur", "dur_mean", "dur_std"
        """
        # build step database
        self.stepdb = pd.DataFrame(np.zeros((self.total_steps,4)), columns=['walk_id', 'step_id', 'duration', 'ptp'])
        idx = 0
        for i in range(self.n_walks):
            n_steps = self.step_n_list[i]
            self.stepdb.step_id.iloc[idx:idx+n_steps] = range(n_steps)
            self.stepdb.duration.iloc[idx:idx+n_steps] = np.array(self.step_dur_list[i])/self.fields['fs']
            self.stepdb.ptp.iloc[idx:idx+n_steps] = self.step_ptp_list[i]
            self.stepdb.walk_id.iloc[idx:idx+n_steps] = i
            idx += n_steps

        # build walk database
        self.walkdb = pd.DataFrame(np.zeros((self.n_walks, 12)),
                columns=['walk_id', 'step_n', 'walk_dur', 'dur_mean', 'dur_std', 'dur_min', 'dur_25p', 'dur_50p', 'dur_75p', 'dur_max', 'ptp_mean', 'ptp_std'])

        walk_group = self.stepdb.groupby('walk_id')
        self.walkdb.walk_id = walk_group.walk_id.first()
        self.walkdb.step_n = walk_group.step_id.count()
        self.walkdb.walk_dur = walk_group.duration.sum()
        self.walkdb.dur_mean = walk_group.duration.mean()
        self.walkdb.dur_std = walk_group.duration.std()
        self.walkdb.dur_min = walk_group.duration.min()
        self.walkdb.dur_25p = walk_group.duration.quantile(q=.25)
        self.walkdb.dur_50p = walk_group.duration.median()
        self.walkdb.dur_75p = walk_group.duration.quantile(q=.75)
        self.walkdb.dur_max = walk_group.duration.max()
        self.walkdb.ptp_mean = walk_group.ptp.mean()
        self.walkdb.ptp_std = walk_group.ptp.std()

        return self.walkdb

    def view_walks(self, varname:str='duration', walk_range:list=None):
        """ bout_plot2D

        :param str:
        :type str: varname
        """
        
        """ box plot of varname over different bouts """

        if not varname in self.stepdb.columns:
            return

        if walk_range is None:
            walk_range = range(self.n_walks)

        tmp = self.stepdb.loc[self.stepdb['walk_id'].isin(walk_range)]

        xsize = 20 if self.n_bouts > 30 else 14
        fig = plt.figure(figsize=(xsize,5))
        ax = sns.violinplot(x='walk_id', y=varname, data=tmp)
        ax.set_xticklabels(ax.get_xticklabels(), rotation = 45)
        plt.show()

        fig.savefig(self._getfields('tag')+'_walks_boxplot.pdf', dpi=300)

        return (ax)

    def view_steps(self, pos:int=0, channels:list=[]) -> None:
        """ view signals in bout units """

        if len(channels) == 0:
            channels = range(self.signals.shape[1])

        fig, axs = plt.subplots(ncols=1, nrows=len(channels), figsize=(20,4*len(channels)), sharex='col')

        if len(channels) == 1:
            axs = [axs]

        for j, i in enumerate(channels):
            source = self._get_bout(self.signals[:,i], pos=pos, walk=True)
            if len(self.step_point_list) > 0:
                self._plot_single(source, i, axs[j], step_pos=self.step_point_list[pos])
            else:
                self._plot_single(source, i, axs[j])

        fig.tight_layout()
        plt.show()

        fig.savefig(self._getfields('tag')+'_walk'+str(pos)+'_channels.png', dpi=300)

    def view_signal(self, start_time:float=0, channel:int=2, nrows:int=5, width_time:float=60, save:bool=False):
        """ plot signals with long data """

        start_pos = int(start_time * self.fields['fs'] * 60)
        start_pos = self.end_index if start_pos > self.end_index else start_pos
        end_pos = start_pos + width_time*self.fields['fs']*nrows

        if len(self.signals[start_pos:, channel]) < width_time*self.fields['fs']*nrows:
            nrows = len(self.signals[start_pos:, channel])//(width_time*self.fields['fs'])
            if nrows == 0: nrows = 1
            print('... warning! signal is short: {}/{} - new rows: {}'.format(len(self.signals[start_pos:, channel]), len(self.signals[:,channel]), nrows))

        fig, axs = plt.subplots(ncols=1, nrows=nrows, figsize=(18, 2*nrows), sharey='col')
        if nrows == 1: axs = [axs]
        for i in range(nrows):
            i0 = start_pos + i*width_time*self.fields['fs']
            i1 = start_pos + (i+1)*width_time*self.fields['fs']
            axs[i].plot(self.seconds[i0:i1]/60, self.signals[i0:i1, channel], color='navy')
            axs[i].set_xlabel('Time [mins]')
            axs[i].set_ylabel(self.fields['sig_name'][channel])

            # annotate detected bouts
            if len(self._bout_index_list) > 0:
                b_index = 0
                for (b0, b1) in self._bout_index_list:
                    detect_flag = False
                    # check starting point of bout
                    if (b0 > i0) and (b0 < i1):
                        # check ending point
                        if (b1 < i1): width = (b1 - b0)/(self.fields['fs']*60)
                        else: width = (i1 - b0)/(self.fields['fs']*60)
                        start = self.seconds[int(b0)]/60
                        detect_flag = True
                    # check ending point of bout
                    if not detect_flag and (b1 > i0) and (b1 < i1):
                        # check starting point
                        if (b0 > i0): width = (b1 - b0)/(self.fields['fs']*60)
                        else: width = (b1 - i0)/(self.fields['fs']*60)
                        start = self.seconds[int(i0)]/60
                        detect_flag = True

                    if detect_flag:
                        miny = np.min(self.signals[int(b0):int(b1), channel])
                        maxy = np.max(self.signals[int(b0):int(b1), channel])
                        axs[i].add_patch(patches.Rectangle((start, miny), width, maxy-miny,
                            linewidth=2, edgecolor='red', facecolor='none', alpha=0.8))
                        axs[i].text(start+width, maxy, str(b_index), color='red', verticalalignment='bottom')
                    b_index += 1

            if len(self._walk_index_list) > 0:
                w_index = 0
                for (w0, w1) in self._walk_index_list:
                    detect_flag = False
                    # check starting point of bout
                    if (w0 > i0) and (w0 < i1):
                        # check ending point
                        if (w1 < i1): width = (w1 - w0)/(self.fields['fs']*60)
                        else: width = (i1 - w0)/(self.fields['fs']*60)
                        start = self.seconds[int(w0)]/60
                        detect_flag = True
                    # check ending point of bout
                    if not detect_flag and (w1 > i0) and (w1 < i1):
                        # check starting point
                        if (w0 > i0): width = (w1 - w0)/(self.fields['fs']*60)
                        else: width = (w1 - i0)/(self.fields['fs']*60)
                        start = self.seconds[int(i0)]/60
                        detect_flag = True

                    if detect_flag:
                        miny = np.min(self.signals[int(w0):int(w1), channel])
                        maxy = np.max(self.signals[int(w0):int(w1), channel])
                        axs[i].add_patch(patches.Rectangle((start, miny), width, maxy-miny,
                            linewidth=2, edgecolor='green', facecolor='none', alpha=0.8))
                        axs[i].text(start, maxy, str(w_index), color='green', verticalalignment='bottom')

                    w_index += 1

        fig.tight_layout()
        plt.show()

        if save:
            fname = self._getfields('tag')+'_walklong_from_'+str(start_time)+'_to_'+str(end_pos/(self.fields['fs']*60))+'_channel_'+str(channel)+'.png'
            fig.savefig(fname, dpi=300)
            print('... save to {}'.format(fname))

    def _get_bout(self, source:NDArray[float], pos:int = 0, walk:bool = False) -> NDArray[float]:
        """_get_bout. obtain specific signals from bout

        :param source: source signal
        :type source: List[float]
        :param pos: position in bout list
        :type pos: int
        :rtype: List[float] 1D numpy array of signal
        """

        if walk:
            if (pos > self.n_walks - 1):
                print('... out of index number: {}/{}'.format(pos, self.n_walks))
                return

            start_i = int(self._walk_index_list[pos][0])
            end_i = int(self._walk_index_list[pos][1])
        else:
            if (pos > self.n_bouts - 1):
                print('... out of index number: {}/{}'.format(pos, self.n_bouts))
                return

            start_i = int(self._bout_index_list[pos][0])
            end_i = int(self._bout_index_list[pos][1])

        return source[start_i:end_i]

    def _get_amplitude(self, channels:list=None, update:bool=False) -> int:
        """_get_amplitude.

        :param channels: list of amplitude components
        :type channels: list
        :rtype: book amplitude are stored in signals
        """

        # check amplitude is present
        if (not update) and ('Amplitude' in self.fields['sig_name']):
            print("... Amplitude is already present!")
            return True

        if channels is None:
            channels = [0, 1, 2]

        # calculate norm of 3D signals
        amp = [ self.signals[:, i]**2 for i in channels ]
        amp = np.sqrt(sum(amp))

        # add to original data
        if 'Amplitude' not in self.fields['sig_name']:
            signals = np.zeros((self.signals.shape[0], self.signals.shape[1]+1))
            signals[:, :-1] = self.signals
            signals[:, -1] = amp
            self.signals = signals
        else:
            self.signals[:, -1] = amp

        # update fields
        if not isinstance(self.fields['sig_name'], list):
            self.fields['sig_name'] = self.fields['sig_name'].tolist()
            self.fields['units'] = self.fields['units'].tolist()
        self.fields['sig_name'].append('Amplitude')
        self.fields['units'].append('m/s^2')
        self.fields['n_sig'] = self.signals.shape[1]

        print('... [PP] new column is added on {} with channels {}'.format(self.signals.shape[1]-1, channels))

        return self.signals.shape[1]-1

    def _plot_single(self, source, idx, axe, step_pos=None, color='navy'):
        """ plot single channel source with labels and step positions """

        if len(source) > 8000:
            source = source[:8000]
            step_pos = np.array(step_pos)
            step_pos = step_pos[step_pos < 8000]
        t = np.linspace(0, len(source)/self.fields['fs'], num=len(source))

        axe.plot(t, source, color=color)
        axe.set_xlabel('Time')
        axe.set_ylabel(self.fields['sig_name'][idx])

        if step_pos is not None:
            axe.plot(t[step_pos], source[step_pos], 'o', color='red', alpha=0.5, markersize=12)

        return axe

    def _getfields(self, key:str, item=None) -> Any:
        """_getfields. load parameters from field and save to field dictionary

        :param key: key name string
        :type key: str
        :param item: item value to be saved in fields
        :rtype: Any
        """

        if (key in self.fields) and (item is None):
            return self.fields[key]

        if (key in self.fields) and (item is not None):
            self.fields[key] = item
            return item

        if (key not in self.fields) and (item is not None):
            if self._debug: print('... add {} to {}'.format(item, key))
            self.fields[key] = item
            return item

        # set initial values
        if (key not in self.fields) and (item is None):
            if key == "window": item = 0.1     # bout detection std window [sec]
            if key == "bout_threshold": item = 0.04
            if key == "bout_dis": item = 0.5    # between bout time [sec]
            if key == "bout_dur": item = 30    # minimum duration of bout [sec]
            if key == "threshold": item = 0.5   # step detection percentage
            if key == "window_len": item = 15   # step detection smoothing window size [points]
            if key == "th_dur": item = 0.6      # step detection local minimum window percentage
            if key == "WPD_MAw": item = 0.3     # WPD step detection Moving Average Window [sec]
            if key == "WPD_PW": item = 0.2      # WPD step detection Peak Window [sec]

            if self._debug: print('... add {} to {}'.format(item, key))
            self.fields[key] = item
            return item

    def load_ltmm_data(self, tag:str, channels:Any=None, base_dir:str='physionet.org/files/ltmm/1.0.0', update:bool = False) -> (NDArray, Dict):
        """load_ltmm_data. read signal information from physionet database (locally stored)

        :param tag: name of patient used for find file
        :type tag: str
        :param channels: list of channels
        :type channels: Any
        :param base_dir: directory where physionet data is located
        :type base_dir: str
        :param update: decide to read from h5 file
        :type update: bool
        :rtype: (NDArray, Dict) signal numpy array and field dictionary
        """

        if tag.find('base') > -1:
            filename = base_dir + '/LabWalks/' + tag
        else:
            filename = base_dir + '/' + tag

        # check update and existance of h5 file
        self.h5_fname = filename + '.h5'
        if not update:
            if os.path.exists(self.h5_fname):
                return self.from_h5(self.h5_fname)

        # if update or no h5 file
        if os.path.exists(filename + '.hea'):
            print('... read from {}'.format(filename))
            signals, fields = rdsamp(filename, channels=channels)
        else:
            print('... file not found: {}'.format(filename + '.hea'))
            return

        # add new fields
        fields['tag'] = tag
        fields['filename'] = filename

        # clean up fields
        sig_name = fields['sig_name']
        sig_name = [ x.split(' ')[-1] for x in sig_name]
        fields['sig_name'] = sig_name

        # convert [g] to [m/s2]
        for i, u in enumerate(fields['units']):
            if u == 'g':
                signals[:, i] = 9.8*signals[:,i]
                fields['units'][i] = 'm/s^2'

        # write as h5 file
        self.to_h5(signals, fields, fname=self.h5_fname)

        return signals, fields

    def load_pain_data(self, tag:str, channels:Any=None, base_dir:str='.', update:bool=False) -> (NDArray, Dict):
        """load_pain_data.

        :param tag: tag name and directory name
        :type tag: str
        :param channels: for now this is not working
        :type channels: Any
        :param base_dir: '.'
        :type base_dir: str
        :param update: used for converting and loading from h5 file
        :type update: bool
        :rtype: (NDArray, Dict)
        """

        filename = base_dir + '/' + tag + '/EDA/ACC'

        # check update and existance of h5 file
        self.h5_fname = filename + '.h5'
        if not update:
            if os.path.exists(self.h5_fname):
                return self.from_h5(self.h5_fname)

        filename += '.csv'
        if os.path.exists(filename):
            signals = pd.read_csv(filename, header=1, names=['x', 'y', 'z'])
            signals = np.array(signals)*9.84/64.        # conversion to m/s2 - original unit 1/64 g
        else:
            print('... file not found! {}'.format(filename))
            return

        fields = {'tag': tag, 'filename':filename, 'sig_len':signals.shape[0],
                'n_sig': signals.shape[1], 'fs':32, 'base_date':None, 'base_time':None,
                'units': ['m/s^2', 'm/s^2', 'm/s^2'],
                'sig_name': ['x-acceleration', 'y-acceleration', 'z-acceleration']}

        self.to_h5(signals, fields, fname=self.h5_fname)

        return signals, fields

    def to_h5(self, signals: NDArray[float] = None, fields: Dict = None, fname:str=None) -> None:
        """to_h5. save signal and fields to h5 file

        :param signals: numpy array of signals
        :type signals: List[float]
        :param fields: meta information dictionary
        :type fields: Dict
        :param fname: file name
        :type fname: str
        :rtype: None
        """

        # check initial parameter
        if signals is None: signals = self.signals
        if fields is None: fields = self.fields
        if fname is None: fname = self.h5_fname

        # write signal as h5
        print('... write to {}'.format(fname))
        hf = h5py.File(fname, 'w')
        ds = hf.create_dataset('signals', data=signals, compression='lzf')

        # write fields as attributes
        for k, i in fields.items():
            if i is None: i = 'NA'
            ds.attrs[str(k)] = i
        hf.close()

    def from_h5(self, fname:str=None) -> (NDArray, Dict):
        """from_h5. load signal and meta info fields from h5 file

        :param fname: file name
        :type fname: str
        :rtype: (NDArray, Dict) -> signal numpy array and meta information dictionary
        """

        if fname is None: fname = self.h5_fname

        # read signal from h5
        print('... read from {}'.format(fname))
        hf = h5py.File(fname, 'r')
        ds = hf.get('signals')

        # load fields from atttributes
        fields = {}
        for k, i in ds.attrs.items():
            fields[k] = i
        signals = np.array(ds)
        hf.close()

        return signals, fields



# vim:foldmethod=indent:foldlevel=0
