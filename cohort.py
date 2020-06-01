'''
cohort.py - Utility object to process patient cohort

autho: Sung-Cheol Kim 

version:
    1.0.0 - 2020/05/19 - initial version
'''

from typing import Any, Dict
from gait import GAIT

import pandas as pd

import plotly.express as px


class COHORT:
    """ COHORT - object for processing multiple GAIT objects
    """
    def __init__(self, idlist:list, parameter_dict:dict=None, group:str='ltmm', update=False) -> None:
        """__init__.

        :param idlist:
        :type idlist: list
        :param parameter_dict:
        :type parameter_dict: dict
        :param group:
        :type group: str
        :param update:
        :rtype: None
        """

        self.id_list = idlist
        self.para_dict = parameter_dict
        self.group = group
        self.update = update
        self.current_gait = None
        self.current_walkdb = None
        self.walkdb = None
        self.check_clinical_db = False

    def _add_clinical_data(self, df, score_list:list=None) -> None:
        """_add_clinical_data.

        :param df: panda dataframe with clinical data column
        :param score_list: name of clinical tests
        :type score_list: list
        :rtype: None
        """

        if score_list is None:
            score_list = ['TUG']

        for i in score_list:
            if not i in df.columns:
                print("... score {} is not in database".format(i))
                return

        self.check_clinical_db = True
        self.clinical_db = df
        self.score_list = score_list

    def analysis_one(self, idx:int=0, detect_channel:int=-1, para_dict:dict=None, debug=False):
        """analysis_one.

        :param idx:
        :type idx: int
        :param detect_channel:
        :type detect_channel: int
        :param para_dict:
        :type para_dict: dict
        :param debug:
        """

        # load data
        try:
            g = GAIT(self.id_list[idx], group=self.group, update=self.update, debug=debug)
        except:
            print('... [{}] error on reading file'.format(idx))
            return False

        if para_dict is None:
            para_dict = self.para_dict

        # preprocess
        # calculate amplitude
        idx_amp = g._get_amplitude(update=self.update)
        if (detect_channel < 0) or (detect_channel >= g.fields['n_sig']):
            detect_channel = idx_amp

        # detect bouts - if it does not find bouts, algorithm is used with lower threshold
        old_std_th = self.para_dict['STD_TH']
        min_bout_n = self.para_dict.get('Min_Bout_n', 1)
        while True:
            g._detect_walk_std_th(idx=detect_channel, **para_dict)
            if (g.n_bouts >= min_bout_n) or (self.para_dict['STD_TH'] <= 0): break
            self.para_dict['STD_TH'] -= 0.02
        # if autocorrection algorithm does not find bouts
        if self.para_dict['STD_TH'] < 0:
            print('... [{}] error on detecting bouts'.format(idx))
            self.para_dict['STD_TH'] = old_std_th
            return False

        # detect walks
        old_nasc_th = self.para_dict['NASC_TH']
        while True:
            g._filter_walk_autocorr(idx=detect_channel, **para_dict)
            if (g.n_walks >= min_bout_n) or (self.para_dict['NASC_TH'] <= 0): break
            self.para_dict['NASC_TH'] -= 0.02
        if self.para_dict['NASC_TH'] <= 0:
            print('... [{}] error on detecting walks'.format(idx))
            self.para_dict['NASC_TH'] = old_nasc_th
            return False

        # count steps
        g._count_steps_slope(idx=detect_channel, **para_dict)

        # save as h5
        g.to_h5()
        self.para_dict['STD_TH'] = old_std_th
        self.para_dict['NASC_TH'] = old_nasc_th

        # feature selection
        patient_walk_db = g.walk_analysis()
        patient_walk_db['pid'] = self.id_list[idx]

        # check clinical data
        if self.check_clinical_db:
            for s in self.score_list:
                tmp = self.clinical_db[self.clinical_db.pid.str.contains(self.id_list[idx])]
                if len(tmp) == 0:
                    print('... score {} is not found! {}'.format(s, tmp))
                    patient_walk_db[s] = None
                else:
                    patient_walk_db[s] = float(tmp[s])

        # set gait as internal object
        self.current_gait = g
        self.current_walkdb = patient_walk_db

        return True

    def analysis_range(self, idx_range:list=None, detect_channel:int=-1):

        if idx_range is None:
            idx_range = range(len(self.id_list))

        self.walkdb = pd.DataFrame()
        for i in idx_range:
            if self.analysis_one(idx=i, detect_channel=detect_channel):
                self.walkdb = pd.concat([self.walkdb, self.current_walkdb], ignore_index=True)

        return self.walkdb

    def walk_plot3D(self) -> Any:
        """ plot each bout in 3D axes

        :rtype: matplotlib figure object
        """

        mean_db = self.stepdb.groupby(['bout_id']).mean()

        fig = px.scatter_3d(mean_db, x='duration', y='ap_acc_ptp', z='v_acc_ptp',
                    color='ml_acc_ptp', size_max=18,
                    opacity=0.7)

        # tight layout
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))

        return fig



# vim:foldmethod=indent:foldlevel=0
