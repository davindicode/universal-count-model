import numpy as np

import scipy.io  # needed for older versions of MATLAB files
from scipy.interpolate import interp1d, Akima1DInterpolator, CubicSpline
import h5py  # MATLAB files > v7.3

import pickle

import glob



### base class ###
class _dataset():
    """
    utility functions for data cleaning
    """
    def __init__(self, interp_type):
        self.interp_type = interp_type
    
    def interpolator(self, t, x, kind=None):
        if kind is None:
            kind = self.interp_type
            
        if kind == 'natural':
            return CubicSpline(t, x, bc_type='natural')
        elif kind == 'akima':
            return Akima1DInterpolator(t, x)
        elif kind == 'linear':
            return lambda t_: np.interp(t_, t, x)
        elif kind == 'nearest':
            return interp1d(t, x, kind='nearest')
        else:
            raise ValueError('Invalid interpolation spline type')
    
    @staticmethod
    def consecutive_arrays(arr, step=1):
        """
        Finds consecutive subarrays satisfying monotonic stepping with step in array.
        Returns on each array element the island index (starting from 1).

        :param list colors: colors to be included in the colormap
        :param string name: name the colormap
        :returns: figure and axis
        :rtype: tuple
        """
        islands = 1 # next island count
        island_ind = np.zeros(arr.shape)
        on_isl = False
        for k in range(1, arr.shape[0]):
            if arr[k] == arr[k-1] + step:
                if on_isl is False:
                    island_ind[k-1] = islands
                    on_isl = True
                island_ind[k] = islands
            elif on_isl is True:
                islands += 1
                on_isl = False

        return island_ind

    @staticmethod
    def true_subarrays(arr):
        """
        Finds consecutive subarrays in booleans array.

        :param list colors: colors to be included in the colormap
        :param string name: name the colormap
        :returns: figure and axis
        :rtype: tuple
        """
        on_isl = False
        island_start_ind = []
        island_size = []
        cnt = 0
        for k in range(len(arr)):
            if arr[k]:
                if on_isl is False:
                    island_start_ind.append(k)
                    island_size.append(1)
                    on_isl = True
                else:
                    island_size[cnt] += 1
            elif on_isl is True:
                cnt += 1
                on_isl = False

        return island_start_ind, island_size
    
    @staticmethod
    def stitch_nans(series, invalids, angular):
        """
        Interpolate between points with NaNs islands, unless they are the ends
        in place operation
        """
        for ind, size in zip(*invalids):
            dinds = np.arange(size)

            if ind == 0: # copy
                series[dinds+ind] = series[ind+size]
                continue
            elif ind+size == len(series):
                series[dinds+ind] = series[ind-1]
                continue

            if angular:  # ensure in [0, 2*pi)
                series = series % (2*np.pi)

            dseries = series[ind+size] - series[ind-1]

            if angular: # interpolate with geodesic distances
                if dseries > np.pi:
                    dseries -= 2*np.pi
                elif dseries < -np.pi:
                    dseries += 2*np.pi

            series[dinds+ind] = (series[ind-1] + dseries * (dinds+1)/(size+1))




class peyrache_th1(_dataset):
    """
    [th1] head direction cell detection in anterior nuclei of thalamus and postsubiculum in mice

    References:

    [1] Peyrache, A., Petersen P., Buzsáki, G. (2015)
        `Extracellular recordings from multi-site silicon probes in the anterior thalamus and subicular formation of freely moving mice', 
        CRCNS.org.
        http://dx.doi.org/10.6080/K0G15XS1
    """
    
    def __init__(self, datadir, mouse_id, session_id, electrode_groups, interp_type='akima'):
        """
        :param dict electrode_groups: dictionary indicating channels belonging to neuron type
        """
        super().__init__(interp_type)
        self.datadir = datadir
        self.mouse_id = mouse_id
        self.session_id = session_id
        self.electrode_groups = electrode_groups
        
    def get_periods(self):
        """
        Get periods of wake and sleep labelled by the experimenter
        """
        datadir = self.datadir
        ext_name = self.mouse_id+"-"+self.session_id
        
        f_wake = open(datadir+ext_name+"/"+ext_name+".states.Wake", "r").read()
        l = f_wake.split(sep='\n')[:-1]
        wake = [{'start': float(i.split('\t')[0]), 'end': float(i.split('\t')[1])} for i in l]

        f_REM = open(datadir+ext_name+"/"+ext_name+".states.REM", "r").read()
        l = f_REM.split(sep='\n')[:-1]
        REM = [{'start': float(i.split('\t')[0]), 'end': float(i.split('\t')[1])} for i in l]

        f_SWS = open(datadir+ext_name+"/"+ext_name+".states.SWS", "r").read()
        l = f_SWS.split(sep='\n')[:-1]
        SWS = [{'start': float(i.split('\t')[0]), 'end': float(i.split('\t')[1])} for i in l]
        
        return {'wake': wake, 'REM': REM, 'SWS': SWS}


    def load_preprocess_save(self, savefile, time_limits):
        """
        time of the first video frame was randomly misaligned by 0–60 ms 

        The behaviour is recorded at a different frequency, resample to get it at the same frequency as 
        spike recordings. Note folder "PosFiles" and "AngFiles" are the same data

        :param bool interpolate_invalid: indicates whether to remove invalid data segments or interpolate
        """
        datadir, mouse_id, session_id = self.datadir, self.mouse_id, self.session_id
        electrode_groups = self.electrode_groups
        ext_name = mouse_id+"-"+session_id
        
        # -1 values indicate that LED detection failed in ang, NaN values present in XY data
        f_ang = open(datadir+"PositionFiles/"+mouse_id+"/"+ext_name+"/"+ext_name+".ang", "r").read()
        f_pos = open(datadir+"PositionFiles/"+mouse_id+"/"+ext_name+"/"+ext_name+".pos", "r").read()
        
        l = f_pos.split(sep='\n')[:-1]
        pos = np.array([i.split('\t') for i in l]).astype(np.float)
        
        l = f_ang.split(sep='\n')[:-1]
        ang = np.array([i.split('\t') for i in l]).astype(np.float)
        
        
        ### synchronization and period used ###
        
        # get electrode and behaviour time resolutions
        sample_bin = 1./20000
        behav_times = ang[:, 0]  # in seconds
        behav_tbin = np.diff(behav_times).mean()
        print("Behaviour time bin size {:.2e} s.".format(behav_tbin))
        
        request_left_T, request_right_T = time_limits
        window = (behav_times <= request_right_T) & (behav_times >= request_left_T)
        use_times = behav_times[window]  # in seconds
        left_T, right_T = use_times[0], use_times[-1]
        left_T_sb = int(np.ceil(left_T/sample_bin))
        right_T_sb = int(np.floor(right_T/sample_bin))
        synch_times = np.arange(left_T_sb, right_T_sb+1, 1) * sample_bin
        if synch_times[-1] >= use_times[-1]:  # remove right edge for interpolator range
            synch_times = synch_times[:-1]
        use_sample_num = len(synch_times)  # maximum number of sample bins for spike train lengths
        
        
        ### spikes ###
        
        # extract spike times and clustering
        spiketimes = []
        spikeclusters = []
        totclusters = []
        for key in electrode_groups:
            for e in electrode_groups[key]:
                f_res = open(datadir+ext_name+"/"+ext_name+".res.{}".format(e), "r").read()
                f_clu = open(datadir+ext_name+"/"+ext_name+".clu.{}".format(e), "r").read()

                # spike times are indices of bins at resolution of sample_bin
                spiketimes.append(np.array(f_res.split(sep='\n')[:-1]).astype(np.float))
                arr = np.array(f_clu.split(sep='\n')[:-1]).astype(np.float)

                spikeclusters.append(arr[1:])
                totclusters.append(arr[0])

        totclusters = np.array(totclusters).astype(int)
        units = totclusters.sum()

        # separate spikes into different neurons from spike sorting clusters
        sep_t_spike = []
        neuron_groups = {}
        k = 0  # keep track of electrode/channel groups
        u = 0  # keep track of units
        for key, value in electrode_groups.items():
            neurons = []
            for _ in range(len(value)):
                for l in range(totclusters[k]):  # assume clusters are separate neurons for each file
                    arr = np.sort(spiketimes[k][spikeclusters[k] == l]).astype(int)
                    if (len(arr) == 0): # silent neurons
                        units -= 1
                        print('Empty channel shank {} cluster {}.'.format(k, l))
                        continue

                    neurons += [u]
                    u += 1
                    sep_t_spike.append(arr)

                k += 1
            neuron_groups[key] = neurons

        # extract spikes for the period used
        use_t_spike = []  # bin spike times into resolution of sample_bin
        for u in range(units):
            times = sep_t_spike[u]
            use_t_spike.append(
                times[(times <= right_T_sb) & (times >= left_T_sb)] - left_T_sb
            )  # set starting left edge to 0
            
        # ISI statistics
        ISI = []
        for u in range(units):
            ISI.append((use_t_spike[u][1:] - use_t_spike[u][:-1]) * sample_bin*1000) # ms

        refract_viol = np.empty((units))
        viol_ISI = 2.0 # ms
        for u in range(units):
            refract_viol[u] = (ISI[u] <= viol_ISI).sum()/len(ISI[u])
            
        neural = {
            'units': units, 
            'spike_time_inds': use_t_spike, 
            'refract_viol': refract_viol,
            'electrode_groups': electrode_groups,
            'neuron_groups': neuron_groups,
        }
            
            
        ### behaviour ###
        x_beh, y_beh = pos[window, 1], pos[window, 2]
        hd_beh = ang[window, 1]

        # interpolator for invalid points
        hd_nan = (hd_beh == -1.)
        invalids = self.true_subarrays(hd_nan)
        self.stitch_nans(hd_beh, invalids, angular=True)
        
        xy_nan = (x_beh != x_beh)
        invalids = self.true_subarrays(xy_nan)
        self.stitch_nans(x_beh, invalids, angular=False)
        self.stitch_nans(y_beh, invalids, angular=False)

        # resample with spikes
        x_t = self.interpolator(use_times, x_beh)(synch_times)
        y_t = self.interpolator(use_times, y_beh)(synch_times)
        hd_t = self.interpolator(use_times, np.unwrap(hd_beh))(synch_times)

        hd_invalids = self.interpolator(use_times, hd_nan.astype(float), kind='nearest')(synch_times)
        xy_invalids = self.interpolator(use_times, xy_nan.astype(float), kind='nearest')(synch_times)
        
        # label segments with originally invalid behaviour
        invalid_behaviour = {
            'HD': [{'index': duo[0], 'length': duo[1]} for duo in zip(*self.true_subarrays(hd_invalids.astype(bool)))], 
            'XY': [{'index': duo[0], 'length': duo[1]} for duo in zip(*self.true_subarrays(xy_invalids.astype(bool)))], 
        }
        
        covariates = {
            'x': x_t, 
            'y': y_t, 
            'hd': hd_t, 
            'invalid_behaviour': invalid_behaviour,
        }
        

        ### export ###
        data = {
            'sample_bin': sample_bin, 
            'use_sample_num': use_sample_num, 
            'neural': neural, 
            'covariates': covariates,
        }
        
        if savefile is None:
            return data
        else:
            pickle.dump(data, open(savefile, 'wb'), pickle.HIGHEST_PROTOCOL)