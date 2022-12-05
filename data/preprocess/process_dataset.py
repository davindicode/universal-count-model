%load_ext autoreload
%autoreload 2


import neural_datasets

from pprint import pprint


save_dir = '/scratches/ramanujan_2/dl543/HDC_PartIII/'
data_dir = '/scratches/sagarmatha_2/ktj21/data/crcns/'




mice_channels = {
    'Mouse28': {'PoS': [1, 2, 3, 4, 5, 6, 7], 
                'ANT': [8, 9, 10, 11]},
}

mice_sessions = {
    'Mouse28': ['140313']
}



# Mice sleep/wake exploration, head direction cells
phase = 'wake'

for mouse_id in mice_sessions.keys():
    for session_id in mice_sessions[mouse_id]:
        print(mouse_id, session_id)
        
        channels = mice_channels[mouse_id]
        data_class = neural_datasets.peyrache_th1(data_dir+'/th-1/data/', mouse_id, session_id, channels)
        
        periods = data_class.get_periods()
        time_limits = [periods['wake'][0]['start'], periods['wake'][0]['end']]  # pick wake session
        savef = save_dir + 'th1_{}_{}_{}.p'.format(mouse_id, session_id, phase)
        d = data_class.load_preprocess_save(None, time_limits)
        
        
        
pprint(periods, indent=2, sort_dicts=False)

time_limits = [periods['wake'][0]['start'], periods['wake'][0]['end']]  # pick wake session