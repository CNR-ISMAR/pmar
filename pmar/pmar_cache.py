from pathlib import Path
import hashlib
import json
import logging
logger = logging.getLogger("PMARCache")

class PMARCache(object):
    def __init__(self, cachedir): # cachedir è una sottodirectory di basedir
        self.cachedir = Path(cachedir)
        self.cachedir.mkdir(exist_ok=True)
        
    def get_data_file(self, extension, **kwargs):
        _data_file = hashlib.md5(str(sorted(kwargs.items())).encode('utf-8')).hexdigest()
        data_file = f"{_data_file}.{str(extension)}" # chiave della cache e nome del file, generalizzata sia per get_trajectories che particle_raster
        path_data_file = Path(self.cachedir) / data_file 
        return path_data_file
    
    def set_metadata(self, extension, **kwargs):
        path_data_file = self.get_data_file(extension, **kwargs)
        path_metadata_file = str(path_data_file) + '_metadata' #TODO rendere più robusto
        with open(path_metadata_file,'w') as fi:
            json.dump(kwargs,fi,default=str)
            
    def particle_cache(self, poly_path, pnum, start_time, season, duration_days, s_bounds, seeding_radius, beaching, z, tstep, hdiff, termvel, crs, stokes_drift):
        cache_key = {'poly_path': poly_path, 'pnum': pnum, 'start_time': start_time.strftime("%Y-%m-%d"), 'season': season, 'duration_days': duration_days, 's_bounds': s_bounds, 'seeding_radius': seeding_radius, 'beaching': beaching, 'z': z, 'tstep': tstep, 'hdiff': hdiff, 'termvel': termvel, 'crs': crs, 'stokes_drift': stokes_drift}
        path_data_file = self.get_data_file('nc', **cache_key) # chiave della cache e nome del file
        self.set_metadata('nc', **cache_key) #TODO spostare
        logger.error('particle cache = '+str(cache_key))
        return path_data_file, path_data_file.exists()
        

    def raster_cache(self, res, poly_path, pnum, ptot, duration, start_time, reps, tshift, use_path, use_label, decay_coef, r_bounds):
        cache_key = {'res': res, 'poly_path': poly_path, 'pnum': pnum, 'ptot': ptot, 'duration': duration, 'start_time': start_time, 'reps': reps, 'tshift': tshift, 'use_path': use_path, 'use_label': use_label, 'decay_coef': decay_coef, 'r_bounds': r_bounds}
        path_data_file = self.get_data_file('tif', **cache_key) # chiave della cache e nome del file
        #path_data_file = Path(str(_path_data_file).split('.tif')[0]+'_use-RES-TIME.tif')
        self.set_metadata('tif', **cache_key) #TODO spostare
        logger.error('raster cache = '+str(cache_key))
        return path_data_file, path_data_file.exists()