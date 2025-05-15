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
            
    def particle_cache(self, context, pressure, chemical_compound, seeding_shapefile, pnum, start_time, duration_days, s_bounds, seeding_radius, beaching, z, tstep, hdiff, termvel, stokes_drift, seeding_id):
        cache_key = {'context': context, 'pressure': pressure, 'chemical_compound': chemical_compound, 'seeding_shapefile': seeding_shapefile, 'pnum': pnum, 'start_time': start_time, 'duration_days': duration_days, 's_bounds': s_bounds, 'seeding_radius': seeding_radius, 'beaching': beaching, 'z': z, 'tstep': tstep, 'hdiff': hdiff, 'termvel': termvel, 'stokes_drift': stokes_drift, 'seeding_id': seeding_id}
        path_data_file = self.get_data_file('nc', **cache_key) # chiave della cache e nome del file
        self.set_metadata('nc', **cache_key) #TODO spostare
        logger.info('particle cache = '+str(cache_key))
        return path_data_file, path_data_file.exists()
        

    def raster_cache(self, context, pressure, chemical_compound, seeding_shapefile, seeding_radius, res, pnum, ptot, duration, start_time, seedings, seeding_id, tshift, use_path, use_label, emission, decay_coef, r_bounds):
        cache_key = {'context': context, 'pressure': pressure, 'chemical_compound': chemical_compound, 'seeding_shapefile': seeding_shapefile, 'seeding_radius': seeding_radius, 'res': res, 'pnum': pnum, 'ptot': ptot, 'duration': duration, 'start_time': start_time, 'seedings': seedings, 'seeding_id': seeding_id, 'tshift': tshift, 'use_path': use_path, 'use_label': use_label, 'emission': emission, 'decay_coef': decay_coef, 'r_bounds': r_bounds}
        path_data_file = self.get_data_file('tif', **cache_key) # chiave della cache e nome del file
        #path_data_file = Path(str(_path_data_file).split('.tif')[0]+'_use-RES-TIME.tif')
        self.set_metadata('tif', **cache_key) #TODO spostare
        logger.info('raster cache = '+str(cache_key))
        return path_data_file, path_data_file.exists()