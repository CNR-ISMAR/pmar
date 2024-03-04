import logging
import rioxarray as rxr
import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
from datetime import datetime, timedelta, date
from shapely.geometry import Point, Polygon
from opendrift.models.oceandrift import OceanDrift
from opendrift.models.openoil import OpenOil
#from opendrift.readers import reader_netCDF_CF_generic
import copernicusmarine
from opendrift.readers.reader_netCDF_CF_generic import Reader
import opendrift
from pydap.client import open_url
from pydap.cas.get_cookies import setup_session
import time as T
from pathlib import Path
from xhistogram.xarray import histogram
import seaborn as sns
spectral_r = sns.color_palette("Spectral_r", as_cmap=True)
import cartopy.crs as ccrs
import cartopy
from matplotlib.ticker import LogFormatter, PercentFormatter
from matplotlib import ticker
import matplotlib.pyplot as plt
import matplotlib.colors as col
import os
from opendrift.readers import reader_shape
import cartopy.io.shapereader as shpreader
import netrc
import random
import rasterio
from rasterio.enums import Resampling
import tempfile
from cachetools import LRUCache
from netCDF4 import Dataset
import hashlib
import json

logger = logging.getLogger("LagrangianDispersion")

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

class PMARCache(object):
    def __init__(self, cachedir): # cachedir è una sottodirectory di basedir
        self.cachedir = Path(cachedir)
        self.cachedir.mkdir(exist_ok=True)
        
    def get_data_file(self, extension, **kwargs):
        _data_file = hashlib.md5(str(sorted(kwargs.items())).encode('utf-8')).hexdigest()
        data_file = f"{_data_file}.{str(extension)}" # chiave della cache e nome del file, generalizzata sia per particle_simulation che particle_raster
        path_data_file = Path(self.cachedir) / data_file 
        return path_data_file
    
    def set_metadata(self, extension, **kwargs):
        path_data_file = self.get_data_file(extension, **kwargs)
        path_metadata_file = str(path_data_file) + '_metadata' #TODO rendere più robusto
        with open(path_metadata_file,'w') as fi:
            json.dump(kwargs,fi,default=str)
            
    def particle_cache(self, poly_path, pnum, start_time, season, duration_days, s_bounds, z, tstep, hdiff, termvel, crs):
        cache_key = {'poly_path': poly_path, 'pnum': pnum, 'start_time': start_time, 'season': season, 'duration_days': duration_days, 's_bounds': s_bounds, 'z': z, 'tstep': tstep, 'hdiff': hdiff, 'termvel': termvel, 'crs': crs}
        path_data_file = self.get_data_file('nc', **cache_key) # chiave della cache e nome del file
        self.set_metadata('nc', **cache_key) #TODO spostare
        return path_data_file, path_data_file.exists()
        
    def raster_cache(self, poly_path, pnum, ptot, duration_days, s_bounds, z, tstep, hdiff, termvel, crs, tinterp, r_bounds, use_path, decay_rate, aggregate, depth_layer, z_bounds, particle_status):
        cache_key = {'poly_path': poly_path, 'pnum': pnum, 'ptot': ptot, 'duration_days': duration_days, 's_bounds': s_bounds, 'z': z, 'tstep': tstep, 'hdiff': hdiff, 'termvel': termvel, 'crs': crs, 'tinterp': tinterp, 'r_bounds': r_bounds, 'use_path': use_path, 'decay_rate': decay_rate, 'aggregate':aggregate, 'depth_layer': depth_layer, 'z_bounds': z_bounds, 'particle_status': particle_status}
        path_data_file = self.get_data_file('tif', **cache_key) # chiave della cache e nome del file
        self.set_metadata('tif', **cache_key) #TODO spostare
        return path_data_file, path_data_file.exists()
    
        
class LagrangianDispersion(object): 
    """
    Developed at CNR ISMAR in Venice.

    ...

    Attributes
    ----------
    depth : bool
        boolean stating whether particle simulation is 2D (depth==False) or 3D (depth==True)
    o : OceanDrift
        OpenDrift object producing particle simulation
    poly_path : str
        path to shapefile containing polygon used for particle seeding
    bathy_path : str
        path to netcdf file for bathymetry (GEBCO)
    particle_path : str
        path to netcdf file containing output of OpenDrift simulation
    raster : xarray
        xarray object containing output of particle_raster() method

    Methods
    -------
    run()
        runs OpenDrift simulation and produces desired raster
    make_poly()
        if no polygon is given for seeding, creates and writes polygon based on given lon lat
    particle_simulation()
        runs OpenDrift simulation
    particle_raster()
        computes 2D histogram of particle concentration and writes to tif (if requested)
    plot()
        plots output of particle_raster()
    scatter()
        plots trajectories over time 
    animate()
        WIP

    """


    def __init__(self, context, pressure='general', basedir='lpt_output', localdatadir = None, poly_path = None, uv_path='cmems', wind_path='cmems', mld_path='cmems', bathy_path=None, particle_path=None, depth=False, netrc_path=None):
        """
        Parameters
        ---------- 
        context : str
            String defining the context of the simulation i.e., the ocean model output to be used for particle forcing. Options are 'med-cmems', 'bs-cmems' and 'bridge-bs'. 
        pressure : str, optional
            
        basedir : str, optional
            path to the base directory where all output will be stored. Default is to create a directory called 'lpt output' in the current directory.
        poly_path : str, optional
            path to shapefile containing polygon to be used for seeding of particles
        uv_path : str, optional
            path to the netcdf file containing ocean current data. Default is 'cmems', meaning CMEMS data will be streamed from Copernicus.
        wind_path : str, optional
            path to the netcdf file containing wind velocity data. Default is 'cmems', meaning CMEMS data will be streamed from Copernicus.        
        mld_path : str, optional
            path to the netcdf file containing mixed layer depth data. Default is 'cmems', meaning CMEMS data will be streamed from Copernicus.  
        bathy_path : str
            Path to bathymetry file to be used in 3D simulations. Default is None. An error is raised if depth is True and bathy_path is None. 
        particle_path : str, optional
            path to netcdf file containing output of OpenDrift simulation. Default is None. If a particle_path is given in initialisation,    
        depth : bool, optional
            boolean stating whether particle simulation is 2D (depth==False) or 3D (depth==True). Default is False
        """

        Path(basedir).mkdir(parents=True, exist_ok=True)
        self.uv_path = None
        self.wind_path = None
        self.mld_path = None 
        self.bathy_path = bathy_path 
        self.basedir = Path(basedir)
        self.particle_path = particle_path # i can import an existing particle_path
        self.ds = None
        self.o = None
        self.poly_path = poly_path 
        self.raster = None
        self.origin_marker = 0
        self.netrc_path = netrc_path
        self.tstep = None
        self.tseed = None
        self.pnum = None
        self.depth = depth
        self.termvel = 1e-3
        self.decay_rate = 0
        self.context = context
        self.outputdir = None
        self.pressure = pressure
        self.localdatadir = localdatadir
        self.particle_status = None
        self.reps = None
        self.tshift = None
        self.cache = PMARCache(Path(basedir) / 'cachedir')
        self.raster_path = None
        
        pres_list = ['general', 'microplastic', 'bacteria']
        pressures = pd.DataFrame(columns=['pressure', 'termvel', 'decay_rate'], 
                    data = {'pressure': pres_list, 
                            'termvel': [0, 1e-3, 0], 
                            'decay_rate': [0, 0, 1]})
        
        
        if pressure in pres_list:
            self.termvel = pressures[pressures['pressure'] == f'{pressure}']['termvel'].values[0]
            self.decay_rate = pressures[pressures['pressure'] == f'{pressure}']['decay_rate'].values[0]
        
        elif pressure == 'oil':
            pass
        
        else:
            pass
        
        # if a path to a shapefile is given to be used for seeding, read it and save it in the lpt_output/polygons dir
        if poly_path is not None: 
            Path(self.basedir / 'polygons').mkdir(parents=True, exist_ok=True)            
            poly = gpd.read_file(poly_path).to_crs('epsg:4326')
            bds = np.round(poly.total_bounds, 4) 
            local_poly = f'polygon-crs_epsg:{poly.crs}-lon_{bds[0]}_{bds[2]}-lat—{bds[1]}_{bds[3]}.shp'
            q = self.basedir / 'polygons' / local_poly
            poly.to_file(str(q), driver='ESRI Shapefile')
            self.poly_path = str(q)
        else:
            if 'med' in self.context:
                self.poly_path = f'{DATA_DIR}/polygon-med-full-basin.shp'
            elif 'bs' in self.context:
                self.poly_path = f'{DATA_DIR}/polygon-bs-full-basin.shp'
            else:
                pass
            
            
        # if particle_path is given, retrieve attributes from filename and load ds
        if self.particle_path is not None: 
            
            avail_contexts = ['med', 'bs', 'bridge-bs', 'med-cmems', 'bs-cmems']
            
            for c in avail_contexts:
                if c in self.particle_path:
                    self.context = c
            
            ## extract info from filename (BAD)
            if type(self.particle_path) == str:
                ppath = self.particle_path 
            else:
                ppath = self.particle_path[0]
                
            self.depth = ppath.split('depth_')[1][0] == 'T'  # THIS IS BAD. does not work if particle_path is a list of files
            
            if ppath.find('tseed') != -1:
                self.tseed = timedelta(seconds=float(ppath.split('tseed_')[1].split('s')[0]))
            
            self.tstep = timedelta(seconds=float(ppath.split('tstep_')[1].split('s')[0]))
            
            self.pnum = int(ppath.split('pnum_')[1].split('-')[0])
            
            self.get_ds
            
            # if a particle_path is given, meaning a run with those configs already exists, the poly_path contained in the file's attributes "wins" over poly_path
            if self.ds.poly_path is not None:
                self.poly_path = str(self.ds.poly_path)
                
        pass

    def get_userinfo(self, machine):
        try:
            secrets = netrc.netrc(self.netrc_path)
        except FileNotFoundError:
            ''
        auth = secrets.authenticators(machine)
        if auth is None:
            return ''
        return f'{auth[0]}:{auth[2]}@'
    
    #### testing cachetools
    #def _file_exists(self):
     #   try:
      #      with Dataset(self.particle_path, 'r'):
       #         return True
        #except FileNotFoundError:
         #   return False   
    
    
    ###### TODO: the filename used for the raster file is misleading with this method, if reps>1. ####### 
    #def get_raster_paths(self, decay_rate, reps=None, tshift=None, use_label='even'): # 
     #   if self.particle_path is None:
      #      raise ValueError('particle file not found')
        
       # if type(self.particle_path) is str:
        #    filename = self.particle_path.split('/')[-1].split('-marker')[0]
            
        #else:
         #   filename = self.particle_path[0].split('/')[-1].split('-marker')[0]
        
        #raster_paths = f'{filename}-decay_rate_{decay_rate}-use_{use_label}.tif'
        
        #if reps is not None:
         #   raster_paths = f'{filename}-decay_rate_{decay_rate}-use_{use_label}-totreps_{reps}-tshift_{tshift}.tif'
        
        #return raster_paths
    
    @property
    def get_ds(self):
        ds = xr.open_mfdataset(self.particle_path, concat_dim='trajectory', combine='nested', join='override', parallel=True, chunks={'trajectory': 10000, 'time':1000})
        
        # if the run contained reps, ensure trajectories have unique IDs for convenience
        if self.reps is not None:
            ds['trajectory'] = np.arange(0, len(ds.trajectory)) 
        
        self.ds = ds
        return ds

    def repeat_run(self, reps, tshift, ptot, start_time='2019-01-01', season=None, duration_days=30, s_bounds=None, z=-0.5, tstep=timedelta(hours=4), hdiff=10, termvel=None, raster=True, res=4000, crs='4326', tinterp=None, r_bounds=None, use_path='even', use_label='even', decay_rate=None, aggregate='mean', depth_layer='full_depth', z_bounds=[10,100], loglevel=40, save_to=None, plot=True, particle_status='all'):
        """
        Optimization method to produce one single raster from multiple runs. 
        The runs all have the same configuration but are shifted in time by a time period (tshift). 
        """
        self.reps = reps
        self.tshift = tshift
        self.ptot = ptot
        particle_dict = {}
        
                # commented as now everything goes into cachedir
        ##### THIS COULD PROBS GO IN INIT. for now copied from .run()
        #if save_to is None:
         #   Path(self.basedir / 'rasters').mkdir(parents=True, exist_ok=True)
          #  outputdir = self.basedir / 'rasters'
        #else:
         #   Path(save_to).mkdir(parents=True, exist_ok=True)
          #  outputdir = Path(save_to)
        #self.outputdir = outputdir
        ###############################
        
        
        for n in range(0, reps):
            print(f'----- STARTING REP #{n+1} -----')
            if n==0:
                t0 = datetime.strptime(start_time, '%Y-%m-%d')
            else:
                t0 = datetime.strptime(start_time, '%Y-%m-%d')+timedelta(days=tshift)

            start_time = t0.strftime("%Y-%m-%d")
            
            self.origin_marker = n
            
            self.particle_simulation(pnum=int(np.round(self.ptot/self.reps)), start_time=start_time, season=season, duration_days=duration_days, crs=crs, s_bounds=s_bounds, z=z, tstep=tstep, hdiff=hdiff, termvel=termvel, loglevel=loglevel)    
            particle_dict[n] = self.particle_path
            print(self.o)
            print(f'----- DONE WITH REP #{n+1} -----')
            
        self.particle_path = list(particle_dict.values())
        self.origin_marker = np.arange(0,reps)
        
        #raster_exists = Path(self.get_raster_paths(decay_rate, reps, tshift, use_label)).exists()
        #if raster_exists:
         #   print('raster file with these configurations already exists')
        
        raster_path, raster_exists = self.cache.raster_cache(poly_path=self.poly_path, pnum=None, ptot=ptot, duration_days=duration_days, s_bounds=s_bounds, z=z, tstep=tstep, hdiff=hdiff, termvel=termvel, crs=crs, tinterp=tinterp, r_bounds=r_bounds, use_path=use_path, decay_rate=decay_rate, aggregate=aggregate, depth_layer=depth_layer, z_bounds=z_bounds, particle_status=particle_status)
        
        self.raster_path = raster_path
        
        print('STARTING RASTER COMPUTATION')
        # compute raster
        if raster == True and not raster_exists:
            if type(use_path) == str:
                r = self.particle_raster(res=res, crs=crs, tinterp=tinterp, r_bounds=r_bounds, use_path=use_path, decay_rate=decay_rate,  aggregate=aggregate, depth_layer=depth_layer, z_bounds=z_bounds, particle_status=particle_status)
                #self.raster = r
            elif type(use_path) == list:
                for idx, use in enumerate(use_path):
                    r = self.particle_raster(res=res, crs=crs, tinterp=tinterp, r_bounds=r_bounds, use_path=use, decay_rate=decay_rate,  aggregate=aggregate, depth_layer=depth_layer, z_bounds=z_bounds, particle_status=particle_status, save_r=False)
                    if idx == 0:
                        self.raster = r
                    else:
                        self.raster = xr.merge([self.raster, r.rename({'r0': f'r{idx}', 'lon_bin': f'lon_bin_{idx}', 'lat_bin': f'lat_bin_{idx}'})], compat='broadcast_equals', join='outer', fill_value=0, combine_attrs='no_conflicts')
            
            else:
                raise ValueError('"use_path" not recognised. "use_path" must be either a string or list of strings containing path(s) to use layers.')
        
            # save final raster(s) and delete temporary files

            for i, r in enumerate(self.raster.data_vars): # save all available rasters in output directory
                    print(f'saving tiff file #{i+1}...')
                    self.raster[f'{r}'].rio.to_raster(str(raster_path)) ###### WARNING: NEL CASO DI PIù USI DI INPUT VIENE SOVRASCRITTO, CORREGGI
                    # save corresponding thumbnails in save output directory
                    print(f'saving thumbnail #{i+1}...')
                    self.plot(r=self.raster[f'{r}'], save_fig=f'{raster_path}_{i}_thumbnail.png')

        elif raster == True and raster_exists: 
            self.raster = xr.Dataset()
            self.raster['r0'] = rxr.open_rasterio(raster_path).isel(band=0).rename({'x': 'lon_bin', 'y': 'lat_bin'}).assign_attrs({'use_path': use_path})
        

        return self
        
        
        
    def run(self, pnum=100, start_time='2019-01-01', season=None, duration_days=30, s_bounds=None, z=-0.5, tstep=timedelta(hours=4), hdiff=10, termvel=None, raster=True, res=4000, crs='4326', tinterp=None, r_bounds=None, use_path='even', use_label='even', decay_rate=None, aggregate='mean', depth_layer='surface', z_bounds=[10,100], loglevel=40, save_to=None, plot=True, particle_status='all'):         
        """
        Launches methods particle_simulation and particle_raster. 
        
        
        Parameters 
        ----------
        reps = int, optional
            Number of desired iterations for the particle simulation. Default is 1
        pnum : int, optional 
            The number of particles to be seeded. Default is 100 (to be used for a test run)
        start_time : str, optional
            The start time of the simulation. Default is '2019-01-01' 
        season : str, optional
            Season in which simulation should be run. Defines start_time automatically. Default is None
        duration_days : int, optional
            Integer defining the length, or duration, of the particle simulation in days. Default is 30 (to be used for test run)
        s_bounds : list, optional
            Spatial bounds for seeding written as bounds=[x1,y1,x2,y2]. Default is None, i.e. the whole basin is covered
        z : float, optional
            Depth at which to seed particles in [m]. Default is -0.5m. 
        tstep : timedelta, optional
            Time step used for OpenDrift simulation. Default is 6 hours
        hdiff : float, optional
            Horizontal diffusivity of particles, in [m2/s]. Default is 10m2/s
        termvel : float, optional
            Terminal velocity representing buoyancy of particles, in [m/s]. Default is None, meaning termvel is defined in __init__     
        raster : bool, optional
            Boolean stating whether to launch 'particle_raster' method. Default is True 
        res : float, optional
            Spatial resolution for raster in [m]. Default is 3km
        crs : str, optional
            EPSG string for raster. Default is 4326        
        tinterp : float, optional
            Timestep (in hours) used for interpolation of trajectories in particle_raster method. Default is None
        r_bounds : list, optional
            Spatial bounds for computation of histogram (particle_raster) written as bounds=[x1,y1,x2,y2]. Default is None (full basin)
        use_path : str, optional
            Path to .tif file representing density of human use of marine environment, used for 'weights' of particles in histogram calculation. 
            If no use_path is given, a weight of 1 is given to all particles ('even' for even distribution).
        decay_rate : float, optional
            Decay rate of particles in [days-1]. Default is 0 defined in __init__
        aggregate : str, optional
            String indicating whether trajectories should be aggregated by calculating their maximum ('max') or 'mean' over time. Default is 'mean'
        depth_layer : str, optional
            Depth layer chosen for computing histogram. One of 'full_depth', 'water_column', 'surface' or 'seafloor'. Default is 'full_depth'
        z_bounds : list, optional
            Two parameters, given as z_bounds=[z_surface, z_bottom], determining the depth layers' thickness in [m]. 
            The first represents vertical distance from the ocean surface (z=0), whhile the second represents vertical distance from the ocean bottom, given by the bathymetry. 
            Default is z_bounds=[10,100].
        loglevel : int, optional
            OpenDrift loglevel. Set to 0 (default) to retrieve all debug information.
            Provide a higher value (e.g. 20) to receive less output.
        save_to : str, optional
            Name of the folder to save raster figures into. If None (default), rasters are saved within the rasters directory in basedir. 
        plot : 
        
        particle_status : str, optional
            Filter particles based on their status at the end of the run i.e., whether they are still active or have beached or sunk. Options are ['all', 'stranded', 'seafloor', 'active'], Default is 'active' 
        """
        
        
        if use_path != 'even' and use_label == 'even':
            raise ValueError('When specifying a use_path, please also specify a use_label to make the tif file recognizable.')
        
        # raise error if particle_path is already given -> DEPRECATED. IF PARTICLE_PATH IS NOT GIVEN, MAKE PARTICLE_SIMULATION. OTHERWISE GO STRAIGHT TO RASTER. 
        context = self.context 
        
        reps = None 
        tshift = None
        
        self.particle_status = particle_status
        
        if termvel is None:
            termvel = self.termvel
         
        if decay_rate is None:
            decay_rate = self.decay_rate
        
        if self.particle_path is None:
            #logger.debug('Run starting')
                
            self.pnum = pnum
            
            
            print(f'----- STARTING RUN -----')
            t0 = datetime.strptime(start_time, '%Y-%m-%d')
            start_time = t0.strftime("%Y-%m-%d")
            self.origin_marker = 0
            self.particle_simulation(pnum=pnum, start_time=start_time, season=season, duration_days=duration_days, crs=crs, s_bounds=s_bounds, z=z, tstep=tstep, hdiff=hdiff, termvel=termvel, loglevel=loglevel)
      
        
        
        ##### THIS COULD PROBS GO IN INIT. for now copying it also in repeat_run
        if save_to is None:
            Path(self.basedir / 'rasters').mkdir(parents=True, exist_ok=True)
            outputdir = self.basedir / 'rasters'
        else:
            Path(save_to).mkdir(parents=True, exist_ok=True)
            outputdir = Path(save_to)
        self.outputdir = outputdir
        ###############################
        
        raster_path, raster_exists = self.cache.raster_cache(poly_path=self.poly_path, pnum=pnum, ptot=None, duration_days=duration_days, s_bounds=s_bounds, z=z, tstep=tstep, hdiff=hdiff, termvel=termvel, crs=crs, tinterp=tinterp, r_bounds=r_bounds, use_path=use_path, decay_rate=decay_rate, aggregate=aggregate, depth_layer=depth_layer, z_bounds=z_bounds, particle_status=particle_status)
        
        print('STARTING RASTER COMPUTATION')
        # compute raster
        if raster == True and not raster_exists:
            if type(use_path) == str:
                r = self.particle_raster(res=res, crs=crs, tinterp=tinterp, r_bounds=r_bounds, use_path=use_path, decay_rate=decay_rate,  aggregate=aggregate, depth_layer=depth_layer, z_bounds=z_bounds, particle_status=particle_status)
                #self.raster = r
            elif type(use_path) == list:
                for idx, use in enumerate(use_path):
                    r = self.particle_raster(res=res, crs=crs, tinterp=tinterp, r_bounds=r_bounds, use_path=use, decay_rate=decay_rate,  aggregate=aggregate, depth_layer=depth_layer, z_bounds=z_bounds, particle_status=particle_status, save_r=False)
                    if idx == 0:
                        self.raster = r
                    else:
                        self.raster = xr.merge([self.raster, r.rename({'r0': f'r{idx}', 'lon_bin': f'lon_bin_{idx}', 'lat_bin': f'lat_bin_{idx}'})], compat='broadcast_equals', join='outer', fill_value=0, combine_attrs='no_conflicts')
            
            else:
                raise ValueError('"use_path" not recognised. "use_path" must be either a string or list of strings containing path(s) to use layers.')
        
            # save final raster(s) and delete temporary files

            for i, r in enumerate(self.raster.data_vars): # save all available rasters in output directory
                    print(f'saving tiff file #{i+1}...')
                    #raster_path = self.get_raster_paths(decay_rate, reps, tshift, use_label)
                    #filename = self.particle_path.split('/')[-1][:-3]
                    #self.raster[f'{r}'].rio.to_raster(outputdir / f'raster_{i+1}.tif')
                    self.raster[f'{r}'].rio.to_raster(raster_path) ###### WARNING: NEL CASO DI PIù USI DI INPUT VIENE SOVRASCRITTO, CORREGGI
                    # save corresponding thumbnails in save output directory
                    #self.plot(r=self.raster[f'{r}'], save_fig=f'{str(outputdir)}/thumbnail_raster_{i+1}.png')
                    print(f'saving thumbnail #{i+1}...')
                    self.plot(r=self.raster[f'{r}'])#, save_fig=f'{str(self.outputdir)}/thumbnail_{raster_path[:-4]}.png')

        elif raster == True and raster_exists: 
            self.raster = xr.Dataset()
            self.raster['r0'] = rxr.open_rasterio(raster_path).isel(band=0).rename({'x': 'lon_bin', 'y': 'lat_bin'})
        
        
        return self
    
    
    def make_poly(self, lon, lat, crs='4326', write=True):
        """
        Creates shapely.Polygon where particles will be released homogeneously and writes it to a shapefile. 
        
        Parameters
        ----------
        lon : list or array
            Either a list giving 2 bounds or an array of lon coordinates.
        lat : list or array
            Either a list giving 2 bounds or an array of lat coordinates.
        crs : str, optional
            EPSG string for Polygon crs. Default is '4326'
        write : bool, optional
            Whether to save the polygon as a shapefile in the 's' directory. Default is True
        """
        
        
        Path(self.basedir / 'polygons').mkdir(parents=True, exist_ok=True)
        poly_path = f'polygon-crs_epsg:{crs}-lon_{np.round(lon[0],4)}_{np.round(lon[1],4)}-lat—{np.round(lat[0],4)}_{np.round(lat[1])}.shp'
        q = self.basedir / 'polygons' / poly_path
        
        if len(np.array(lon)) == 2 and len(np.array(lat)) == 2:
            _df = pd.DataFrame() # create dataframe 
            _df['lon'] = lon # add lon and lat arrays
            _df['lat'] = lat
            _df['geometry'] = Polygon(zip([lon[0],lon[0],lon[1],lon[1]], [lat[0],lat[1],lat[1],lat[0]]))
            poly = gpd.GeoDataFrame(_df, geometry="geometry").set_crs(epsg=crs).to_crs(epsg='4326') # transform into geodataframe

        elif len(np.array(lon)) > 2 and len(np.array(lat)) > 2:
            x, y = np.meshgrid(lon, lat)

            _df = pd.DataFrame() # create dataframe 
            _df['lon'] = x.ravel() # add lon and lat arrays

            _df['lat'] = y.ravel()
            _df['geometry'] = _df.apply(lambda r: Point(r.lon, r.lat), axis=1) # create point geometries

            df = gpd.GeoDataFrame(_df, geometry="geometry") # transform into geodataframe

            # buffer. cap_style = 3 means a square buffer. (creates square around centroid with distance 'res' between centroid and side)
            buffer = df['geometry'].buffer(np.diff(lon).mean(), cap_style = 3)
            # add buffered polygon to geo df
            df['squares'] = buffer
            gds = gpd.GeoDataFrame(df, geometry="squares")
            # create polygon by dissolving squares into each other
            poly = gds.dissolve()
            # set crs and reproject to desired crs 
            poly = poly.set_crs(epsg=crs).to_crs(epsg='4326')
        else:
            raise ValueError("'lon' and 'lat' must have lenght larger than 2")
        
        if write is True:
            poly.to_file(str(q), driver='ESRI Shapefile')
            self.poly_path = str(q)
        else:
            return poly
    
# [pnum, start_time, season, duration_days, s_bounds, z, tstep, hdiff, termvel, crs]
    def particle_simulation(self, pnum, start_time='2019-01-01', season=None, duration_days=30, s_bounds=None, z=-0.5, tstep=timedelta(hours=4), hdiff=10, termvel=None, crs='4326', loglevel=40):
        """
        Method to start a trajectory simulation, after initial configuration, using OpenDrift by MET Norway.
        Uses OceanDrift module. 
                
        This method is currently optimised for use over the Mediterranean and Black Sea only. 
        Forcing data is streamed from Copernicus (ocean currents, wind, mixed layer depth), while GEBCO bathymetry is stored locally. 
        
        Particles are seeded homogeneously over polygon created using make_poly() method. 
        
        
        Parameters
        ----------
        context : str
            String defining the context of the simulation i.e., the model data to be used for particle forcing. Options are 'med-cmems', 'bs-cmems' and 'bridge-bs'. 
        pnum : int 
            The number of particles to be seeded
        start_time : str, optional
            The start time of the simulation. Default is '2019-01-01'
        season : str, optional
            Season in which simulation should be run. Defines start_time automatically. Default is None
        duration_days : int, optional
            Integer defining the length, or duration, of the particle simulation in days. Default is 30 (to be used for test run)
        s_bounds : list, optional
            Spatial bounds for seeding written as bounds=[x1,y1,x2,y2]. Default is None, i.e. the whole basin is covered
        z : float, optional
            Depth at which to seed particles in [m]. Default is -0.5m. 
        tstep : timedelta, optional
            Time step used for OpenDrift simulation. Default is 6 hours
        hdiff : float, optional
            Horizontal diffusivity of particles, in [m2/s]. Default is 10m2/s
        termvel : float, optional
            Terminal velocity representing buoyancy of particles, in [m/s]. Default is None, meaning termvel is defined in __init__      
        crs : str, optional
            EPSG string for polygon. Default is 4326   
        loglevel : int, optional
            OpenDrift loglevel. Set to 0 (default) to retrieve all debug information.
            Provide a higher value (e.g. 20) to receive less output.
        """
        
        t_0 = T.time()   
        
        context = self.context 
 
        if termvel is None:
            termvel = self.termvel
        
        
        # raise error if unsupported context is requested
        avail_contexts = ['bridge-bs', 'med-cmems', 'bs-cmems']
        if context not in avail_contexts:
            raise ValueError(f"Unsupported context given. Context variable must be one of {avail_contexts}")
        self.context = context
        
        # commented as now everything goes into cachedir
        #Path(self.basedir / 'particles').mkdir(parents=True, exist_ok=True)
        
        ### time settings ####
        ssns = {'summer': datetime(2019,6,1),
               'winter': datetime(2019,12,1),
               'spring': datetime(2019,3,1),
               'autumn': datetime(2019,9,1)}

        if season in ssns.keys():
            start_time = ssns[season]  
        else:
            start_time = datetime.strptime(start_time, '%Y-%m-%d')
        end_time = start_time + timedelta(days=duration_days) # this is printed in particle_path
        time_step = tstep

        if self.tseed is None:
            self.tseed = timedelta(days=int(duration_days * 20 / 100)) #tseed is 20% of total duration
            
        tseed = self.tseed
        
        duration = timedelta(days=duration_days)+tseed-timedelta(days=1) # true duration of the run. the tseed time period is then deleted.
        
        self.tstep = tstep
        
        # polygon used for seeding of particles. by default, it is the whole basin. but if lon and lat are given, a new polygon is created using those bounds. 
        if s_bounds is not None:
            lon = s_bounds[0::2]
            lat = s_bounds[1::2]
            self.make_poly(lon, lat, crs=crs)
        else:
            pass

        
        # path to write particle simulation file. also used for our 'cache'    
        poly = gpd.read_file(self.poly_path)
        bds = np.round(poly.total_bounds, 4) # only used in particle_path
        t0 = start_time.strftime('%Y-%m-%d')
        t1 = end_time.strftime('%Y-%m-%d')
        
        # initialise OpenDrift object
        if self.pressure == 'oil':
            self.o = OpenOil(loglevel=loglevel)
        else:
            self.o = OceanDrift(loglevel=loglevel) 
        
        
        file_path, file_exists = self.cache.particle_cache(poly_path=self.poly_path, pnum=pnum, start_time=start_time, season=season, duration_days=duration_days, s_bounds=s_bounds, z=z, tstep=tstep, hdiff=hdiff, termvel=termvel, crs=crs)
        
        # if a file with that name already exists, simply import it  
        if file_exists == True:
            #self.o.io_import_file(str(q)) # this is sometimes too heavy
            #print('about to aggregate the following files:', file_path)
            ps = xr.open_mfdataset(file_path)
            print(f'NOTE: File with these configurations already exists within {self.basedir} and has been imported. Please delete the existing file to produce a new simulation.') ### this might be irrelevant now with cachedir

        # otherwise, run requested simulation
        elif file_exists == False:
            print('adding landmask...')
            # landmask from cartopy (from "use shapefile as landmask" example on OpenDrift documentation)
            shpfilename = shpreader.natural_earth(resolution='10m',
                                    category='cultural',
                                    name='admin_0_countries')
            reader_landmask = reader_shape.Reader.from_shpfiles(shpfilename)
            
            self.o.add_reader([reader_landmask])
            self.o.set_config('general:use_auto_landmask', False)  # Disabling the automatic GSHHG landmask
            
            # bridge-bs context is VERY OLD, NEEDS FIXING. import relevant readers based on context
            if context == 'bridge-bs': # local WP2 data
                
                if self.localdatadir is None:
                    raise ValueError('bridge-bs data not found. Please provide absolute path to directory containing oceanographic / atmospheric data from bridge-bs, i.e. localdatadir')
                else:    
                    bridge_dir = Path(self.localdatadir)
                
                dates = pd.date_range(start_time.strftime("%Y-%m-%d"),(start_time+duration).strftime("%Y-%m-%d"),freq='d').strftime("%Y-%m-%d").tolist()
                
                uvpaths={}
                mldpaths={}
                windpaths = {}
                
                #for idx in range(start_time.year, end_time.year+1):
                for idx, d in enumerate(dates):
                    date = d.replace('-', '')
                    uvpaths[idx] = str(bridge_dir / f'BS*{date}*UV.nc')
                    mldpaths[idx] = str(bridge_dir / f'BS*{date}*MLD.nc') 
                
                uv_path = list(uvpaths.values())
                mld_path = list(mldpaths.values())
                
                for i in range(start_time.year, end_time.year+1):
                    windpaths[i] = str(bridge_dir / f'era5_y{i}.nc')
                
                wind_path = list(windpaths.values())
                bathy_path = str(bridge_dir / 'bs_bathymetry.nc')
                
            elif context == 'bs-cmems': # copernicus Black Sea data (remote connection)
                ocean_id = "cmems_mod_blk_phy-cur_my_2.5km_P1D-m"
                wind_id = "cmems_obs-wind_glo_phy_my_l4_P1M" # this is a global reanalysis product, so same in all contexts
                mld_id = 'cmems_mod_blk_phy-mld_my_2.5km_P1D-m'
                bathy_id = 'cmems_mod_glo_phy_my_0.083deg_static' # this is a global reanalysis product, so same in all contexts
                
                #DATASET_ID = 'cmems_mod_blk_phy-cur_my_2.5km_P1D-m'
                #userinfo = self.get_userinfo('my.cmems-du.eu')
                #uv_path = f'https://{userinfo}my.cmems-du.eu/thredds/dodsC/{DATASET_ID}'
                
                #WIND_ID = 'cmems_obs-wind_glo_phy_my_l4_P1M' 
                #wind_path = f'https://{userinfo}my.cmems-du.eu/thredds/dodsC/{WIND_ID}'
                
                #mld_ID = 'cmems_mod_blk_phy-mld_my_2.5km_P1D-m'#'bs-cmcc-mld-rean-d'
                #mld_path = f'https://{userinfo}my.cmems-du.eu/thredds/dodsC/{mld_ID}'        
                
                #bathy_path = '/home/sbosi/data/input/bathymetry_gebco_2022_n46.8018_s29.1797_w-6.5918_e43.8574.nc'
            
            elif context == 'med-cmems': # copernicus Med Sea data (remote connection)
                ocean_id = "med-cmcc-cur-rean-h"
                wind_id = "cmems_obs-wind_glo_phy_my_l4_P1M" # this is a global reanalysis product, so same in all contexts
                mld_id = 'med-cmcc-mld-rean-d'
                bathy_id = 'cmems_mod_glo_phy_my_0.083deg_static' # this is a global reanalysis product, so same in all contexts
                
                #DATASET_ID = 'med-cmcc-cur-rean-h' #hourly
#                DATASET_ID = 'med-cmcc-cur-rean-d' #daily
                #userinfo = self.get_userinfo('my.cmems-du.eu') # is this correct?
                #uv_path = f'https://{userinfo}my.cmems-du.eu/thredds/dodsC/{DATASET_ID}'                
                
                #WIND_ID = 'cmems_obs-wind_glo_phy_my_l4_P1M'              
                #wind_path = f'https://{userinfo}my.cmems-du.eu/thredds/dodsC/{WIND_ID}'
            
                #mld_ID = 'med-cmcc-mld-rean-d'
                #mld_path = f'https://{userinfo}my.cmems-du.eu/thredds/dodsC/{mld_ID}'      
                
                #bathy_path = '/home/sbosi/data/input/bathymetry_gebco_2022_n46.8018_s29.1797_w-6.5918_e43.8574.nc'

            else:
                raise ValueError("Unsupported context. Please choose one of 'bridge-bs', 'bs-cmems' or 'med-cmems'.")

            reader_ids = [ocean_id, wind_id] # list of copernicus product ids, including bathymetry and mld if simulation is 3D
            if self.depth == True:
                reader_ids.append(mld_id)
                reader_ids.append(bathy_id)

            readers = [] # list of opendrift readers created from those copernicus products
            for path in reader_ids: 
                data = copernicusmarine.open_dataset(dataset_id = path,
                                                   start_datetime = start_time, 
                                                   end_datetime = start_time + duration) # this is the entire time range, including tseed
                r = Reader(data)
                readers.append(r)

            print('adding readers...')
            self.o.add_reader(readers)
            
            #print('adding ocean readers...')
            #self.o.add_readers_from_list(uv_path, lazy=True) 
            #print('adding wind readers...')
            #self.o.add_readers_from_list(wind_path, lazy=True) # add reader from local file
            
            #if self.depth == True:
             #   print('adding mixed layer readers...')
              #  self.o.add_readers_from_list(mld_path, lazy=True) # add reader from local file
              #  print('adding bathymetry readers...')
               # bathy_reader = reader_netCDF_CF_generic.Reader(self.bathy_path)
                #self.o.add_reader(bathy_reader)

            
            if context == 'bridge-bs':
                for k,r in self.o.readers.items(): 
                    r.always_valid = True                  
            
                
            # some OpenDrift configurations
            self.o.set_config('general:coastline_action', 'stranding') # behaviour at coastline. 'stranding' means beaching of particles is allowed
            self.o.set_config('drift:horizontal_diffusivity', hdiff)  # horizontal diffusivity
            self.o.set_config('drift:advection_scheme', 'euler') # advection schemes (default is 'euler'). other options are 'runge-kutta', 'runge-kutta4'
            
            # uncertainty
            #self.o.set_config('drift:current_uncertainty', .1)
            #self.o.set_config('drift:wind_uncertainty', 1)

            ##### SEEDING #####
            print('seeding particles...')
            np.random.seed(None) # to avoid seeding in same exact position each time             

            # if simulation is 3D, set 3D parameters (terminal velocity, vertical mixing, seafloor action) and seed particles over polygon
            if self.depth == True:
                self.o.set_config('seed:terminal_velocity', termvel) # terminal velocity
                self.o.seed_from_shapefile(shapefile=str(self.poly_path), number=pnum, time=[start_time, start_time+self.tseed], 
                                           terminal_velocity=termvel, z=z, origin_marker=self.origin_marker, radius=1e4)
                #self.o.set_config('vertical_mixing:diffusivitymodel', 'windspeed_Large1994')
                self.o.set_config('general:seafloor_action', 'deactivate')
                self.o.set_config('drift:vertical_mixing', True)
            
            # if simulation is 2D, simply seed particles over polygon
            else:
                self.o.seed_from_shapefile(shapefile=str(self.poly_path), number=pnum, time=[start_time, start_time+self.tseed],
                                           origin_marker=self.origin_marker, radius=1e4)
            
            # run simulation and write to temporary file
            #### THIS IS THE CORRECT WAY
            #with tempfile.TemporaryDirectory("particle", dir=self.basedir) as qtemp:
            qtemp = tempfile.TemporaryDirectory("particle", dir=self.basedir)
            temp_outfile = qtemp.name + f'/temp_particle_file_marker-{self.origin_marker}.nc'

            #### INCORRECT WAY FOR DEBUGGING
            #Path(self.basedir / 'temp-particle-files').mkdir(parents=True, exist_ok=True)
            #temppath = self.basedir / 'temp-particle-files' / f'temp_particle_file-marker-{self.origin_marker}.nc'
            #temp_outfile = str(temppath)
            #################################

            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")

            self.o.run(duration=duration, #end_time=end_time, 
                       time_step=time_step, #time_step_output=timedelta(hours=24), 
                       outfile=temp_outfile, export_variables=['lon', 'lat', 'z', 'status', 'age_seconds', 'origin_marker', 'sea_floor_depth_below_sea_level', 'ocean_mixed_layer_thickness'])# file was getting too big, 'x_sea_water_velocity', 'y_sea_water_velocity', 'x_wind', 'y_wind'])

            elapsed = (T.time() - t_0)
            print("total simulation runtime %s" % timedelta(minutes=elapsed/60)) 


            #### CHECK HERE IF READERS WERE PROCESSED CORRECTLY ####
            if hasattr(self.o, 'discarded_readers'):
                logger.warning(f'Readers {self.o.discarded_readers} were discarded. Particle transport will be affected')

            #### A BIT OF POST-PROCESSING ####
            print('writing to netcdf...')

            _ps = xr.open_dataset(temp_outfile) # open temporary file
            #print('time len before processing', len(_ps.time))

            # keep 'inactive' particles visible (i.e. particles that have beached or gotten stuck on seafloor)
            ps = _ps.where(_ps.status>=0).ffill('time') 
            #print('time len after ffill inactive', len(ps.time))

            # align trajectories by particles' age
            shift_by = -ps.age_seconds.argmin('time') 
            if self.tseed.days!=0: 

                def shift_by_age(da, shift_by):
                    newda = xr.apply_ufunc(np.roll, da, shift_by, input_core_dims=[['time'], []], output_core_dims=[['time']], vectorize=True, dask='parallelized', keep_attrs='drop_conflicts')
                    return newda

                ps=ps.apply(shift_by_age, shift_by=shift_by) 
                #print('time len after shift by age', len(ps.time))

                # remove tail of nan values
                if np.any(np.isnan(ps.age_seconds)): # if there are any nan values in age_seconds
                    _idx = ps.age_seconds.argmin('time', skipna=False) 
                    idx = _idx.where(_idx!=0).min() # time index of first nan value across all trajectories
                    ps = ps.isel(time=slice(None, int(idx)))
                    #print('idx', idx)
                    #print('time len after chopping tail of nans', len(ps.time))

            # write useful attributes
            ps = ps.assign_attrs({'total_bounds': poly.total_bounds, 'start_time': t0, 'duration_days': duration_days, 'pnum': pnum, 'hdiff': hdiff,
                                  'tseed': self.tseed.total_seconds(), 'tstep': tstep.total_seconds(), 'termvel': termvel, 'depth': str(self.depth),
                                  'poly_path': str(self.poly_path)}) 
                                    # , 'opendrift_log': str(self.o)}) 
                                    #removing this attribute as i already have the check for discarded readers elsewhere. 
                                    #This way I can compare attributes more easily for new caching method

            #self.tstep = tstep
            #self.tseed = tseed # should already be there

            #print('time len just before writing temp file', len(ps.time))

            ps.to_netcdf(str(file_path))
            print(f"done. NetCDF file '{self.particle_path}' created successfully.")

        
        self.particle_path = str(file_path)

        self.ds = ps
        
        if 'qtemp' in locals():
            Path(temp_outfile).unlink()
            os.rmdir(qtemp.name)

        

        
        pass

    
    def particle_raster(self, res=4000, crs='4326', tinterp=None, r_bounds=None, use_path='even', decay_rate=None, aggregate='mean', depth_layer='full_depth', z_bounds=[1,-10], particle_status='all', save_r=True):
        """
        Method to compute a 2D horizontal histogram of particle concentration using the xhistogram.xarray package. 
        
        If a use_path is given, particle 'weight' is the value of the given human use in the particle's initial position. 
        Default weight is 1 for all particles.
        
        
        Parameters
        ----------
        res : float, optional
            Spatial resolution for raster in [m]. Default is 3km
        crs : str, optional
            EPSG string for raster. Default is 4326
        tinterp : float, optional
            Timestep (in hours) used for interpolation of trajectories in particle_raster method. Default is None
            ds
        r_bounds : bounds, optional
            Spatial bounds for computation of raster written as bounds=[x1,y1,x2,y2]. Default is None (bounds are taken from self.)
        use_path : str, optional
            Path to file representing density of human use, used for 'weights' of particles in histogram calculation. 
            If no use_path is given, a weight of 1 is given to all particles by default ('even').
        decay_rate : float, optional
            Decay rate of particles in [days-1]. Default is None, meaning decay_rate is defined in __init__
        aggregate : 
            String indicating whether trajectories should be aggregated by calculating their maximum ('max') or 'mean' over time. Default is 'mean'
        depth_layer : str, optional
            Depth layer chosen for computing histogram. One of 'full_depth', 'water_column', 'surface' or 'seafloor'. Default is 'full_depth'
        z_bounds : list, optional
            Two parameters, given as z_bounds=[z_surface, z_bottom], determining the depth layers' thickness in [m]. The first represents vertical distance from the ocean surface (z=0), whhile the second represents vertical distance from the ocean bottom, given by the bathymetry. Default is z_bounds=[10,100].
        particle_status : str, optional
            Parameter controlling which particles to include in raster, based on their status at the end of the run. Options are ['all', 'stranded', 'seafloor', 'active'], Default is 'active' 
        save_r : bool, optional
            Whether to save newly calculated raster in self.raster
        """
    
        t_0 = T.time()
         
        if decay_rate is None:
            decay_rate = self.decay_rate
        
        _ds = self.get_ds
                
        status = {'active': 0, 'stranded': 1, 'seafloor': 2}
        
        if particle_status in status.keys():
            traj = _ds.trajectory.where(_ds.isel(time=-1).status==status[particle_status]).dropna('trajectory').data
            ds = _ds.sel(trajectory=traj)
        else: 
            ds = _ds
    
        ### TIME INTERPOLATION ###
        if tinterp is not None:
            new_time = np.arange(pd.to_datetime(ds.time[0].values), pd.to_datetime(ds.time[-1].values),timedelta(hours=tinterp)) #new time variables used for interpolation
            ds = ds.interp(time=new_time, method='slinear') # interpolate dataset 
            ds['tinterp'] = tinterp
        else:
            pass
            

        ### BINS ### 
        
        # this polygon is only used to extract bounds for construction of bins 
        if r_bounds is not None: # if r_bounds are given, meaning we are not using the full basin, create polygon to use for aggregation / visualisation
            poly = self.make_poly(lon=[r_bounds[0], r_bounds[2]], lat=[r_bounds[1], r_bounds[3]], write=False)
        else:    # if no r_bounds are given, use seeding polygon
            if self.poly_path is not None:
                poly = gpd.read_file(self.poly_path).to_crs('epsg:4326').buffer(distance=.2) # buffer is added because of radius=1e4 when seeding. if this is not done, some particles may be cut out even in their starting position
                r_bounds = poly.total_bounds
            elif self.poly_path is None: 
                raise ValueError("No spatial domain found to perform aggregation. Please provide 'r_bounds' manually.")
                
           
        # if no use path is given, compute bins from resolution and bounds of polygon
        if use_path == 'even':
            weight = np.ones((ds.lon.shape))             # all particles have weight 1
            ds = ds.assign({'weight': (('trajectory', 'time'), weight)}, )
            xbin = np.arange(r_bounds[0],r_bounds[2]+res/1e5,res/1e5) # factor 1e5 is to approximately convert m to latlon deg
            ybin = np.arange(r_bounds[1],r_bounds[3]+res/1e5,res/1e5)
        else: 
            _use = rxr.open_rasterio(use_path).sortby('x').sortby('y').isel(band=0).drop('band')
            spatial_ref = _use.spatial_ref.crs_wkt
            bds_reproj = poly.to_crs(spatial_ref).total_bounds 
            #create `weight` variable from value of `use` at starting positions of particles
            
            use = _use.sel(x=slice(bds_reproj[0], bds_reproj[2]), y=slice(bds_reproj[1], bds_reproj[3])).rio.reproject(spatial_ref, resolution=res, nodata=0).rio.reproject('epsg:4326', nodata=0).sortby('x').sortby('y').fillna(0)
            
            _weight = use.sel(x=ds.isel(time=0).lon, y=ds.isel(time=0).lat, method='nearest')/len(self.ds.time) # matching timestep of simulation
            
            ds['weight'] = _weight

            #use grid from use file as `bins` to compute histogram. (but need to shift from center to get same coordinates)
            #xbin = np.append(use.x - res/2, use.x[-1]+res/2)
            #ybin = np.append(use.y - res/2, use.y[-1]+res/2) 
            # res is now given in m
            xbin = np.append(use.x - np.diff(use.x).mean()/2, use.x[-1]+np.diff(use.x).mean()/2)
            ybin = np.append(use.y - np.diff(use.y).mean()/2, use.y[-1]+np.diff(use.y).mean()/2) 
        
        # give 0 weight to beached particles after they have beached to avoid strange peaks
        ds['weight'] = ds.weight.where(ds.status==0, 0)
        
        ### DECAY RATE ####
        k = decay_rate #decay coefficient given by user
        y = np.exp(-k*(ds.time-ds.time.min()).astype(int)/60/60/1e9/24) #decay function 
        ds['decay'] = y  
        ds['weight'] = ds.weight*ds.decay
        
        #### HISTOGRAMS. 2 CASES: 2D or 3D ####
        if self.depth is None:
            if len(np.unique(ds.z)) > 2:
                self.depth = True
            else:
                self.depth = False
        else:
            pass
        
        # FILTER OUT PARTICLES WHERE WEIGHT IS 0 TO FREE UP MEMORY.
        # NB: filter out entire TRAJECTORIES of particles which are released in cells where use layer is 0. this is done by extracting their particle ID. 
        p_id = ds.isel(time=0).where(ds.weight.isel(time=0)!=0).trajectory.data
        ds = ds.sel(trajectory=p_id).dropna('trajectory', 'all')
        
        # 3D
        if self.depth == True:
            ds = ds.assign(depth=-ds.z)
            # condition to filter out ds based on depth_layer
            cond = {'surface': ds.depth<=z_bounds[0], 'seafloor': ds.depth>(ds.sea_floor_depth_below_sea_level+z_bounds[1]), 'water_column': np.logical_and(ds.depth>z_bounds[0], ds.depth<-z_bounds[1]), 'mixed_layer': ds.depth <= ds.ocean_mixed_layer_thickness}
            
            if depth_layer in cond.keys():
                ds = ds.where(cond[depth_layer])
                
        elif self.depth == False:
            pass
        else:
            raise ValueError('cannot detect whether 2D or 3D')
        
        self.ds = ds

        
        ### chunking + aggregation method (sum, max, etc)
        #step = 100 # this is completely arbitrary for now
        #slices = int(len(ds.time)/step) # numer of slices / chunks
        
        ### need to rewrite this and make it cleaner (maybe use .exec()?), but for now:
        #### AGGREGATION METHOD ####
        # qtemp = str('tempfiles'+"{:05d}".format(random.randint(0,99999)))
        # Path(self.basedir / qtemp).mkdir(parents=True, exist_ok=True)
        qtemp = tempfile.TemporaryDirectory("raster", dir=self.basedir)
        
        if self.reps is not None:
            reps = self.reps
        else:
            reps = 1
        
        for i in range(0,reps):
            d = ds.where(ds.origin_marker==i).dropna('trajectory', 'all')
            hh = histogram(d.lon, d.lat, bins=[xbin, ybin], dim=['trajectory'], weights=d.weight, block_size=len(d.trajectory)).chunk({'lon_bin':10, 'lat_bin': 10}).sum('time') 
            hh.to_netcdf(f'{qtemp.name}/temphist_{i}.nc') 
            print(f'Raster done for rep {i}')
            del hh, d  
                            
        if aggregate == 'mean':    
            _h = xr.open_mfdataset(f'{qtemp.name}/temphist*.nc', concat_dim='slices', combine='nested').mean('slices').histogram_lon_lat            
        elif aggregate == 'max': 
            _h = xr.open_mfdataset(f'{qtemp.name}/temphist*.nc', concat_dim='slices', combine='nested').max('slices').histogram_lon_lat
        elif aggregate.startswith('p'):
            quantile = eval(aggregate.split('p')[-1])/100
            _h = xr.open_mfdataset(f'{qtemp.name}/temphist*.nc', concat_dim='slices', combine='nested').load().quantile(quantile, dim='slices').histogram_lon_lat # does not work unless I .load()
        else:
            raise ValueError("'aggregate' must be one of 'mean', 'max' or 'p95'.")        
        
        h = _h.transpose()   
        
        # normalizzazione. divido per numero di traiettorie (aka particelle) e moltiplico per numero totale di celle
        #tot_cells = len(h.stack(box=('lon_bin', 'lat_bin')).dropna('box'))
        #tot_cells = h.where(np.isnan(h),1).sum().load()
        tot_cells = int(h.count().load())
        #h = h/self.pnum*tot_cells        
        h = h/len(ds.trajectory)*tot_cells   
        
        # write geo information to xarray and save as geotiff
        r = (
            xr.DataArray(h) # need to transpose it because xhistogram does that for some reason
            .rio.write_nodata(np.nan)
            .rio.write_crs('epsg:'+str(crs))
            .rio.write_coordinate_system())
        
        r=r.assign_attrs({'use_path': use_path}).to_dataset().rename({'histogram_lon_lat': 'r0'})
        
        ####### Landmask ######
        if 'bs' in self.context:
            shpfilename = f'{DATA_DIR}/polygon-bs-full-basin.shp' #use black sea polygon for masking in bs contexts

        else: # otherwise, use cartopy natural earth polygon
            import cartopy.io.shapereader as shpreader

            shpfilename = shpreader.natural_earth(resolution='10m',
                                        category='physical',
                                        name='ocean')

        _mask = gpd.read_file(shpfilename)

        if particle_status in ['active', 'seafloor']:
            mask = _mask
        else:
            mask = _mask.buffer(distance=0.1) # if including beached particles, consider a buffer around ocean 
            
        # remove temporary files and folder
        for p in Path(qtemp.name).glob("temphist*.nc"):
            p.unlink()
        
        # rm qtemp        
        os.rmdir(qtemp.name)
        
        elapsed = (T.time() - t_0)
        print("--- RASTER CREATED IN %s seconds ---" % timedelta(minutes=elapsed/60))
        
        if save_r == True:
            self.raster = r
        else:
            pass
        
        return r
    
    def plot(self, r=None, xlim=None, ylim=None, cmap=spectral_r, shading='flat', vmin=None, vmax=None, norm=None, coastres='10m', proj=ccrs.PlateCarree(), dpi=120, figsize=[10,7], rivers=False, title=None, save_fig='thumbnail'):
        """
        Plot particle_raster outputs.
        
        Parameters
        ----------
        xlim, ylim : array-like, optional
            Specify *x*- and *y*-axis limits.
        cmap : matplotlib colormap name or colormap, optional
            Plot colormap. Default is seaborn's 'spectral' map, reversed. 
        shading : {'flat', 'nearest', 'gouraud', 'auto'}, optional
            The fill style for the quadrilateral; defaults to
            'flat' or :rc:`pcolor.shading`. Possible values:

            - 'flat': A solid color is used for each quad. The color of the
              quad (i, j), (i+1, j), (i, j+1), (i+1, j+1) is given by
              ``C[i, j]``. The dimensions of *X* and *Y* should be
              one greater than those of *C*; if they are the same as *C*,
              then a deprecation warning is raised, and the last row
              and column of *C* are dropped.
            - 'nearest': Each grid point will have a color centered on it,
              extending halfway between the adjacent grid centers.  The
              dimensions of *X* and *Y* must be the same as *C*.
            - 'gouraud': Each quad will be Gouraud shaded: The color of the
              corners (i', j') are given by ``C[i', j']``. The color values of
              the area in between is interpolated from the corner values.
              The dimensions of *X* and *Y* must be the same as *C*. When
              Gouraud shading is used, *edgecolors* is ignored.
            - 'auto': Choose 'flat' if dimensions of *X* and *Y* are one
              larger than *C*.  Choose 'nearest' if dimensions are the same.

            See :doc:`/gallery/images_contours_and_fields/pcolormesh_grids`
            for more description.
        vmin, vmax : float, optional
            Values to anchor the colormap, otherwise they are inferred from the
            data and other keyword arguments. When a diverging dataset is inferred,
            setting one of these values will fix the other by symmetry around
            ``center``. Setting both values prevents use of a diverging colormap.
            If discrete levels are provided as an explicit list, both of these
            values are ignored. 
        norm : matplotlib.colors.Normalize, optional
            If ``norm`` has ``vmin`` or ``vmax`` specified, the corresponding
            kwarg must be ``None``.
        coastres : str or :class:`cartopy.feature.Scaler`, optional
            A named resolution to use from the Natural Earth
            dataset. Currently can be one of "auto" (default), "110m", "50m",
            and "10m", or a Scaler object.
        proj : {None, 'aitoff', 'hammer', 'lambert', 'mollweide', 'polar', 'rectilinear', str}, optional
            The projection type of the `~.axes.Axes`. *str* is the name of
            a custom projection, see `~matplotlib.projections`. The default
            None results in a 'rectilinear' projection.
            Default is cartopy.crs.PlateCarree().
        dpi : float, default: :rc:`figure.dpi`
            The resolution of the figure in dots-per-inch.
        figsize : tuple, optional
            A tuple (width, height) of the figure in inches.   
        rivers : bool, optional
            Whether to plot rivers. Default is False
        save_fig : bool, optional
            Whether to save figure as png. Default is True
        """
                
        SMALL_SIZE = 8
        MEDIUM_SIZE = 10
        BIGGER_SIZE = 16
        
        if self.raster is None: 
            raise ValueError("No raster has been calculated yet. Please launch particle_raster method first.")
        
        if r is None:
            r = self.raster[list(self.raster.data_vars)[-1]]
        
        else:
            pass
        
        ### this is actually creating problems in plots, commenting for now
        # drop nan values to have cleaner thumbnails
        #lon_var = [varname for varname in r.coords if "lon" in varname][0] # find name of lon variable
        #lat_var = [varname for varname in r.coords if "lat" in varname][0] # find name of lat variable

        r = r.where(r>0)
        #r = r.dropna(str(lon_var), 'all').dropna(str(lat_var), 'all')
        
        fig = plt.figure(figsize=figsize)
        ax = plt.axes(projection=proj)
        ax.coastlines(coastres, zorder=11, color='k', linewidth=.5)
        ax.add_feature(cartopy.feature.LAND, zorder=0, facecolor='0.9') #'#FFE9B5'
        ax.add_feature(cartopy.feature.BORDERS, zorder=11, linewidth=.5, linestyle=':')
        if rivers is True:
            ax.add_feature(cartopy.feature.RIVERS, zorder=12)
        gl = ax.gridlines(draw_labels=True, dms=False, x_inline=False, y_inline=False, linewidth=.5, color='gray', linestyle='--')
        gl.top_labels = False
        gl.right_labels = False    

        im = r.plot(vmin=vmin, vmax=vmax, norm=norm, shading=shading, cmap=cmap, add_colorbar=False)
        cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
        cbar = plt.colorbar(im, cax=cax, extend='max')
        cbar.set_label('uom from input layer', rotation=90)

        if title is not None:
            ax.set_title(title+'\n use_path: '+r.use_path, fontsize=12)
        else:
            ax.set_title('use_path: '+r.use_path, fontsize=8)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)


        plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
        plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

        if save_fig is not None:
            #Path(self.basedir / 'thumbnails').mkdir(parents=True, exist_ok=True) # useful for geoplatform to keep them in same dir as raster
            plt.savefig(str(save_fig), dpi=160, bbox_inches='tight')

        return fig, ax

    
    def scatter(self, t=None, xlim=None, ylim=None, s=1, alpha=1, c='age', cmap='rainbow', coastres='10m', proj=ccrs.PlateCarree(), dpi=120, figsize=[10,7], rivers=False, save_to=None):
        """
        Visualise particle trajectories over time [days elapsed], defined by age_seconds. 
        
        
        Parameters
        ----------
        t : int, optional
            If a t is given, the scatter plot will represent the selected instant in time (int) or time interval (slice). Otherwise, it will show the entire trajectory (default). 
        xlim, ylim : array-like, optional
            Specify *x*- and *y*-axis limits.
        s : float, optional
            Marker size for scatter plot. Default is 1
        c : str, optional
            String defining variable used for colormap. Options are 'z' or 'age' (Default)
        cmap : matplotlib colormap name or colormap, optional
            Plot colormap. Default is seaborn's 'spectral' map, reversed. 
        coastres : str or :class:`cartopy.feature.Scaler`, optional
            A named resolution to use from the Natural Earth
            dataset. Currently can be one of "auto" (default), "110m", "50m",
            and "10m", or a Scaler object.
        proj : {None, 'aitoff', 'hammer', 'lambert', 'mollweide', 'polar', 'rectilinear', str}, optional
            The projection type of the `~.axes.Axes`. *str* is the name of
            a custom projection, see `~matplotlib.projections`. The default
            None results in a 'rectilinear' projection.
            Default is cartopy.crs.PlateCarree().
        dpi : float, default: :rc:`figure.dpi`
            The resolution of the figure in dots-per-inch.
        figsize : tuple, optional
            A tuple (width, height) of the figure in inches.   
        rivers : bool, optional
            Whether to plot rivers. Default is False
        """        
        
        O = np.floor(np.log10(len(self.ds.time)))
        
        if t is None:
            if O >= 3:
                ds = self.ds.isel(time=slice(None,None,int(len(self.ds.time)/10**(O-2))))
            else:
                ds = self.ds
        else:
            ds = self.ds.isel(time=t)
        
        
        if c == 'age': 
            c = ds.age_seconds/60/60/24
            #cmap = 'rainbow' #'copper'
            cbar_label = 'elapsed days'
        elif c == 'z':
            c = ds.z
            #cmap = 'viridis'
            cbar_label = 'depth [m]'
        elif c == 'marker':
            c = ds.origin_marker
            cbar_label = 'origin marker (#rep)'
        
        SMALL_SIZE = 8
        MEDIUM_SIZE = 10
        BIGGER_SIZE = 16
        
        fig = plt.figure(figsize=figsize)
        ax = plt.axes(projection=proj)
        ax.coastlines(coastres, zorder=11, color='k', linewidth=.5)
        ax.add_feature(cartopy.feature.LAND, facecolor='0.9', zorder=10) #'#FFE9B5'
        ax.add_feature(cartopy.feature.BORDERS, zorder=10, linewidth=.5, linestyle=':')
        if rivers is True:
            ax.add_feature(cartopy.feature.RIVERS, zorder=12)
        gl = ax.gridlines(draw_labels=True, dms=False, x_inline=False, y_inline=False, linewidth=.5, color='gray', linestyle='--')
        gl.top_labels = False
        gl.right_labels = False    

        im = ax.scatter(ds.lon, ds.lat, s=s, c=c, cmap=cmap, alpha=alpha)
        cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label(cbar_label, rotation=90)
        
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        
        ax.set_title('Particle trajectories over time', fontsize=10);

        plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
        plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
        
        if save_to is not None:
            plt.savefig(save_to, dpi=160, bbox_inches='tight')
        
        return fig, ax
    
    
    def animate(self):
        """
        WIP. Histogram animation using xmovie
        
        """
        
        pass


    
def check_particle_file(path_to_file):
    ds = xr.open_dataset(path_to_file)
    vars_to_check = ['x_sea_water_velocity', 'y_sea_water_velocity', 'x_wind', 'y_wind']
    for i in vars_to_check:
        if np.all(ds[i].load() == 0):
            print(f'ATTENTION: all 0 values detected for variable {i}.')
        else:
            print(f'all good: variable {i} has non-zero values.')
