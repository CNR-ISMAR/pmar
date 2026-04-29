import logging
import rioxarray as rxr
import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
from pmar.pmar import PMAR
from datetime import datetime, timedelta, date
from shapely.geometry import Point, Polygon
from opendrift.models.oceandrift import OceanDrift
from opendrift.models.openoil import OpenOil
from opendrift.models.plastdrift import PlastDrift
from opendrift.models.chemicaldrift import ChemicalDrift
from opendrift.models.larvalfish import LarvalFish
#import copernicusmarine
from opendrift.readers.reader_copernicusmarine import Reader
from opendrift.readers.unstructured import shyfem
from opendrift.readers.reader_netCDF_CF_generic import Reader as GenericReader
import opendrift
#from pydap.client import open_url
#from pydap.cas.get_cookies import setup_session
import time as T
from xhistogram.xarray import histogram
import cartopy.crs as ccrs
import cartopy
from matplotlib.ticker import LogFormatter, PercentFormatter
from matplotlib import ticker
import matplotlib.pyplot as plt
import matplotlib.colors as col
import os
from opendrift.readers import reader_shape
import cartopy.io.shapereader as shpreader
#import netrc
#import random
#import rasterio
#import tempfile
#from netCDF4 import Dataset
from pathlib import Path
#from flox.xarray import xarray_reduce # for xarray grouping over multiple variables
os.environ['PROJ_LIB'] = '/var/miniconda3/envs/opendrift/share/proj/'
from functools import partial
from rasterio.enums import Resampling
#import glob
#from copy import deepcopy
#from geocube.api.core import make_geocube
import networkx as nx
from pmar.cache import PMARCache
from pmar.utils import traj_distinct, check_particle_file, get_marine_polygon, make_poly, harmonize_use, make_grid, make_vector_grid
#from cmap import Colormap
from pmar.plotting import plot_map
from scipy.sparse import coo_array
#tol_inc = Colormap('tol:incandescent').to_mpl()
#tol_muted = Colormap('tol:muted').to_mpl()
#rainbow = Colormap('tol:rainbow_PuRd').to_mpl()
#tol_div = Colormap('tol:BuRd').to_mpl()
#from xgcm import Grid
from scipy.sparse import diags
from scipy.sparse.linalg import matrix_power
from scipy.io import mmwrite
logger = logging.getLogger('pmar')
#logger.setLevel(logging.DEBUG)
# sh = logging.StreamHandler()
# sf = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
# sh.setFormatter(sf)
# logger.addHandler(sh)

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
        
        
class PMARmulti(PMAR): 
    """
    Developed at CNR ISMAR in Venice.

    ...

    Attributes
    ----------


    Methods
    -------


    """
        


    def get_trajectories(self, pnum, start_time, seedings=1, tshift=0, end_time=None, duration_days=30, study_area=None, seeding_shapefile = None, seed_within_bounds=None, z=-0.5, tstep=timedelta(hours=1), hdiff=10, termvel=0, crs='4326', seeding_radius=0, beaching=False, stokes_drift=False, pnum_per_shape=False):

        self.seedings = seedings
        self.tshift = tshift
        self.seeding_radius = seeding_radius
        
        for n in range(0, seedings):
    
            self.seeding_id = n
            
            start_time_dt = datetime.strptime(start_time, '%Y-%m-%d')+timedelta(days=tshift)*n #convert start_time into datetime to add tshift
            rep_start_time = start_time_dt.strftime("%Y-%m-%d") # bring back to string to feed to opendrift
            if end_time is not None:
                rep_duration = (datetime.strptime(end_time, '%Y-%m-%d') - start_time_dt).days
            else:
                rep_duration = duration_days
            
            logger.info(f'Starting rep #{self.seeding_id} with start_time = {rep_start_time}, end_time: {end_time}, and duration: {rep_duration}')
            
            #pnum = int(np.round(ptot/seedings)) #  ptot should be split among the seedings
            
            super().get_trajectories(pnum=pnum, start_time=rep_start_time, duration_days=rep_duration, study_area=study_area, seeding_shapefile=seeding_shapefile, seed_within_bounds=seed_within_bounds, z=z, tstep=tstep, hdiff=hdiff, termvel=termvel, crs=crs, seeding_radius=seeding_radius, beaching=beaching, stokes_drift=stokes_drift, pnum_per_shape=pnum_per_shape)
            logger.info(f'Done with rep #{n}.')
            self.particle_list.append(self.particle_path)

        self.import_trajectories(self.particle_list)

    def get_ppi(self, res, use=None, use_label=None, emission=None, study_area=None, decay_coef=0): 
        
        ppi_output_dir = Path(self.basedir) / f'ppi-{use_label}'

        self.ppi_cache = PMARCache(ppi_output_dir)
             
        ppi_path, ppi_exists = self.ppi_cache.raster_cache(context=self.context, pressure=self.pressure, chemical_compound=self.chemical_compound, seeding_shapefile=self.seeding_shapefile, seeding_radius=self.seeding_radius, res=res, pnum=None, ptot=self.pnum, start_time=self.start_time, duration=self.duration, seedings=self.seedings, seeding_id=self.seeding_id, tshift=self.tshift, use=use, use_label=use_label, emission=emission, decay_coef=decay_coef, study_area=study_area)#, aggregate=aggregate, depth_layer=depth_layer, z_bounds=z_bounds, particle_status=particle_status, traj_dens=traj_dens)
        
        self.output = xr.Dataset()
        self.grid = make_grid(res=res, study_area=study_area)
        
        if ppi_exists == True:
            logger.info('PPI with requested configurations found in cache.')
            self.output['ppi'] = rxr.open_rasterio(ppi_path)
        else:
        
            for pfile in self.particle_list:
                temp_self = PMAR()
                temp_self.__dict__ = self.__dict__
                
                temp_self.import_trajectories(pfile)
                if hasattr(self, 'use') and self.use is not None: # to avoid re-rasterizing use for every seeding
                    use = self.use
                temp_self.get_ppi(res=res, use=use, use_label=use_label, emission=emission, study_area=study_area, decay_coef=decay_coef)
                self.ppi_list.append(temp_self.ppi_path)
                if hasattr(temp_self, 'use') and temp_self.use is not None: # to avoid errors if ppi already exists
                    self.use = temp_self.use
                  #  self.use_label = temp_self.use_label
                del temp_self

            self.import_trajectories(self.particle_list)
            
            self.output['ppi'] = self.sum_seedings(self.ppi_list).squeeze().rio.reproject_match(self.grid.mask.rename({'y_c':'y','x_c':'x'})).rename({'x':'lon','y':'lat'})/len(self.ds.seeding)
            self.write_tiff(self.output['ppi'], path=ppi_path)
            self.ppi_path = ppi_path
        
    

    def sum_seedings(self, rep_path):
        '''
        Sum ppi rasters of single seedings, return summed raster. 
    
        Parameters 
        ----------
        rep_path : list
            list of paths to each rep 
        '''
        if self.seedings == 1:
            r1 = rxr.open_rasterio(rep_path)
        else:
            for idx, rep in enumerate(rep_path):
                if idx == 0:
                    r1 = r0 = rxr.open_rasterio(rep) 
                else:
                    r0 = r1
                    r1 = r0 + rxr.open_rasterio(rep)#.rio.reproject_match(r0, crs=r0.rio.crs, all_touched=True)
        
        return r1

    # TODO import trajectories from particle_list even if multippi already exists
    # can get particle_list from particle_cache
    def run(self, pnum, start_time, duration, res, seedings=1, tshift=0, end_time=None, tstep=timedelta(hours=1), study_area=None, seeding_shapefile=None, seed_within_bounds=None, z=-0.5, seeding_radius=0, pnum_per_shape=False, beaching=False, hdiff=10, termvel=0, stokes_drift=False, decay_coef=0, use=None, use_label=None, emission=None, output_dir=None, save_tiffs=False, thumbnail=None, crs='4326'):
        '''
        Compute trajectories and produce ppi raster over required number of seedings. 
    
        Parameters
        ----------
        pnum : int
    
        seedings : int
    
        tshift : int
    
        duration : int, optional
    
        start_time : string, optional
    
        res : float, optional

        study_area : list, optional
    
        use : string, optional

        use_label : string, optional
    
        decay_coef : float, optional
        
        '''

        self.seedings = seedings
        
        ppi_output_dir = Path(self.basedir) / f'ppi-{use_label}'

        self.ppi_cache = PMARCache(ppi_output_dir)
             
        ppi_path, ppi_exists = self.ppi_cache.raster_cache(context=self.context, pressure=self.pressure, chemical_compound=self.chemical_compound, seeding_shapefile=seeding_shapefile, seeding_radius=seeding_radius, res=res, pnum=None, ptot=pnum, start_time=start_time, duration=duration, seedings=seedings, seeding_id=self.seeding_id, tshift=tshift, use=use, use_label=use_label, emission=emission, decay_coef=decay_coef, study_area=study_area)#, aggregate=aggregate, depth_layer=depth_layer, z_bounds=z_bounds, particle_status=particle_status, traj_dens=traj_dens)
        
        self.output = xr.Dataset()
        
        if ppi_exists == True:
            logger.info('PPI with requested configurations found in cache.')
            self.output['ppi'] = rxr.open_rasterio(ppi_path)
        else:
            self.get_trajectories(pnum=pnum, start_time=start_time, seedings=seedings, tshift=tshift, end_time=end_time, duration_days=duration, study_area=study_area, seeding_shapefile=seeding_shapefile, seed_within_bounds=seed_within_bounds, z=z, tstep=tstep, hdiff=hdiff, termvel=termvel, crs=crs, seeding_radius=seeding_radius, beaching=beaching, stokes_drift=stokes_drift, pnum_per_shape=pnum_per_shape)
            self.get_ppi(res=res, use=use, use_label=use_label, emission=emission, study_area=study_area, decay_coef=decay_coef)
            
        pass


    def get_Pij(self, cm_grid=None, res=None, study_area=None, t0=0, t1=-1, filename=None):
        '''
        Average transition matrix 
        '''
        for i, file in enumerate(self.particle_list):
            logger.info(f'Calculating average CM out of {len(self.particle_list)} particle files.')
            ds = xr.open_dataset(file).isel(time=slice(t0,t1))
            temp_self = PMAR()
            temp_self.__dict__ = self.__dict__
            temp_self.import_trajectories(ds)
            if hasattr(self, 'use') and self.use is not None: # to avoid re-rasterizing use for every seeding
                temp_self.set_weights(use=self.use, res=res, study_area=study_area)
                weighted=True
            else:
                weighted=False
            temp_self.get_Pij(cm_grid=cm_grid, res=res, study_area=study_area, t0=t0, t1=t1, filename=filename, weighted=weighted)
            if i == 0:
                Pij = temp_self.CM/len(self.particle_list)
            else:
                Pij = Pij + temp_self.CM/len(self.particle_list)
            del temp_self
            logger.info(f'Done calculating CM for particle file {i}.')
        self.CM = Pij