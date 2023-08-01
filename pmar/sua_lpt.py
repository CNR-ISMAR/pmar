import logging
import numpy as np
import xarray as xr
from datetime import datetime, timedelta
import time as T
import warnings
from pmar.lpt import LagrangianDispersion
from SALib.sample import saltelli
from SALib.analyze import sobol
from joblib import Parallel, delayed
from copy import deepcopy

class PMARCaseStudy(object):
    def __init__(self, context, pnum=20000, duration=30, tstep=timedelta(hours=6), particle_status='all', poly_path=None):
        self.pnum = pnum
        self.duration = duration
        self.tstep = tstep
        self.context = context
        self.particle_status = particle_status
        self.runtypelevel = None
        self.poly_path = poly_path
        pass

    def get_main_output(self):
        return self.lpt.raster.r0
    
    def get_SUA_target(self):
        return self.lpt.raster.r0.sum()
    
    def run(self, runtypelevel=3):
        self.lpt = LagrangianDispersion(context=self.context, poly_path=self.poly_path)        
        self.runtypelevel=runtypelevel
        self.lpt.run(pnum=self.pnum, duration_days=self.duration, tstep=self.tstep, particle_status=self.particle_status)
        pass


class RunningStats2D:
    """
    This is a class for online computation of mean, variance and convergence arrays.
    """
    def __init__(self, percentiles=None):
        """
        :param percentiles: list-like (list, tuple, ecc) of percentile threshold values. E.g. [25, 50, 75].
        """
        self.n = 0
        self.old_m = None
        self.new_m = None
        self.old_s = None
        self.new_s = None
        self.initialized = False

        self.percentiles = percentiles
        self._convergence_arrays = []

    def clear(self):
        self.n = 0

    def push(self, x):
        self.n += 1

        # mean and variances
        if self.n == 1:
            self.old_m = self.new_m = x
            self.old_s = np.zeros_like(x)
            if self.percentiles is not None:
                for p in self.percentiles:
                    self._convergence_arrays.append(np.zeros_like(x))
        else:
            self.new_m = self.old_m + (x - self.old_m) / self.n
            self.new_s = self.old_s + (x - self.old_m) * (x - self.new_m)

            self.old_m = self.new_m
            self.old_s = self.new_s

        if self.percentiles is not None:
            pvals = np.percentile(np.ndarray.flatten(x[~x.mask]),
                                  self.percentiles)
            logger.debug('min={} max={} pvals={}'.format(x.min(), x.max(), pvals))
            for i, pval in enumerate(pvals):
                r = x.copy()
                r[x<pval] = 0
                r[x>=pval] = 1
                self._convergence_arrays[i] += r

    @property
    def mean(self):
        return self.new_m if self.n else 0.0

    @property
    def variance(self):
        return self.new_s / (self.n - 1) if self.n > 1 else 0.0

    @property
    def std(self):
        return np.sqrt(self.variance)

    def convergence_arrays(self):
        return [m / self.n for m in self._convergence_arrays]
    
    
class CaseStudySUA(object):
    """
    This is a base class for support Sensitivity and Uncertainty Analysis.
    Child classes have to implement "set_problem" and "set_params" methods.
    """
    def __init__(self, module_cs, nparams=40, bygroup=True, calc_second_order=False,
                 kwargs_run={}):
        self.module_cs = module_cs
        self.kwargs_run = kwargs_run
        # check runtypelevel
        #if self.module_cs.runtypelevel is None or self.module_cs.runtypelevel < 3:
         #   raise Exception("Case Study doesn't have a sufficient runlevel")
          #  pass
        # TODO: casestudy should store info on input parameters (e.g. uses,
        #  pres, envs) in order to allow multiple runs
        self.problem = {
            'num_vars': 0,
            'names': [],  # ['mscf'],  # , 'rfunc_b', 'rfuncb_m'],
            'bounds': [],  # [[0, 1]],  # , [1, 30], [0.3, 0.7]],
            'dists': [],  # ['unif'],  # 'unif', 'unif'],
            # 'groups': [],  # ['mscf'],  # 'rfunc', 'rfunc']
        }

        if bygroup:
            self.problem['groups'] = []

        self.var_index = []
        self.model_output_stats = None
        self.target_values = None
        self.nparams = nparams
        self.bygroup = bygroup
        self.calc_second_order = calc_second_order

    def set_problem(self):
        """This is override by child classes"""
        pass

    def set_params(self, params):
        """This is override by child classes"""
        pass

    def add_problem_var(self, var_idx, bound, dist, group=None):
        name = ' '.join(var_idx)
        self.problem['num_vars'] += 1
        self.problem['names'].append(name)
        self.problem['bounds'].append(bound)
        self.problem['dists'].append(dist)
        if group is not None and self.bygroup:
            self.problem['groups'].append(group)
        elif group is None and self.bygroup:
            self.problem['groups'].append(name)
        else:
            pass

        self.var_index.append(var_idx)
        #logger.debug('add problem var {} {} {} {}'.format(name, bound, dist, group))
        
    def sample(self, nruns, calc_second_order=False):
        return saltelli.sample(self.problem, nruns,
                               calc_second_order=calc_second_order)

    def runall(self, nruns, calc_second_order=False, njobs=1):
        self.set_problem()
        samples = self.sample(nruns, calc_second_order)
        #logger.debug('Samples={} calc_second_order={}'.format(len(samples), calc_second_order))
        model_output_stats = RunningStats2D(percentiles=None)
        # TODO: add parallelization
        def _single_run(i, params):
            module_cs = deepcopy(self.module_cs)
            self.set_params(params, module_cs=module_cs)
            module_cs.run(runtypelevel=0, **self.kwargs_run)
            model_output = module_cs.get_main_output()
            model_output_stats.push(model_output)
            target_value = module_cs.get_SUA_target()
            #logger.debug('run {} target={}'.format(i, target_value))
            return target_value
            
        with Parallel(n_jobs=njobs, backend='threading', require='sharedmem') as parallel:
            target_values = parallel(delayed(_single_run)(i, params) for i, params in enumerate(samples))

        # for i, params in enumerate(samples):
        #     self.set_params(params)
        #     self.module_cs.run(runtypelevel=0, **self.kwargs_run)
        #     model_output = self.module_cs.get_main_output()
        #     model_output_stats.push(model_output)
        #     target_value = self.module_cs.get_SUA_target()
        #     target_values.append(target_value)
        #     logger.debug('run {} target={}'.format(i, target_value))
        self.model_output_stats = model_output_stats
        self.target_values = np.array(target_values)

    def analyze(self, calc_second_order=False):
        sa_results = sobol.analyze(self.problem,
                                   self.target_values,
                                   calc_second_order=calc_second_order)
        return sa_results

    @property
    def mean(self):
        return self.model_output_stats.mean

    @property
    def std(self):
        return self.model_output_stats.std

    @property
    def cv(self):
        mean = self.mean
        std = self.std
        return std/mean

class PMARCaseStudySUA(CaseStudySUA):
    def set_problem(self):
        nparams = self.nparams # ?
        module_cs = self.module_cs # ? 
        self.normalize_distance = None # ?
        
        # the problem is defined by a number of vairables, names of the variables, and their bounds.
        # D = 2 # number of variables
        
        self.add_problem_var(['time_var', 'duration', 'duration'], # length of the simulation in days # ['duration_days', 'duration', label], "label" is for tools4msp like ENV, USEPRE ecc
                     [50, 90, 0.5], # bounds
                     'triang', # distribution
                     #group_label if self.bygroup else None
                     )

        self.add_problem_var(['time_var', 'tstep', 'tstep'], # length of timestep in hours
                     [6, 24], # bounds
                     'unif', # distribution
                     #group_label if self.bygroup else None
                     )
        
        self.add_problem_var(['hdiff', 'hdiff', 'hdiff'], # length of timestep in hours
                     [5, 15, 0.5], # bounds
                     'triang', # distribution
                     #group_label if self.bygroup else None
                     )
        
        #self.add_problem_var(['termvel', 'termvel', 'termvel'], # length of timestep in hours
         #            [1e-3, 1e-2], # bounds
          #           'unif', # distribution
           #          #group_label if self.bygroup else None
            #         )        
        
        self.add_problem_var(['decay_rate', 'decay_rate', 'decay_rate'], # length of timestep in hours
                     [0.1, 1, 0.5], # bounds
                     'triang', # distribution
                     #group_label if self.bygroup else None
                     )
        
    
    def set_params(self, params, module_cs=None):
        if module_cs is None:
            module_cs = self.module_cs
            
        module_cs.duration = int(np.round(params[0]))
        module_cs.tstep = timedelta(hours=int(np.round(params[1])))