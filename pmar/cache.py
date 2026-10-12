from pathlib import Path
import hashlib
import json
import logging
logger = logging.getLogger("PMARCache")

class PMARCache(object):
    """
    On-disk cache for expensive PMAR outputs (OpenDrift trajectory files, PPI rasters),
    keyed by a hash of the parameters that determine the output. See _lookup for the
    hashing/versioning scheme, and particle_cache/raster_cache for the two supported kinds
    of cached artifact.

    Cache keys are built from an explicit, permanently-named field set in particle_cache/
    raster_cache -- never auto-derived from locals() -- so that renaming a PMAR method's
    own Python parameter (e.g. duration_days -> duration) never changes the hash. Only a
    deliberate edit to the field set itself (paired with bumping `version`) should do that.
    """

    def __init__(self, cachedir): # cachedir è una sottodirectory di basedir
        """
        Parameters
        ----------
        cachedir : str or pathlib.Path
            Directory to store cached files (and their metadata sidecars) in; created if it
            doesn't already exist.
        """
        self.cachedir = Path(cachedir)
        self.cachedir.mkdir(parents=True, exist_ok=True)

    def _lookup(self, extension, version, cache_key):
        """
        Shared cache-lookup core used by particle_cache/raster_cache.

        Hashes `version` together with `cache_key` (via json.dumps(..., sort_keys=True,
        default=str)) into an MD5 key, and derives a cache file path from it under
        self.cachedir. Does not write metadata itself -- callers write it explicitly via
        write_metadata(), only once the real artifact has actually been created, so an
        interrupted run never leaves an orphaned metadata-only sidecar behind.

        `version` is not part of the caller-visible cache_key; it is a manual escape hatch: bump the
        `version` default on a particle_cache/raster_cache call site whenever the code that produces
        the cached data (or the cache_key field set itself) changes in a way that would make
        previously-cached files (keyed on the same cache_key) incompatible with what a fresh call
        would now produce. Since version feeds into the hash, bumping it changes the derived path,
        so old entries are naturally orphaned rather than silently reused.

        Args:
            extension: file extension for the cached artifact (e.g. 'nc', 'tif'), without a leading dot.
            version: integer (or any JSON/str-able value) folded into the hash; see above.
            cache_key: dict of the parameters that uniquely determine the cached output.

        Returns:
            (path, exists): pathlib.Path to the cache entry, and whether it already exists on disk.
        """
        payload = {'version': version, **cache_key}
        key_hash = hashlib.md5(json.dumps(payload, sort_keys=True, default=str).encode('utf-8')).hexdigest()
        path = self.cachedir / f"{key_hash}.{extension}"
        return path, path.exists()

    def write_metadata(self, path, cache_key):
        """
        Write `cache_key` as JSON sidecar metadata next to `path` (at f"{path}_metadata"), so the
        parameters that produced a given cache file can be inspected later without decoding the hash.

        Callers should call this themselves exactly once, right after the real artifact at `path`
        has been successfully written on a cache miss -- never before, and never on a cache hit
        (the sidecar from the original write is still valid).
        """
        with open(str(path) + '_metadata', 'w') as fi:
            json.dump(cache_key, fi, default=str)

    def particle_cache(self, reader_dataset_ids, pressure, chemical_compound, seeding_shapefile, pnum, start_time, seed_time, duration, seed_within_bounds, seed_from_file, seeding_radius, beaching, z, tstep, hdiff, termvel, stokes_drift, seeding_id, version=2):
        """
        Cache lookup for OpenDrift particle trajectory output (a .nc file).

        Returns (path, exists, cache_key): the cache file path for this exact combination of
        simulation parameters, whether it already exists, and the cache_key dict itself (pass it to
        write_metadata() after writing the artifact on a miss). Callers should run the OpenDrift
        simulation and write its output to `path` when `exists` is False, and read directly from
        `path` when True.

        Parameters
        ----------
        reader_dataset_ids : list of str
            The actual Copernicus Marine dataset ids used to force this simulation (e.g.
            sorted(self.spatial_domain['readers'].values())) -- by value, not by whatever
            descriptive dict/key-names they happen to be organized under, so renaming a
            reader key or rewording SPATIAL_DOMAINS' 'desc' never invalidates this cache.
            self.spatial_domain must already be resolved (not the 'auto' sentinel) before
            calling this -- see PMAR._resolve_spatial_domain().

        Bump `version` when the OpenDrift run logic, or this cache_key's field set, changes in a
        way that makes previously-cached trajectory files incompatible with what today's code
        would produce for the same parameters.
        """
        if not isinstance(reader_dataset_ids, (list, tuple)) or not all(isinstance(x, str) for x in reader_dataset_ids):
            raise TypeError(f"reader_dataset_ids must be a list of Copernicus dataset-id strings (did self.spatial_domain get resolved before this call?); got {reader_dataset_ids!r}")

        cache_key = {
            'reader_dataset_ids': sorted(reader_dataset_ids),
            'pressure': pressure,
            'chemical_compound': chemical_compound,
            'seeding_shapefile': seeding_shapefile,
            'pnum': pnum,
            'start_time': start_time,
            'seed_time': seed_time,
            'duration': duration,
            'seed_within_bounds': seed_within_bounds,
            'seed_from_file': seed_from_file,
            'seeding_radius': seeding_radius,
            'beaching': beaching,
            'z': z,
            'tstep': tstep,
            'hdiff': hdiff,
            'termvel': termvel,
            'stokes_drift': stokes_drift,
            'seeding_id': seeding_id,
        }
        path, exists = self._lookup('nc', version, cache_key)
        logger.info(f'particle cache = {cache_key}')
        return path, exists, cache_key

    def raster_cache(self, reader_dataset_ids, pressure, chemical_compound, seeding_shapefile, seeding_radius, res, pnum, ptot, duration, start_time, seedings, seeding_id, tshift, use, use_label, emission, decay_coef, study_area, version=2):
        """
        Cache lookup for rasterized PPI output (a .tif file).

        Returns (path, exists, cache_key): the cache file path for this exact combination of PPI
        parameters, whether it already exists, and the cache_key dict itself (pass it to
        write_metadata() after writing the artifact on a miss). Callers should compute the raster
        and write it to `path` when `exists` is False, and read directly from `path` (e.g. via
        rioxarray.open_rasterio) when True.

        Parameters
        ----------
        reader_dataset_ids : list of str
            See particle_cache(); same by-value convention.

        Bump `version` when the PPI/histogram computation, or this cache_key's field set, changes
        in a way that makes previously-cached raster files incompatible with what today's code
        would produce for the same parameters.
        """
        if not isinstance(reader_dataset_ids, (list, tuple)) or not all(isinstance(x, str) for x in reader_dataset_ids):
            raise TypeError(f"reader_dataset_ids must be a list of Copernicus dataset-id strings (did self.spatial_domain get resolved before this call?); got {reader_dataset_ids!r}")

        cache_key = {
            'reader_dataset_ids': sorted(reader_dataset_ids),
            'pressure': pressure,
            'chemical_compound': chemical_compound,
            'seeding_shapefile': seeding_shapefile,
            'seeding_radius': seeding_radius,
            'res': res,
            'pnum': pnum,
            'ptot': ptot,
            'duration': duration,
            'start_time': start_time,
            'seedings': seedings,
            'seeding_id': seeding_id,
            'tshift': tshift,
            'use': use,
            'use_label': use_label,
            'emission': emission,
            'decay_coef': decay_coef,
            'study_area': study_area,
        }
        path, exists = self._lookup('tif', version, cache_key)
        logger.info(f'raster cache = {cache_key}')
        return path, exists, cache_key
