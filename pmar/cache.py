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

        Hashes `version` together with `cache_key` (via json.dumps(..., sort_keys=True, default=str),
        which is more robust than the old str(sorted(kwargs.items())) approach) into an MD5 key, and
        derives a cache file path from it under self.cachedir.

        `version` is not part of the caller-visible cache_key; it is a manual escape hatch: bump the
        `version` default on a particle_cache/raster_cache call site whenever the code that produces
        the cached data changes in a way that would make previously-cached files (keyed on the same
        cache_key) incompatible with what a fresh call would now produce. Since version feeds into the
        hash, bumping it changes the derived path, so old entries are naturally orphaned rather than
        silently reused.

        Metadata (see _write_metadata) is written only on a cache miss (path does not yet exist) so
        that a cache hit never touches disk beyond the existence check.

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
        exists = path.exists()
        if not exists:
            self._write_metadata(path, cache_key)
        return path, exists

    def _write_metadata(self, path, cache_key):
        """
        Write `cache_key` as JSON sidecar metadata next to `path` (at f"{path}_metadata"), so the
        parameters that produced a given cache file can be inspected later without decoding the hash.
        Called by _lookup only when creating a new cache entry, never on a hit.
        """
        with open(str(path) + '_metadata', 'w') as fi:
            json.dump(cache_key, fi, default=str)

    def particle_cache(self, context, pressure, chemical_compound, seeding_shapefile, pnum, start_time, seed_time, duration_days, seed_within_bounds, seed_from_file, seeding_radius, beaching, z, tstep, hdiff, termvel, stokes_drift, seeding_id, version=1):
        """
        Cache lookup for OpenDrift particle trajectory output (a .nc file).

        Returns (path, exists): the cache file path for this exact combination of simulation
        parameters, and whether it already exists. Callers should run the OpenDrift simulation and
        write its output to `path` when `exists` is False, and read directly from `path` when True.

        Bump `version` when the OpenDrift run logic changes in a way that makes previously-cached
        trajectory files incompatible with what today's code would produce for the same parameters.
        """
        cache_key = {k: v for k, v in locals().items() if k not in ('self', 'version')}
        path, exists = self._lookup('nc', version, cache_key)
        logger.info(f'particle cache = {cache_key}')
        return path, exists

    def raster_cache(self, context, pressure, chemical_compound, seeding_shapefile, seeding_radius, res, pnum, ptot, duration, start_time, seedings, seeding_id, tshift, use, use_label, emission, decay_coef, study_area, version=1):
        """
        Cache lookup for rasterized PPI output (a .tif file).

        Returns (path, exists): the cache file path for this exact combination of PPI parameters, and
        whether it already exists. Callers should compute the raster and write it to `path` when
        `exists` is False, and read directly from `path` (e.g. via rioxarray.open_rasterio) when True.

        Bump `version` when the PPI/histogram computation changes in a way that makes previously-cached
        raster files incompatible with what today's code would produce for the same parameters.
        """
        cache_key = {k: v for k, v in locals().items() if k not in ('self', 'version')}
        path, exists = self._lookup('tif', version, cache_key)
        logger.info(f'raster cache = {cache_key}')
        return path, exists