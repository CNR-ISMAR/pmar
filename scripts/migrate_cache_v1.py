#!/usr/bin/env python
"""
Migrate PMARCache entries created before the cache.py hashing rewrite.

The rewrite (see pmar/cache.py `_lookup`) changed the MD5 hash input from
`str(sorted(kwargs.items()))` to `json.dumps({'version': version, **cache_key}, sort_keys=True,
default=str)`, and `particle_cache` gained a new `seed_from_file` parameter that is now part of the
hashed cache_key. Both changes mean a cache file created before the rewrite now has the "wrong" name
for the same logical parameters, so the new code can't find it and treats it as a miss.

This script recomputes the new-style hash for each existing `<hash>.<ext>_metadata` sidecar (reading
the original cache_key straight from the sidecar, no need to know original Python types since the
hash is round-tripped through json.dumps/json.load either way) and renames the data file + sidecar
to match, so old runs become cache hits again.

Usage:
    python scripts/migrate_cache_v1.py CACHEDIR [CACHEDIR ...]           # dry run
    python scripts/migrate_cache_v1.py CACHEDIR [CACHEDIR ...] --apply   # actually rename

Pass every cache directory you want migrated: the particle cache (basedir/cachedir) and each PPI
raster cache (basedir/ppi-<use_label>) are separate directories.
"""
import argparse
import hashlib
import json
from pathlib import Path

# Mirrors the current particle_cache/raster_cache signatures in pmar/cache.py (minus self/version).
EXPECTED_KEYS = {
    'nc': {
        'context', 'pressure', 'chemical_compound', 'seeding_shapefile', 'pnum', 'start_time',
        'seed_time', 'duration_days', 'seed_within_bounds', 'seed_from_file', 'seeding_radius',
        'beaching', 'z', 'tstep', 'hdiff', 'termvel', 'stokes_drift', 'seeding_id',
    },
    'tif': {
        'context', 'pressure', 'chemical_compound', 'seeding_shapefile', 'seeding_radius', 'res',
        'pnum', 'ptot', 'duration', 'start_time', 'seedings', 'seeding_id', 'tshift', 'use',
        'use_label', 'emission', 'decay_coef', 'study_area',
    },
}


def new_hash_for(cache_key, version=1):
    """Recompute the current PMARCache._lookup MD5 hash for a given cache_key/version."""
    payload = {'version': version, **cache_key}
    return hashlib.md5(json.dumps(payload, sort_keys=True, default=str).encode('utf-8')).hexdigest()


def plan_migration(cachedir):
    """
    Scan `cachedir` for legacy `<hash>.<ext>_metadata` sidecars and work out the rename each one
    needs under the new hashing scheme.

    Returns a list of (old_data_path, new_data_path, old_metadata_path, new_metadata_path) tuples
    for entries that need renaming. Entries whose new hash already matches the current filename, or
    whose metadata doesn't match a known cache_key shape, are skipped (the latter with a warning).
    """
    renames = []
    for metadata_path in sorted(Path(cachedir).glob('*_metadata')):
        data_path = Path(str(metadata_path)[: -len('_metadata')])
        ext = data_path.suffix.lstrip('.')
        if ext not in EXPECTED_KEYS:
            print(f"WARNING: skipping {metadata_path.name}: unrecognized extension '{ext}'")
            continue

        cache_key = json.loads(metadata_path.read_text())

        if ext == 'nc' and 'seed_from_file' not in cache_key:
            cache_key['seed_from_file'] = None

        if set(cache_key) != EXPECTED_KEYS[ext]:
            missing = EXPECTED_KEYS[ext] - set(cache_key)
            extra = set(cache_key) - EXPECTED_KEYS[ext]
            print(
                f"WARNING: skipping {metadata_path.name}: cache_key doesn't match current "
                f"particle_cache/raster_cache signature (missing={sorted(missing)}, extra={sorted(extra)})"
            )
            continue

        new_hash = new_hash_for(cache_key)
        if new_hash == data_path.stem:
            continue  # already migrated / already matches current scheme

        new_data_path = data_path.with_stem(new_hash)
        new_metadata_path = Path(str(new_data_path) + '_metadata')
        renames.append((data_path, new_data_path, metadata_path, new_metadata_path))

    return renames


def apply_migration(renames):
    """Rename each (old_data, new_data, old_metadata, new_metadata) tuple, skipping collisions."""
    for old_data, new_data, old_metadata, new_metadata in renames:
        if new_data.exists() or new_metadata.exists():
            print(f"WARNING: skipping {old_data.name}: destination {new_data.name} already exists")
            continue
        old_data.rename(new_data)
        old_metadata.rename(new_metadata)
        print(f"renamed {old_data.name} -> {new_data.name}")


def main():
    """CLI entry point: parse args, plan the rename for each given cache dir, and apply it if --apply was passed."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('cachedirs', nargs='+', type=Path, help='cache directories to migrate')
    parser.add_argument('--apply', action='store_true', help='actually rename files (default: dry run)')
    args = parser.parse_args()

    for cachedir in args.cachedirs:
        print(f"== {cachedir} ==")
        renames = plan_migration(cachedir)
        if not renames:
            print("nothing to migrate")
            continue
        for old_data, new_data, _, _ in renames:
            print(f"{'would rename' if not args.apply else 'renaming'} {old_data.name} -> {new_data.name}")
        if args.apply:
            apply_migration(renames)
        else:
            print(f"({len(renames)} file(s) pending; re-run with --apply to rename)")


if __name__ == '__main__':
    main()
