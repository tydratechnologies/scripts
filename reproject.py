#!/usr/bin/env python3
"""Mosaic ordered BAG/TIF files, using gdalinfo to identify horizontal UTM CRS."""
import json
import re
import subprocess
from collections import Counter
from pathlib import Path
from osgeo import gdal

gdal.UseExceptions()
ROOT = Path.cwd()
VRT_DIR = ROOT / 'warped_vrts'


def projection(path):
    """Read gdalinfo's CRS WKT and identify WGS84/NAD83 UTM zone, north."""
    result = subprocess.run(['gdalinfo', '-json', str(path)], capture_output=True, text=True, check=True)
    info = json.loads(result.stdout)
    wkt = info.get('coordinateSystem', {}).get('wkt', '')
    if not wkt:
        raise ValueError(f'No projection in {path}')
    if wkt.lstrip().startswith(('GEOGCRS[', 'GEOGCS[')):
        if re.search(r'WGS[_ ]?84|WGS[_ ]?1984|World Geodetic System 1984', wkt, re.I):
            return 'EPSG:4324', wkt
    # BAG compound CRS contains horizontal UTM parameters inside its WKT.
    zone_match = re.search(r'UTM[ _-]*(?:zone[ _-]*)?(\d{1,2})\s*([NS])?', wkt, re.I)
    if not zone_match:
        raise ValueError(f'Not a recognized UTM CRS: {path.name}\n{wkt[:350]}')
    zone = int(zone_match.group(1))
    hemisphere = (zone_match.group(2) or 'N').upper()
    if not 1 <= zone <= 60:
        raise ValueError(f'Invalid UTM zone in {path.name}')
    # Inspect the horizontal datum, not the vertical MLLW metadata.
    # Do not guess when the source datum is ambiguous.
    horizontal_part = wkt.split('VERTCRS[')[0].split('VERT_CS[')[0]
    if re.search(r'NAD[_ ]?83|North American Datum 1983', horizontal_part, re.I):
        if hemisphere != 'N':
            raise ValueError(f'Unsupported NAD83 southern UTM: {path.name}')
        epsg = 26900 + zone
    elif re.search(r'WGS[_ ]?84|WGS[_ ]?1984|World Geodetic System 1984', horizontal_part, re.I):
        epsg = (32600 if hemisphere == 'N' else 32700) + zone
    else:
        raise ValueError(f'Unrecognized horizontal datum: {path.name}\n{wkt[:350]}')
    return f'EPSG:{epsg}', wkt


def main():
    names = (ROOT / 'tiles.txt').read_text(encoding='utf-8-sig').splitlines()
    tiles = []
    for name in names:
        name = name.strip().strip('"').strip("'")
        if not name or name.startswith('#'):
            continue
        path = Path(name)
        path = (path if path.is_absolute() else ROOT / path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        tiles.append(path)
    if not tiles:
        raise ValueError('tiles.txt is empty')

    print(f'Inspecting {len(tiles)} rasters with gdalinfo...', flush=True)
    records = []
    for i, path in enumerate(tiles, 1):
        crs, wkt = projection(path)
        records.append((path, crs, wkt))
        print(f'[{i}/{len(tiles)}] {path.name}: {crs}', flush=True)

    counts = Counter(crs for _, crs, _ in records)
    print('\nHorizontal CRS counts:')
    for crs, count in counts.most_common():
        print(f'  {crs}: {count}')
    target = counts.most_common(1)[0][0]
    print(f'\nTarget: {target}', flush=True)
    VRT_DIR.mkdir(exist_ok=True)
    manifest = []
    vrt_paths = []
    for i, (path, source_crs, original_wkt) in enumerate(records, 1):
        dest = VRT_DIR / f'{i:05d}_{path.stem}.vrt'
        print(f'[{i}/{len(records)}] {"WARP" if source_crs != target else "NORMALIZE"} {path.name}', flush=True)
        # Always supply a horizontal-only CRS so PROJ never attempts to
        # transform the compound MLLW vertical coordinate system.
        opts = gdal.WarpOptions(format='VRT', srcSRS=source_crs,
                                dstSRS=target, resampleAlg='bilinear')
        try:
            ds = gdal.Warp(str(dest), str(path), options=opts)
            if ds is None:
                raise RuntimeError('gdal.Warp returned None')
            ds = None
        except Exception as exc:
            raise RuntimeError(f'Failed at tile {i}: {path.name}: {exc}') from exc
        vrt_paths.append(dest.resolve())
        manifest.append({'position': i, 'source': str(path), 'source_horizontal_crs': source_crs,
                         'original_crs_wkt': original_wkt, 'vrt': str(dest.resolve())})

    (ROOT / 'warped_tiles.txt').write_text(''.join(str(p) + '\n' for p in vrt_paths))
    (ROOT / 'warp_manifest.json').write_text(json.dumps({'target_crs': target, 'inputs': manifest}, indent=2))
    subprocess.run(['gdalbuildvrt', '-overwrite', '-b', '1', '-r', 'bilinear',
                    '-resolution', 'highest', '-input_file_list', 'warped_tiles.txt',
                    'mosaic.vrt'], check=True)
    print('\nDone: mosaic.vrt (tiles.txt order preserved)')


if __name__ == '__main__':
    main()
