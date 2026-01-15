#!/usr/bin/env python3

import argparse
import os
import shutil
import subprocess
import math

web_mercator_max = 20037508.342789244

class Colors:
    GREEN = '\033[92m'
    BLUE = '\033[94m'
    PURPLE = '\033[35m'
    ENDC = '\033[0m'

parser = argparse.ArgumentParser()

parser.add_argument(
    '-m_vrt', '--merged_vrt_path',
    type=str,
    required=True,
    help="Merged virtual mosaic"
)

parser.add_argument(
    '-s_vrt', '--slave_vrt_path',
    type=str,
    required=True,
help="Slave virtual mosaic"
)

parser.add_argument(
    '-z', '--zoom',
    type=str,
    required=True,
    help="Zoom levels"
)

parser.add_argument(
    '-dst_p', '--dst_pyramid_path',
    type=str,
    required=True,
    help="Destination tile pyramid"
)

args = parser.parse_args()

merged_vrt_path = args.merged_vrt_path
slave_vrt_path = args.slave_vrt_path
min_zoom = int(args.zoom.split('-')[0])
max_zoom = int(args.zoom.split('-')[1])
dst_pyramid_path = args.dst_pyramid_path
src_pyramid_path = 'temp'

def world_to_pixel(x_m, y_m, z):
    scale = (256 * (2 ** z)) / (web_mercator_max * 2.0)

    px = (x_m + web_mercator_max) * scale
    py = (web_mercator_max - y_m) * scale
    return px, py

def pixel_to_world(px, py, z):
    world_extent = web_mercator_max * 2.0
    scale = world_extent / (256 * (2 ** z))

    x_m = px * scale - web_mercator_max
    y_m = web_mercator_max - py * scale
    return x_m, y_m

def calculate_snapped_bbox(z):
    px_min, py_max = world_to_pixel(x_min, y_max, z)
    px_max, py_min = world_to_pixel(x_max, y_min, z)

    eps_px = 0.5

    tile_x_min = math.floor(px_min / 256)
    tile_x_max = math.floor((px_max - eps_px) / 256)
    tile_y_min = math.floor(py_max / 256)
    tile_y_max = math.floor((py_min - eps_px) / 256)

    snap_px_min = tile_x_min * 256
    snap_px_max = (tile_x_max + 1) * 256
    snap_py_max = tile_y_min * 256
    snap_py_min = (tile_y_max + 1) * 256

    print(Colors.BLUE + f"Snapped Pixel Box: {snap_px_min} xMin, {snap_py_min} yMin, {snap_px_max} xMax, {snap_py_max} yMax" + Colors.ENDC)

    snapped_x_min, snapped_y_max = pixel_to_world(snap_px_min, snap_py_max, z)
    snapped_x_max, snapped_y_min = pixel_to_world(snap_px_max, snap_py_min, z)

    return snapped_x_min, snapped_y_min, snapped_x_max, snapped_y_max


def move_tiles(src_dir, dst_dir, z):
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)

    for tile in os.listdir(src_dir):
        if os.path.exists(os.path.join(dst_dir, tile)):
            if os.path.getsize(os.path.join(dst_dir, tile)) < os.path.getsize(os.path.join(src_dir, tile)):
                os.remove(os.path.join(dst_dir, tile))
            else:
                continue

        shutil.move(
            os.path.join(src_dir, tile),
            os.path.join(dst_dir, tile)
        )


slave_data = subprocess.run(
    ['gdalinfo', slave_vrt_path],
    capture_output=True,
    text=True,
    check=True
)

upper_left = str(slave_data).split('Upper Left')[1].split(') (')[0].split('(')[1]
lower_right = str(slave_data).split('Lower Right')[1].split(') (')[0].split('(')[1]

x_min = float(upper_left.split(', ')[0])
x_max = float(lower_right.split(', ')[0])
y_min = float(lower_right.split(', ')[1])
y_max = float(upper_left.split(', ')[1])

for zoom in range(min_zoom, max_zoom + 1):
    os.mkdir('temp')

    print(Colors.GREEN + f"Zoom Level: {zoom}" + Colors.ENDC)

    bounded_x_min, bounded_y_min, bounded_x_max, bounded_y_max = calculate_snapped_bbox(zoom)

    print(Colors.PURPLE + f"Snapped Web Mercator Box: {bounded_x_min} xMin, {bounded_y_min} yMin, {bounded_x_max} xMax, {bounded_y_max} yMax" + Colors.ENDC)

    subprocess.run(
        ['gdalwarp',
         '-t_srs', 'EPSG:3857',
         '-te', str(bounded_x_min), str(bounded_y_min), str(bounded_x_max), str(bounded_y_max),
         '-tr', str((2 * web_mercator_max) / (256 * (2 ** zoom))), str((2 * web_mercator_max) / (256 * (2 ** zoom))),
         merged_vrt_path, 'clipped.tif']
    )

    subprocess.run(
        ['gdal2tiles.py',
         '--xyz', '-x', '-z', f'{zoom}-{zoom}',
         'clipped.tif', 'temp']
    )

    zoom_src = os.path.join(src_pyramid_path, str(zoom))
    zoom_dst = os.path.join(dst_pyramid_path, str(zoom))

    with os.scandir(zoom_src) as entries:
        for entry in entries:
            if not entry.is_dir():
                continue
            move_tiles(entry.path, os.path.join(zoom_dst, entry.name), zoom)

    os.remove('clipped.tif')
    shutil.rmtree('temp')