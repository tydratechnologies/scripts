#!/usr/bin/env python3

import argparse
import numpy as np
from osgeo import gdal


RAMPS = {
    # Red -> Orange -> Yellow -> Green -> Cyan -> Blue -> Purple
    "rainbow": [
        (0.00, (76, 0, 115)),
        (0.20, (11, 44, 122)),
        (0.40, (32, 153, 143)),
        (0.55, (0, 219, 0)),
        (0.70, (255, 255, 0)),
        (0.85, (237, 161, 19)),
        (1.00, (211, 0, 0)),
    ],

    # Light blue -> dark blue
    "blue": [
        (0.00, (143, 229, 255)),
	(0.50, (0, 196, 255)),
        (1.00, (0, 92, 230)),
    ],

    # Constant cream
    "cream": [
        (0.00, (255, 255, 229)),
        (1.00, (255, 255, 229)),
    ],
}

def build_lut(ramp, size=256):
    positions = np.array([p for p, _ in ramp], dtype=np.float32)
    colors = np.array([c for _, c in ramp], dtype=np.float32)

    x = np.linspace(0, 1, size)
    lut = np.zeros((size, 3), dtype=np.uint8)

    for channel in range(3):
        lut[:, channel] = np.interp(x, positions, colors[:, channel]).astype(np.uint8)

    return lut


def colorize(input_path, output_path, ramp_name, min_value, max_value, nodata_alpha=0):
    src = gdal.Open(input_path, gdal.GA_ReadOnly)
    if src is None:
        raise RuntimeError(f"Could not open input raster: {input_path}")

    band = src.GetRasterBand(1)
    nodata = band.GetNoDataValue()

    xsize = src.RasterXSize
    ysize = src.RasterYSize

    driver = gdal.GetDriverByName("GTiff")

    dst = driver.Create(
        output_path,
        xsize,
        ysize,
        4,
        gdal.GDT_Byte,
        options=[
            "TILED=YES",
            "COMPRESS=DEFLATE",
            "BIGTIFF=IF_SAFER",
            "NUM_THREADS=ALL_CPUS",
        ],
    )

    if dst is None:
        raise RuntimeError(f"Could not create output raster: {output_path}")

    dst.SetGeoTransform(src.GetGeoTransform())
    dst.SetProjection(src.GetProjection())

    for i in range(1, 5):
        dst.GetRasterBand(1).SetColorInterpretation(gdal.GCI_RedBand)
        dst.GetRasterBand(2).SetColorInterpretation(gdal.GCI_GreenBand)
        dst.GetRasterBand(3).SetColorInterpretation(gdal.GCI_BlueBand)
        dst.GetRasterBand(4).SetColorInterpretation(gdal.GCI_AlphaBand)

    lut = build_lut(RAMPS[ramp_name], size=256)

    block_x, block_y = band.GetBlockSize()
    if block_x == 0 or block_y == 0:
        block_x, block_y = 1024, 1024

    scale = max_value - min_value
    if scale == 0:
        raise ValueError("min_value and max_value cannot be the same.")

    for y in range(0, ysize, block_y):
        rows = min(block_y, ysize - y)

        for x in range(0, xsize, block_x):
            cols = min(block_x, xsize - x)

            arr = band.ReadAsArray(x, y, cols, rows).astype(np.float32)

            valid = np.ones(arr.shape, dtype=bool)
            if nodata is not None:
                valid &= arr != nodata
            valid &= np.isfinite(arr)

            normalized = (arr - min_value) / scale
            normalized = np.clip(normalized, 0, 1)

            idx = (normalized * 255).astype(np.uint8)

            rgb = lut[idx]

            alpha = np.where(valid, 255, nodata_alpha).astype(np.uint8)

            dst.GetRasterBand(1).WriteArray(rgb[:, :, 0], x, y)
            dst.GetRasterBand(2).WriteArray(rgb[:, :, 1], x, y)
            dst.GetRasterBand(3).WriteArray(rgb[:, :, 2], x, y)
            dst.GetRasterBand(4).WriteArray(alpha, x, y)

    dst.FlushCache()
    src = None
    dst = None


def main():
    parser = argparse.ArgumentParser(
        description="Apply simple interpolated color reliefs to large rasters."
    )

    parser.add_argument("input", help="Input raster")
    parser.add_argument("output", help="Output RGBA GeoTIFF")
    parser.add_argument(
        "--ramp",
        choices=RAMPS.keys(),
        required=True,
        help="Color ramp name",
    )
    parser.add_argument("--min", dest="min_value", type=float, required=True)
    parser.add_argument("--max", dest="max_value", type=float, required=True)
    parser.add_argument(
        "--nodata-alpha",
        type=int,
        default=0,
        help="Alpha value for nodata pixels. Default: 0 transparent.",
    )

    args = parser.parse_args()

    colorize(
        input_path=args.input,
        output_path=args.output,
        ramp_name=args.ramp,
        min_value=args.min_value,
        max_value=args.max_value,
        nodata_alpha=args.nodata_alpha,
    )


if __name__ == "__main__":
    main()
