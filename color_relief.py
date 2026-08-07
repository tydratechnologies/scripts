#!/usr/bin/env python3
"""Apply an interpolated color ramp to band 1 of a raster.

The output is an RGBA GeoTIFF. Source NoData/masked pixels receive the
requested alpha value and a neutral RGB value, so they do not appear as a
ramp endpoint in software that ignores alpha.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Sequence

import numpy as np
from osgeo import gdal


gdal.UseExceptions()


RAMPS = {
    # Purple -> blue -> cyan/green -> yellow -> orange -> red
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


def alpha_value(value: str) -> int:
    """argparse converter for an 8-bit alpha value."""
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer from 0 to 255") from exc

    if not 0 <= parsed <= 255:
        raise argparse.ArgumentTypeError("must be an integer from 0 to 255")
    return parsed


def build_lut(ramp: Sequence[tuple[float, tuple[int, int, int]]], size: int = 256) -> np.ndarray:
    """Build an RGB lookup table by interpolating the supplied ramp."""
    if size < 2:
        raise ValueError("LUT size must be at least 2")

    positions = np.asarray([position for position, _ in ramp], dtype=np.float32)
    colors = np.asarray([color for _, color in ramp], dtype=np.float32)

    if positions.ndim != 1 or colors.shape != (len(positions), 3):
        raise ValueError("Invalid color ramp")
    if np.any(np.diff(positions) < 0) or positions[0] != 0.0 or positions[-1] != 1.0:
        raise ValueError("Ramp positions must be sorted and span 0.0 through 1.0")

    x = np.linspace(0.0, 1.0, size, dtype=np.float32)
    lut = np.empty((size, 3), dtype=np.uint8)

    for channel in range(3):
        lut[:, channel] = np.rint(
            np.interp(x, positions, colors[:, channel])
        ).astype(np.uint8)

    return lut


def colorize(
    input_path: str,
    output_path: str,
    ramp_name: str,
    min_value: float,
    max_value: float,
    nodata_alpha: int = 0,
) -> None:
    """Colorize band 1 and write a four-band RGBA GeoTIFF."""
    if max_value <= min_value:
        raise ValueError("--max must be greater than --min")
    if not 0 <= nodata_alpha <= 255:
        raise ValueError("nodata_alpha must be between 0 and 255")

    src = gdal.Open(input_path, gdal.GA_ReadOnly)
    if src is None:
        raise RuntimeError(f"Could not open input raster: {input_path}")
    if src.RasterCount < 1:
        raise RuntimeError("Input raster has no bands")

    band = src.GetRasterBand(1)
    source_mask_band = band.GetMaskBand()
    nodata = band.GetNoDataValue()

    xsize = src.RasterXSize
    ysize = src.RasterYSize
    lut = build_lut(RAMPS[ramp_name])
    scale = max_value - min_value

    driver = gdal.GetDriverByName("GTiff")
    if driver is None:
        raise RuntimeError("GDAL GTiff driver is unavailable")

    dst = None
    try:
        dst = driver.Create(
            output_path,
            xsize,
            ysize,
            4,
            gdal.GDT_Byte,
            options=[
                "TILED=YES",
                "COMPRESS=DEFLATE",
                "PREDICTOR=2",
                "BIGTIFF=IF_SAFER",
                "NUM_THREADS=ALL_CPUS",
                "INTERLEAVE=PIXEL",
            ],
        )
        if dst is None:
            raise RuntimeError(f"Could not create output raster: {output_path}")

        dst.SetGeoTransform(src.GetGeoTransform())
        dst.SetProjection(src.GetProjection())
        dst.SetMetadata(src.GetMetadata())

        red_band = dst.GetRasterBand(1)
        green_band = dst.GetRasterBand(2)
        blue_band = dst.GetRasterBand(3)
        alpha_band = dst.GetRasterBand(4)

        red_band.SetColorInterpretation(gdal.GCI_RedBand)
        green_band.SetColorInterpretation(gdal.GCI_GreenBand)
        blue_band.SetColorInterpretation(gdal.GCI_BlueBand)
        alpha_band.SetColorInterpretation(gdal.GCI_AlphaBand)

        red_band.SetDescription("Red")
        green_band.SetDescription("Green")
        blue_band.SetDescription("Blue")
        alpha_band.SetDescription("Alpha")

        # A dataset mask improves compatibility with readers that do not honor
        # the explicit alpha band. The mask is written with the same values.
        red_band.CreateMaskBand(gdal.GMF_PER_DATASET)
        output_mask_band = red_band.GetMaskBand()

        block_x, block_y = band.GetBlockSize()
        if block_x <= 0 or block_y <= 0:
            block_x, block_y = 1024, 1024

        for y in range(0, ysize, block_y):
            rows = min(block_y, ysize - y)

            for x in range(0, xsize, block_x):
                cols = min(block_x, xsize - x)

                arr = band.ReadAsArray(x, y, cols, rows)
                if arr is None:
                    raise RuntimeError(f"Failed reading source block at x={x}, y={y}")
                arr = arr.astype(np.float64, copy=False)

                # Start with GDAL's source mask. This handles explicit NoData,
                # source mask bands, and alpha-derived masks where applicable.
                source_mask = source_mask_band.ReadAsArray(x, y, cols, rows)
                if source_mask is None:
                    valid = np.ones(arr.shape, dtype=bool)
                else:
                    valid = source_mask != 0

                valid &= np.isfinite(arr)

                # Explicitly enforce the declared NoData value as well. NaN
                # NoData must be handled separately because NaN != NaN.
                if nodata is not None:
                    if np.isnan(nodata):
                        valid &= ~np.isnan(arr)
                    else:
                        valid &= arr != nodata

                # Only normalize valid pixels. Invalid pixels remain index 0,
                # then receive neutral RGB below.
                idx = np.zeros(arr.shape, dtype=np.uint8)
                if np.any(valid):
                    normalized = (arr[valid] - min_value) / scale
                    normalized = np.clip(normalized, 0.0, 1.0)
                    idx[valid] = np.rint(normalized * 255.0).astype(np.uint8)

                rgb = lut[idx].copy()

                # Critical fix: do not leave a ramp endpoint under transparent
                # pixels. This prevents NoData from appearing deep red in
                # software that ignores or strips the alpha channel.
                rgb[~valid] = (0, 0, 0)

                alpha = np.full(arr.shape, nodata_alpha, dtype=np.uint8)
                alpha[valid] = 255

                red_band.WriteArray(rgb[:, :, 0], x, y)
                green_band.WriteArray(rgb[:, :, 1], x, y)
                blue_band.WriteArray(rgb[:, :, 2], x, y)
                alpha_band.WriteArray(alpha, x, y)
                output_mask_band.WriteArray(alpha, x, y)

        dst.FlushCache()

    except Exception:
        # Close datasets before removing a partial output.
        dst = None
        src = None
        if os.path.exists(output_path):
            try:
                driver.Delete(output_path)
            except Exception:
                try:
                    os.remove(output_path)
                except OSError:
                    pass
        raise
    finally:
        dst = None
        src = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply an interpolated color relief to band 1 of a raster."
    )
    parser.add_argument("input", help="Input raster")
    parser.add_argument("output", help="Output RGBA GeoTIFF")
    parser.add_argument(
        "--ramp",
        choices=sorted(RAMPS),
        required=True,
        help="Color ramp name",
    )
    parser.add_argument("--min", dest="min_value", type=float, required=True)
    parser.add_argument("--max", dest="max_value", type=float, required=True)
    parser.add_argument(
        "--nodata-alpha",
        type=alpha_value,
        default=0,
        help="Alpha value for NoData pixels, from 0 to 255 (default: 0)",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        colorize(
            input_path=args.input,
            output_path=args.output,
            ramp_name=args.ramp,
            min_value=args.min_value,
            max_value=args.max_value,
            nodata_alpha=args.nodata_alpha,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
