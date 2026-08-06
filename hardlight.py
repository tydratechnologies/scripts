#!/usr/bin/env python3

import numpy as np
import rasterio
from rasterio.windows import Window
from pathlib import Path


def hard_light(base, top):
    """
    Hard Light blend mode.

    base: (3,H,W) float32 array normalized 0..1
    top:  (H,W) float32 array normalized 0..1
    """
    t = top[None, :, :]

    return np.where(
        t <= 0.5,
        2 * base * t,
        1 - 2 * (1 - base) * (1 - t)
    )


def main():
    print("\n=== Hard Light Raster Blend ===\n")

    color_path = Path(input("Path to color raster (.tif): ").strip())
    hs_path = Path(input("Path to hillshade raster (.tif): ").strip())
    out_path = Path(input("Output path (.tif): ").strip())

    tile_size = 2048

    if not color_path.exists():
        raise FileNotFoundError(f"Color raster not found: {color_path}")

    if not hs_path.exists():
        raise FileNotFoundError(f"Hillshade raster not found: {hs_path}")

    with rasterio.open(color_path) as c, rasterio.open(hs_path) as h:

        if c.width != h.width or c.height != h.height:
            raise ValueError(
                "Raster dimensions do not match.\n"
                f"Color: {c.width}x{c.height}\n"
                f"Hillshade: {h.width}x{h.height}"
            )

        if c.transform != h.transform:
            print("WARNING: Raster transforms differ.")

        if c.crs != h.crs:
            print("WARNING: Raster CRS differs.")

        profile = c.profile.copy()
        profile.update(
            count=3,
            dtype=rasterio.uint8,
            compress="deflate",
            tiled=True,
            blockxsize=512,
            blockysize=512,
            BIGTIFF="IF_SAFER"
        )

        total_x = (c.width + tile_size - 1) // tile_size
        total_y = (c.height + tile_size - 1) // tile_size
        total_windows = total_x * total_y

        print(f"\nRaster size: {c.width} x {c.height}")
        print(f"Processing window size: {tile_size} x {tile_size}")
        print(f"Total windows: {total_windows:,}\n")

        with rasterio.open(out_path, "w", **profile) as dst:

            window_count = 0

            for y in range(0, c.height, tile_size):
                for x in range(0, c.width, tile_size):

                    window = Window(
                        x,
                        y,
                        min(tile_size, c.width - x),
                        min(tile_size, c.height - y)
                    )

                    color = (
                        c.read([1, 2, 3], window=window)
                        .astype(np.float32) / 255.0
                    )

                    hs = (
                        h.read(1, window=window)
                        .astype(np.float32) / 255.0
                    )

                    out = hard_light(color, hs)

                    out = np.clip(
                        out * 255.0 + 0.5,
                        0,
                        255
                    ).astype(np.uint8)

                    dst.write(out, window=window)

                    window_count += 1

                    if window_count % 100 == 0:
                        print(
                            f"Processed {window_count:,} / "
                            f"{total_windows:,} windows"
                        )

    print(f"\nDone.")
    print(f"Output written to:\n{out_path}\n")


if __name__ == "__main__":
    main()
