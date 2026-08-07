#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window


OUTPUT_NODATA = 255
TILE_SIZE = 2048


def hard_light(base: np.ndarray, top: np.ndarray) -> np.ndarray:
    """
    Apply the Hard Light blend mode.

    Parameters
    ----------
    base
        RGB array with shape (3, H, W), normalized to 0..1.
    top
        Single-band hillshade array with shape (H, W), normalized to 0..1.
    """
    t = top[None, :, :]

    return np.where(
        t <= 0.5,
        2.0 * base * t,
        1.0 - 2.0 * (1.0 - base) * (1.0 - t),
    )


def main() -> None:
    print("\n=== Hard Light Raster Blend ===\n")

    color_path = Path(input("Path to color raster (.tif): ").strip())
    hs_path = Path(input("Path to hillshade raster (.tif): ").strip())
    out_path = Path(input("Output path (.tif): ").strip())

    if not color_path.exists():
        raise FileNotFoundError(f"Color raster not found: {color_path}")

    if not hs_path.exists():
        raise FileNotFoundError(f"Hillshade raster not found: {hs_path}")

    if out_path.resolve() in {color_path.resolve(), hs_path.resolve()}:
        raise ValueError("Output path must differ from both input paths.")

    with rasterio.open(color_path) as color_src, rasterio.open(hs_path) as hs_src:
        if color_src.count < 3:
            raise ValueError(
                f"Color raster must have at least 3 bands; found {color_src.count}."
            )

        if hs_src.count < 1:
            raise ValueError("Hillshade raster has no readable bands.")

        if color_src.width != hs_src.width or color_src.height != hs_src.height:
            raise ValueError(
                "Raster dimensions do not match.\n"
                f"Color: {color_src.width}x{color_src.height}\n"
                f"Hillshade: {hs_src.width}x{hs_src.height}"
            )

        if color_src.transform != hs_src.transform:
            raise ValueError("Raster transforms do not match.")

        if color_src.crs != hs_src.crs:
            raise ValueError("Raster CRS values do not match.")

        profile = color_src.profile.copy()
        profile.update(
            driver="GTiff",
            count=3,
            dtype=rasterio.uint8,
            nodata=OUTPUT_NODATA,
            compress="deflate",
            tiled=True,
            blockxsize=512,
            blockysize=512,
            BIGTIFF="IF_SAFER",
        )

        total_x = (color_src.width + TILE_SIZE - 1) // TILE_SIZE
        total_y = (color_src.height + TILE_SIZE - 1) // TILE_SIZE
        total_windows = total_x * total_y

        print(f"Raster size: {color_src.width} x {color_src.height}")
        print(f"Processing window size: {TILE_SIZE} x {TILE_SIZE}")
        print(f"Total windows: {total_windows:,}")
        print(f"Hillshade declared NoData: {hs_src.nodata}")
        print(f"Output NoData: ({OUTPUT_NODATA}, {OUTPUT_NODATA}, {OUTPUT_NODATA})\n")

        try:
            with rasterio.open(out_path, "w", **profile) as dst:
                window_count = 0

                for y in range(0, color_src.height, TILE_SIZE):
                    for x in range(0, color_src.width, TILE_SIZE):
                        window = Window(
                            x,
                            y,
                            min(TILE_SIZE, color_src.width - x),
                            min(TILE_SIZE, color_src.height - y),
                        )

                        color_raw = color_src.read([1, 2, 3], window=window)
                        color_valid = color_src.dataset_mask(window=window) != 0

                        hs_raw = hs_src.read(1, window=window)

                        # Honor the hillshade raster's declared NoData value instead
                        # of assuming a hard-coded value such as 0. This is important
                        # because the processed hillshade may legitimately contain 0
                        # while using 255 as its NoData sentinel.
                        hs_nodata = hs_src.nodata
                        if hs_nodata is None:
                            hillshade_valid = np.ones(hs_raw.shape, dtype=bool)
                        else:
                            hillshade_valid = hs_raw != hs_nodata

                        # Also honor any explicit mask associated with the hillshade.
                        hillshade_valid &= hs_src.read_masks(1, window=window) != 0

                        valid = color_valid & hillshade_valid

                        color = color_raw.astype(np.float32) / 255.0
                        hillshade = hs_raw.astype(np.float32) / 255.0

                        out = hard_light(color, hillshade)
                        out = np.clip(
                            out * 255.0 + 0.5,
                            0,
                            255,
                        ).astype(np.uint8)

                        # Pure white is the output NoData sentinel.
                        out[:, ~valid] = OUTPUT_NODATA

                        dst.write(out, window=window)

                        window_count += 1
                        if window_count % 100 == 0 or window_count == total_windows:
                            print(
                                f"Processed {window_count:,} / "
                                f"{total_windows:,} windows"
                            )

        except Exception:
            # Avoid leaving a partial output that could be mistaken for a valid result.
            if out_path.exists():
                out_path.unlink()
            raise

    print("\nDone.")
    print(f"Output written to:\n{out_path}\n")


if __name__ == "__main__":
    main()
