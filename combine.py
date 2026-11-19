#!/usr/bin/env python3

from pathlib import Path
from PIL import Image

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

TILE_ROOT = Path("/home/scotzark/yourbit/tiles/rtb/master/")
ZOOM = 8
OUTPUT_FILE = Path("/home/scotzark/yourbit/scripts/zoom_8_merged.png")


def main():
    zoom_dir = TILE_ROOT / str(ZOOM)

    if not zoom_dir.exists():
        raise FileNotFoundError(f"Zoom directory not found: {zoom_dir}")

    # ---------------------------------------------------------------
    # Find all tiles
    # Expected layout: tiles/{z}/{x}/{y}.png
    # ---------------------------------------------------------------

    tiles = []

    for x_dir in zoom_dir.iterdir():
        if not x_dir.is_dir():
            continue

        try:
            x = int(x_dir.name)
        except ValueError:
            continue

        for tile_path in x_dir.glob("*.png"):
            try:
                y = int(tile_path.stem)
            except ValueError:
                continue

            tiles.append((x, y, tile_path))

    if not tiles:
        raise RuntimeError(f"No PNG tiles found under {zoom_dir}")

    # ---------------------------------------------------------------
    # Determine tile bounds
    # ---------------------------------------------------------------

    xs = [x for x, y, path in tiles]
    ys = [y for x, y, path in tiles]

    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)

    columns = max_x - min_x + 1
    rows = max_y - min_y + 1

    # Read one tile to determine dimensions.
    with Image.open(tiles[0][2]) as sample:
        tile_width, tile_height = sample.size

    output_width = columns * tile_width
    output_height = rows * tile_height

    print(f"Zoom:       {ZOOM}")
    print(f"Tiles:      {len(tiles):,}")
    print(f"X range:    {min_x} -> {max_x}")
    print(f"Y range:    {min_y} -> {max_y}")
    print(f"Grid:       {columns} x {rows} tiles")
    print(f"Tile size:  {tile_width} x {tile_height}")
    print(f"Output:     {output_width} x {output_height} pixels")

    # ---------------------------------------------------------------
    # Create output image
    # ---------------------------------------------------------------

    mosaic = Image.new(
        "RGBA",
        (output_width, output_height),
        (0, 0, 0, 0)
    )

    # ---------------------------------------------------------------
    # Paste tiles
    # ---------------------------------------------------------------

    for i, (x, y, tile_path) in enumerate(tiles, start=1):

        pixel_x = (x - min_x) * tile_width
        pixel_y = (y - min_y) * tile_height

        with Image.open(tile_path) as tile:
            tile = tile.convert("RGBA")

            if tile.size != (tile_width, tile_height):
                print(
                    f"WARNING: {tile_path} has size {tile.size}, "
                    f"expected {(tile_width, tile_height)}"
                )

            mosaic.paste(tile, (pixel_x, pixel_y))

        if i % 100 == 0 or i == len(tiles):
            print(f"\rProcessed {i:,}/{len(tiles):,} tiles", end="")

    print()

    # ---------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------

    print(f"Saving {OUTPUT_FILE} ...")
    mosaic.save(OUTPUT_FILE, "PNG")

    print("Done.")


if __name__ == "__main__":
    main()
