#!/usr/bin/env python3

import math
import argparse

MIN_ZOOM = 10
MAX_ZOOM = 17
MAX_LAT = 85.05112878  # Web Mercator latitude limit


def lon_to_tile_x(lon, zoom):
    """Convert longitude to XYZ tile X coordinate."""
    n = 2 ** zoom
    return int(math.floor((lon + 180.0) / 360.0 * n))


def lat_to_tile_y(lat, zoom):
    """Convert latitude to XYZ tile Y coordinate."""
    lat = max(min(lat, MAX_LAT), -MAX_LAT)

    lat_rad = math.radians(lat)
    n = 2 ** zoom

    y = (
        1.0
        - math.asinh(math.tan(lat_rad)) / math.pi
    ) / 2.0 * n

    return int(math.floor(y))


def get_tiles(west, south, east, north):
    """
    Yield every XYZ Web Mercator tile intersecting the bounding box
    for zoom levels 10 through 17.
    """

    if south > north:
        raise ValueError("south must be <= north")

    if not (-180 <= west <= 180 and -180 <= east <= 180):
        raise ValueError("Longitude must be between -180 and 180.")

    if not (-90 <= south <= 90 and -90 <= north <= 90):
        raise ValueError("Latitude must be between -90 and 90.")

    if west > east:
        raise ValueError(
            "This script does not currently support bounding boxes "
            "that cross the antimeridian."
        )

    for zoom in range(MIN_ZOOM, MAX_ZOOM + 1):
        n = 2 ** zoom

        min_x = max(0, lon_to_tile_x(west, zoom))
        max_x = min(n - 1, lon_to_tile_x(east, zoom))

        # Tile Y increases southward, so north gives the smaller Y.
        min_y = max(0, lat_to_tile_y(north, zoom))
        max_y = min(n - 1, lat_to_tile_y(south, zoom))

        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                yield zoom, x, y


def main():
    parser = argparse.ArgumentParser(
        description="Generate XYZ Web Mercator tiles for a bounding box."
    )

    parser.add_argument("west", type=float, help="Western longitude")
    parser.add_argument("south", type=float, help="Southern latitude")
    parser.add_argument("east", type=float, help="Eastern longitude")
    parser.add_argument("north", type=float, help="Northern latitude")

    args = parser.parse_args()

    count = 0

    with open("tiles.txt", "w", encoding="utf-8") as f:
        for zoom, x, y in get_tiles(
            args.west,
            args.south,
            args.east,
            args.north
        ):
            f.write(f"{zoom}/{x}/{y}\n")
            count += 1

    print(f"Wrote {count:,} tiles to tiles.txt")


if __name__ == "__main__":
    main()
