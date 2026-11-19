#!/usr/bin/env python3

import csv
import math
import os
import sys


ZOOM = 17
TILE_ROOT = "/home/scotzark/yourbit/tiles/rtb/master"


def latlon_to_xyz(lat, lon, zoom):
    """
    Convert latitude/longitude to a Web Mercator XYZ tile coordinate.
    """

    # Web Mercator is only defined up to approximately +/-85.0511 degrees.
    lat = max(min(lat, 85.05112878), -85.05112878)

    n = 2 ** zoom

    x = int((lon + 180.0) / 360.0 * n)

    lat_rad = math.radians(lat)

    y = int(
        (
            1.0
            - math.asinh(math.tan(lat_rad)) / math.pi
        )
        / 2.0
        * n
    )

    # Protect against coordinates exactly on the outer limits.
    x = max(0, min(x, n - 1))
    y = max(0, min(y, n - 1))

    return x, y


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <input.csv>")
        sys.exit(1)

    input_csv = os.path.abspath(sys.argv[1])

    if not os.path.isfile(input_csv):
        print(f"Error: CSV file does not exist: {input_csv}")
        sys.exit(1)

    # spots.csv goes in the same directory as the input CSV.
    output_csv = os.path.join(
        os.path.dirname(input_csv),
        "spots.csv"
    )

    kept = 0
    discarded = 0
    invalid = 0

    try:
        with open(input_csv, "r", newline="", encoding="utf-8-sig") as infile:
            reader = csv.DictReader(infile)

            if reader.fieldnames is None:
                print("Error: CSV has no header.")
                sys.exit(1)

            # Verify the columns we need exist.
            if "latitude" not in reader.fieldnames:
                print("Error: CSV does not contain a 'latitude' column.")
                sys.exit(1)

            if "longitude" not in reader.fieldnames:
                print("Error: CSV does not contain a 'longitude' column.")
                sys.exit(1)

            with open(
                output_csv,
                "w",
                newline="",
                encoding="utf-8"
            ) as outfile:

                writer = csv.DictWriter(
                    outfile,
                    fieldnames=reader.fieldnames
                )

                # Preserve the same columns/header as the source CSV.
                writer.writeheader()

                for row_number, row in enumerate(reader, start=2):

                    try:
                        lat = float(row["latitude"])
                        lon = float(row["longitude"])
                    except (ValueError, TypeError):
                        print(
                            f"Warning: row {row_number} has invalid "
                            f"latitude/longitude. Skipping."
                        )
                        invalid += 1
                        continue

                    x, y = latlon_to_xyz(lat, lon, ZOOM)

                    tile_path = os.path.join(
                        TILE_ROOT,
                        str(ZOOM),
                        str(x),
                        f"{y}.png"
                    )

                    if os.path.isfile(tile_path):
                        writer.writerow(row)
                        kept += 1
                    else:
                        discarded += 1

    except PermissionError as exc:
        print(f"Error: permission denied: {exc}")
        sys.exit(1)

    except OSError as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    print()
    print("Finished.")
    print(f"Input:     {input_csv}")
    print(f"Output:    {output_csv}")
    print(f"Kept:      {kept}")
    print(f"Discarded: {discarded}")

    if invalid:
        print(f"Invalid:   {invalid}")


if __name__ == "__main__":
    main()
