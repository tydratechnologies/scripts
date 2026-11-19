#!/usr/bin/env python3

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path


def run(cmd, capture=True):
    """Run a command and raise an error if it fails."""
    return subprocess.run(
        cmd,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )


def gdalinfo_json(path):
    """Return gdalinfo -json output as a Python dict."""
    result = run(["gdalinfo", "-json", str(path)])
    return json.loads(result.stdout)


def get_default_metadata(info):
    """
    GDAL normally puts BAG metadata in the default metadata domain "".
    Handle a few possible JSON layouts.
    """
    metadata = info.get("metadata", {})

    if isinstance(metadata, dict):
        if "" in metadata and isinstance(metadata[""], dict):
            return metadata[""]

        # Some builds may expose the values directly
        if "MIN_RESOLUTION_X" in metadata:
            return metadata

    return {}


def inspect_bag(path):
    """
    Inspect a BAG and return information needed to determine
    whether it is a variable-resolution BAG.
    """
    info = gdalinfo_json(path)
    md = get_default_metadata(info)

    min_x = md.get("MIN_RESOLUTION_X")
    min_y = md.get("MIN_RESOLUTION_Y")
    max_x = md.get("MAX_RESOLUTION_X")
    max_y = md.get("MAX_RESOLUTION_Y")

    # No MIN_RESOLUTION metadata = probably not a VR BAG
    if min_x is None or min_y is None:
        return {
            "is_vr": False,
            "info": info,
        }

    min_x = float(min_x)
    min_y = float(min_y)

    if max_x is not None:
        max_x = float(max_x)
    if max_y is not None:
        max_y = float(max_y)

    geotransform = info.get("geoTransform")

    if not geotransform or len(geotransform) < 6:
        raise RuntimeError("Could not determine BAG pixel size.")

    base_x = abs(float(geotransform[1]))
    base_y = abs(float(geotransform[5]))

    finest = min(min_x, min_y)

    return {
        "is_vr": True,
        "base_x": base_x,
        "base_y": base_y,
        "min_x": min_x,
        "min_y": min_y,
        "max_x": max_x,
        "max_y": max_y,
        "finest": finest,
        "ratio_x": base_x / min_x,
        "ratio_y": base_y / min_y,
        "info": info,
    }


def choose_resolution(finest_native, finest_allowed):
    """
    Don't create a raster finer than finest_allowed.

    Examples with --finest 0.5:
       native 0.25 m -> output 0.5 m
       native 0.51 m -> output 0.51 m
       native 1.00 m -> output 1.00 m
    """
    if finest_allowed is None:
        return finest_native

    return max(finest_native, finest_allowed)


def convert_bag(source, destination, resolution):
    """Convert VR BAG to an interpolated tiled compressed GeoTIFF."""

    cmd = [
        "gdal_translate",
        "-oo", "MODE=INTERPOLATED",
        "-oo", f"RESX={resolution}",
        "-oo", f"RESY={resolution}",
        "-of", "GTiff",
        "-co", "TILED=YES",
        "-co", "COMPRESS=DEFLATE",
        "-co", "BIGTIFF=IF_SAFER",
        str(source),
        str(destination),
    ]

    print("  Running:")
    print("   ", " ".join(cmd))

    run(cmd, capture=False)


def validate_output(path):
    """
    Basic validation:
      - file exists
      - nonzero size
      - GDAL can open it
      - raster has dimensions
      - raster contains at least one band
    """
    if not path.exists():
        return False, "Output file does not exist."

    if path.stat().st_size == 0:
        return False, "Output file is empty."

    try:
        info = gdalinfo_json(path)
    except Exception as exc:
        return False, f"gdalinfo could not read output: {exc}"

    size = info.get("size", [])
    bands = info.get("bands", [])

    if len(size) != 2 or size[0] <= 0 or size[1] <= 0:
        return False, "Output raster has invalid dimensions."

    if not bands:
        return False, "Output raster contains no bands."

    return True, f"{size[0]} x {size[1]}, {len(bands)} band(s)"


def process_bag(path, args):
    print()
    print("=" * 78)
    print(path)

    try:
        details = inspect_bag(path)
    except Exception as exc:
        print(f"ERROR inspecting file: {exc}")
        return False

    if not details["is_vr"]:
        print("Not identified as a variable-resolution BAG. Leaving unchanged.")
        return True

    print(
        f"VR BAG detected:"
        f"\n  Base raster:       {details['base_x']:.3f} x "
        f"{details['base_y']:.3f} m"
        f"\n  Finest VR data:    {details['min_x']:.3f} x "
        f"{details['min_y']:.3f} m"
    )

    if details["max_x"] is not None:
        print(
            f"  Coarsest VR data:  {details['max_x']:.3f} x "
            f"{details['max_y']:.3f} m"
        )

    print(
        f"  Resolution ratio:  about "
        f"{max(details['ratio_x'], details['ratio_y']):.1f}x"
    )

    # Only process BAGs where the normal/base grid is meaningfully
    # coarser than the VR contents.
    ratio = max(details["ratio_x"], details["ratio_y"])

    if ratio < args.minimum_ratio:
        print(
            f"Base grid is less than {args.minimum_ratio:g}x coarser "
            f"than VR data. Leaving unchanged."
        )
        return True

    resolution = choose_resolution(
        details["finest"],
        args.finest,
    )

    print(f"  Output resolution: {resolution:.6g} m")

    output = path.with_name(
        path.stem + f"_interpolated_{resolution:g}m.tif"
    )

    if output.exists():
        if not args.overwrite:
            print(f"SKIP: output already exists: {output}")
            return True

        print(f"Removing existing output: {output}")
        output.unlink()

    if args.dry_run:
        print(f"DRY RUN: would create {output}")
        if args.delete:
            print(f"DRY RUN: would delete {path} after validation")
        return True

    try:
        convert_bag(path, output, resolution)
    except subprocess.CalledProcessError as exc:
        print()
        print("CONVERSION FAILED.")
        print("Original BAG WILL NOT be deleted.")
        if exc.stderr:
            print(exc.stderr)
        return False

    print()
    print("Validating output...")

    valid, message = validate_output(output)

    if not valid:
        print(f"VALIDATION FAILED: {message}")
        print("Original BAG WILL NOT be deleted.")
        return False

    print(f"Validation passed: {message}")
    print(f"Created: {output}")

    if args.delete:
        try:
            path.unlink()
            print(f"Deleted original VR BAG: {path}")
        except Exception as exc:
            print(f"WARNING: could not delete original BAG: {exc}")
            return False
    else:
        print("Original BAG retained.")

    return True


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Find NOAA variable-resolution BAG files, convert them "
            "to interpolated GeoTIFFs, validate the outputs, and "
            "optionally delete the original BAGs."
        )
    )

    parser.add_argument(
        "directory",
        nargs="?",
        default=".",
        help="Directory to scan (default: current directory)",
    )

    parser.add_argument(
        "--finest",
        type=float,
        default=0.5,
        help=(
            "Finest output resolution allowed, in meters. "
            "Default: 0.5. Example: a 0.25 m BAG becomes 0.5 m; "
            "a 1 m BAG remains 1 m. Use 0 to use the BAG's "
            "absolute finest native resolution."
        ),
    )

    parser.add_argument(
        "--minimum-ratio",
        type=float,
        default=2.0,
        help=(
            "Only convert BAGs whose base raster is at least this "
            "many times coarser than the finest VR resolution. "
            "Default: 2"
        ),
    )

    parser.add_argument(
        "--delete",
        action="store_true",
        help=(
            "Delete original BAG after successful conversion "
            "and validation."
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without converting or deleting.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output GeoTIFFs.",
    )

    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Only scan the specified directory, not subdirectories.",
    )

    args = parser.parse_args()

    # --finest 0 means unlimited / use native finest resolution
    if args.finest == 0:
        args.finest = None

    root = Path(args.directory).expanduser().resolve()

    if not root.is_dir():
        print(f"ERROR: not a directory: {root}", file=sys.stderr)
        sys.exit(1)

    # Verify GDAL exists
    try:
        version = run(["gdalinfo", "--version"]).stdout.strip()
    except Exception:
        print(
            "ERROR: gdalinfo was not found. "
            "Make sure GDAL is installed and on PATH.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(version)
    print(f"Scanning: {root}")

    pattern = "*.bag"

    if args.no_recursive:
        bags = sorted(root.glob(pattern))
    else:
        bags = sorted(root.rglob(pattern))

    if not bags:
        print("No .bag files found.")
        return

    print(f"Found {len(bags)} BAG file(s).")

    successes = 0
    failures = 0

    for bag in bags:
        if process_bag(bag, args):
            successes += 1
        else:
            failures += 1

    print()
    print("=" * 78)
    print("FINISHED")
    print(f"Successful/skipped: {successes}")
    print(f"Failures:           {failures}")

    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
