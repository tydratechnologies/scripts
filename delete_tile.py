#!/usr/bin/env python3

import argparse
from pathlib import Path


def load_tiles(tile_file):
    """Load z/x/y entries from tiles.txt."""
    tiles = set()

    with open(tile_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            parts = line.split("/")

            if len(parts) != 3:
                print(f"Skipping invalid line: {line}")
                continue

            z, x, y = parts
            tiles.add((z, x, y))

    return tiles


def main():
    parser = argparse.ArgumentParser(
        description="Recursively find and delete map tiles listed in a text file."
    )

    parser.add_argument(
        "tile_file",
        help="Text file containing tiles in z/x/y format"
    )

    parser.add_argument(
        "directory",
        help="Directory to recursively search"
    )

    parser.add_argument(
        "--delete",
        action="store_true",
        help="Actually delete files. Without this option, only show matches."
    )

    args = parser.parse_args()

    root = Path(args.directory)

    if not root.is_dir():
        raise SystemExit(f"Directory does not exist: {root}")

    tiles = load_tiles(args.tile_file)

    print(f"Loaded {len(tiles):,} tiles.")
    print(f"Searching: {root}")

    if not args.delete:
        print("DRY RUN — no files will be deleted.")
        print("Add --delete to actually delete matching files.")

    matched = 0
    deleted = 0

    for path in root.rglob("*"):

        if not path.is_file():
            continue

        try:
            relative = path.relative_to(root)
        except ValueError:
            continue

        parts = relative.parts

        if len(parts) < 3:
            continue

        # Expected directory structure:
        #
        # root/z/x/y.png
        #
        # Take the final three path components.
        z = parts[-3]
        x = parts[-2]

        # Strip extension from Y:
        # 49817.png -> 49817
        y = Path(parts[-1]).stem

        if (z, x, y) in tiles:
            matched += 1

            if args.delete:
                try:
                    path.unlink()
                    deleted += 1
                    print(f"DELETED: {path}")
                except OSError as e:
                    print(f"ERROR deleting {path}: {e}")
            else:
                print(f"MATCH:   {path}")

    print()
    print(f"Matched: {matched:,}")

    if args.delete:
        print(f"Deleted: {deleted:,}")
    else:
        print("Deleted: 0 (dry run)")


if __name__ == "__main__":
    main()
