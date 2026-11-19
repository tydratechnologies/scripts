from pathlib import Path
import re

# Change this to the directory you want to scan.
SEARCH_DIR = Path(".")

OUTPUT_FILE = SEARCH_DIR / "tiles.txt"

# Matches resolutions such as:
# 50cm, 25cm, 1m, 2m, 10m, 15m, etc.
RESOLUTION_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(\d+(?:\.\d+)?)\s*(cm|m)(?![A-Za-z])",
    re.IGNORECASE
)


def get_resolution_meters(filename):
    """
    Return the resolution in meters.

    Examples:
        4m   -> 4.0
        1m   -> 1.0
        50cm -> 0.5
        25cm -> 0.25

    Returns None if no resolution is found.
    """
    match = RESOLUTION_PATTERN.search(filename)

    if not match:
        return None

    value = float(match.group(1))
    unit = match.group(2).lower()

    if unit == "cm":
        value /= 100.0

    return value


def sort_key(path):
    resolution = get_resolution_meters(path.name)

    # Files with no stated resolution go first.
    if resolution is None:
        return (0, 0, path.name.lower())

    # Then sort from largest/coarsest resolution
    # to smallest/finest resolution.
    return (1, -resolution, path.name.lower())


def main():
    files = [
        path
        for path in SEARCH_DIR.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".bag", ".tif"}
    ]

    files.sort(key=sort_key)

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for path in files:
            f.write(path.name + "\n")

    print(f"Found {len(files)} .bag/.tif files.")
    print(f"Wrote sorted list to: {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    main()

