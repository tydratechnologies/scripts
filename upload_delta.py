#!/usr/bin/env python3
"""Upload changed XYZ tiles plus every tile at zoom levels 2–9 to S3 and EFS.

Run from the data directory. Delta manifests are read from:
    <location>/<color>/new.txt
    <location>/<color>/conflicts.txt

Manifest paths are relative to --master-pyramid. Independently, ALL files
within --master-pyramid/{2,3,4,5,6,7,8,9}/ are included on EVERY run.

The script builds one deduplicated staging pyramid using hard links (no data
copies), writes the same files to an already-mounted EFS destination using
parallel, atomic replacement, then uploads them to S3 with ONE parallel
`aws s3 cp --recursive` invocation.

Requires Linux for the safety check that --efs-destination is on an NFS mount.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path

COLORS = ("rtb", "blue", "mono")
MANIFESTS = ("new.txt", "conflicts.txt")
FULL_ZOOMS = range(2, 10)  # inclusive: 2 through 9
DEFAULT_EFS_WORKERS = 16


def positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be an integer") from None
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--locations", required=True, nargs="+", metavar="LOCATION",
        help="Location directory names beneath the current data directory, e.g. bahamas florida",
    )
    parser.add_argument("--color", required=True, choices=COLORS)
    parser.add_argument(
        "--master-pyramid", required=True, type=Path,
        help="Local master XYZ tile directory containing the final tiles",
    )
    parser.add_argument(
        "--s3-destination", required=True,
        help="S3 root for this master pyramid, e.g. s3://mybucket/tiles/rtb",
    )
    parser.add_argument(
        "--efs-destination", required=True, type=Path,
        help="Directory on an already-mounted EFS NFS filesystem, e.g. /mnt/efs/tiles/rtb",
    )
    parser.add_argument(
        "--efs-workers", default=DEFAULT_EFS_WORKERS, type=positive_int,
        help=f"Number of concurrent EFS file copies (default: {DEFAULT_EFS_WORKERS})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate inputs and list what would be uploaded, without copying or uploading",
    )
    parser.add_argument(
        "--keep-staging", action="store_true",
        help="Keep hard-link staging directory after successful transfers",
    )
    parser.add_argument(
        "--staging-parent", type=Path,
        help="Where to create hard-link staging (default: master parent; must share source filesystem)",
    )
    return parser.parse_args(argv)


def checked_location(name: str) -> str:
    if not name or name in (".", "..") or "/" in name or "\\" in name:
        raise ValueError(f"invalid location name: {name!r} (supply a single directory name)")
    return name


def checked_tile_path(value: str, manifest: Path, line_number: int) -> Path:
    """Validate the relative z/x/y.ext format emitted by merge.py."""
    parts = value.split("/")
    if len(parts) != 3 or any(part in ("", ".", "..") for part in parts) or "\\" in value or "\x00" in value:
        raise ValueError(f"invalid XYZ path in {manifest}:{line_number}: {value!r}")
    z, x, filename = parts
    if "." not in filename:
        raise ValueError(f"invalid XYZ filename in {manifest}:{line_number}: {value!r}")
    y, extension = filename.rsplit(".", 1)
    if not z.isdecimal() or not x.isdecimal() or not y.isdecimal() or not extension.isalnum():
        raise ValueError(f"invalid XYZ path in {manifest}:{line_number}: {value!r}")
    return Path(*parts)


def read_manifest(path: Path) -> set[Path]:
    if not path.is_file():
        raise ValueError(f"required manifest not found: {path}")
    entries: set[Path] = set()
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            value = line.strip()
            if not value or value.startswith("#"):
                continue
            entries.add(checked_tile_path(value, path, line_number))
    return entries


def collect_tiles(data_dir: Path, locations: list[str], color: str) -> set[Path]:
    union: set[Path] = set()
    for name in dict.fromkeys(locations):
        root = data_dir / checked_location(name) / color
        location_tiles: set[Path] = set()
        for manifest_name in MANIFESTS:
            path = root / manifest_name
            entries = read_manifest(path)
            print(f"  {name}/{color}/{manifest_name}: {len(entries):,} tiles")
            location_tiles.update(entries)
        added = len(location_tiles - union)
        print(f"  {name}: {len(location_tiles):,} distinct tiles ({added:,} added to combined delta)")
        union.update(location_tiles)
    return union


def collect_full_zooms(master: Path) -> set[Path]:
    """Collect every file under each required low-zoom directory, independent of manifests."""
    tiles: set[Path] = set()
    for z in FULL_ZOOMS:
        directory = master / str(z)
        if not directory.is_dir():
            raise ValueError(f"required full-upload zoom directory missing: {directory}")
        count = 0
        for path in directory.rglob("*"):
            if path.is_file():
                tiles.add(path.relative_to(master))
                count += 1
        print(f"  zoom {z}: {count:,} files")
    return tiles


def validate_sources(master: Path, tiles: set[Path]) -> None:
    missing = [p.as_posix() for p in sorted(tiles) if not (master / p).is_file()]
    if missing:
        preview = "\n  ".join(missing[:10])
        remaining = f"\n  ... and {len(missing) - 10:,} more" if len(missing) > 10 else ""
        raise ValueError(f"{len(missing):,} listed tiles are missing from master {master}:\n  {preview}{remaining}")


def validate_s3_destination(value: str) -> str:
    if not value.startswith("s3://") or any(c.isspace() for c in value):
        raise ValueError("--s3-destination must be an S3 URI such as s3://mybucket/tiles/rtb")
    bucket = value[5:].split("/", 1)[0]
    if not bucket or bucket in (".", ".."):
        raise ValueError("--s3-destination is missing a bucket name")
    return value.rstrip("/") + "/"


def mountpoint_unescape(value: str) -> str:
    """Undo the octal escaping used by Linux /proc/self/mountinfo."""
    return re.sub(r"\\([0-7]{3})", lambda m: chr(int(m.group(1), 8)), value)


def check_efs_nfs_mount(destination: Path) -> None:
    """Refuse to write to a local directory when the expected EFS mount is absent.

    The closest Linux mountpoint containing the requested path must be NFS.
    This also allows the destination to be a (not yet created) subdirectory
    within the EFS mount. It cannot prove that the NFS server is *the intended*
    EFS filesystem: users must supply the correct mounted path.
    """
    mountinfo = Path("/proc/self/mountinfo")
    if not mountinfo.is_file():
        raise ValueError("EFS mount safety check requires Linux (/proc/self/mountinfo)")

    closest: tuple[int, Path, str] | None = None
    with mountinfo.open("r", encoding="utf-8") as handle:
        for line in handle:
            before, sep, after = line.partition(" - ")
            if not sep:
                continue
            columns = before.split()
            fs_fields = after.split()
            if len(columns) < 5 or not fs_fields:
                continue
            point = Path(mountpoint_unescape(columns[4]))
            if destination == point or destination.is_relative_to(point):
                depth = len(point.parts)
                if closest is None or depth > closest[0]:
                    closest = (depth, point, fs_fields[0])

    if closest is None or closest[2] not in ("nfs", "nfs4"):
        found = f"{closest[1]} ({closest[2]})" if closest else "no mount"
        raise ValueError(
            f"--efs-destination {destination} is not under an NFS mount "
            f"(closest: {found}). Mount EFS first; refusing to write to local disk."
        )
    print(f"Verified NFS mount for EFS destination: {closest[1]}")


def atomic_efs_copy(src: Path, dst: Path) -> None:
    """Copy content to a same-directory temp file, then atomically replace dst."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{dst.name}.", suffix=".upload-tmp", dir=dst.parent)
    os.close(fd)
    temp_path = Path(temporary)
    try:
        # Preserve source permissions (important for tile-serving processes),
        # and modification times, instead of publishing a 0600 temp file.
        shutil.copy2(src, temp_path)
        os.replace(temp_path, dst)
    finally:
        temp_path.unlink(missing_ok=True)


def upload_to_efs(stage: Path, destination: Path, tiles: set[Path], workers: int) -> None:
    """Bounded concurrent EFS copies. Raise on any failure."""
    destination.mkdir(parents=True, exist_ok=True)
    pending: set[Future[None]] = set()
    completed = 0
    limit = workers * 4
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for rel in sorted(tiles):
            pending.add(pool.submit(atomic_efs_copy, stage / rel, destination / rel))
            if len(pending) >= limit:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    future.result()
                    completed += 1
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                future.result()
                completed += 1
    print(f"EFS complete: {completed:,} files copied/overwritten at {destination}")


def run(args: argparse.Namespace) -> int:
    data_dir = Path.cwd()
    master = args.master_pyramid.resolve()
    if not master.is_dir():
        raise ValueError(f"master pyramid is not a directory: {master}")
    s3_destination = validate_s3_destination(args.s3_destination)
    efs_destination = args.efs_destination.resolve(strict=False)
    parent = (args.staging_parent or master.parent).resolve()
    if not parent.is_dir():
        raise ValueError(f"staging parent is not a directory: {parent}")
    if os.stat(master).st_dev != os.stat(parent).st_dev:
        raise ValueError("staging parent must share the same filesystem as the master for os.link()")
    if (
        efs_destination == master
        or efs_destination.is_relative_to(master)
        or master.is_relative_to(efs_destination)
    ):
        raise ValueError("EFS destination must be separate from the local master pyramid")
    check_efs_nfs_mount(efs_destination)

    print(f"Data directory: {data_dir}")
    print(f"Master pyramid: {master}")
    print(f"S3 destination: {s3_destination}")
    print(f"EFS destination: {efs_destination}")
    print("Reading manifests:")
    manifest_tiles = collect_tiles(data_dir, args.locations, args.color)
    print("Collecting complete zoom directories 2–9:")
    full_zoom_tiles = collect_full_zooms(master)
    # Keep manifests strict, even if a low-zoom path is no longer in the master.
    validate_sources(master, manifest_tiles)
    tiles = manifest_tiles | full_zoom_tiles
    print(f"Distinct tiles from manifests: {len(manifest_tiles):,}")
    print(f"Tiles in complete zoom levels 2–9: {len(full_zoom_tiles):,}")
    print(f"Total unique files to send to BOTH destinations: {len(tiles):,}")

    if not tiles:
        print("Nothing to upload.")
        return 0
    if args.dry_run:
        print("Dry run: verified sources and EFS mount; no staging directory, EFS copies, or S3 upload.")
        return 0

    stage = Path(tempfile.mkdtemp(prefix=f".tile-delta-{args.color}-", dir=parent))
    staging_complete = False
    both_complete = False
    try:
        for rel in sorted(tiles):
            src = (master / rel).resolve(strict=True)  # resolve symlink-based tile sources
            dst = stage / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(src, dst)
            except OSError as exc:
                raise OSError(f"failed to hard-link {src} -> {dst}: {exc}") from exc
        staging_complete = True
        print(f"Staged {len(tiles):,} unique files as hard links at {stage}")

        # EFS first: an EFS failure stops before S3 begins. A failure may leave
        # partial changes at either destination, so retry the same manifest.
        print(f"Copying {len(tiles):,} files to mounted EFS with {args.efs_workers} workers ...", flush=True)
        upload_to_efs(stage, efs_destination, tiles, args.efs_workers)

        command = ["aws", "s3", "cp", str(stage) + os.sep, s3_destination, "--recursive", "--only-show-errors"]
        print("Uploading all files to S3 with one aws s3 cp --recursive command ...", flush=True)
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            print(f"AWS CLI exited with status {completed.returncode}.", file=sys.stderr)
            return completed.returncode
        both_complete = True
        print(f"S3 complete: {len(tiles):,} files uploaded/overwritten at {s3_destination}")
        print("Both destinations updated successfully.")
        return 0
    finally:
        if not staging_complete or (both_complete and not args.keep_staging):
            shutil.rmtree(stage)
        else:
            print(f"Staging directory retained at: {stage}", file=sys.stderr)
            if not both_complete:
                print("One or both destinations are incomplete; investigate the error and retry.", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    try:
        return run(parse_args(argv))
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
