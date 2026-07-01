#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any


WEB_MERCATOR_HALF_WORLD = 20037508.342789244
DEFAULT_WORKERS = min(8, os.cpu_count() or 1)
RESAMPLING_NAMES = ("nearest", "bilinear", "cubic", "average")
OUTPUT_DTYPES = ("auto", "source", "uint8", "uint16")


@dataclass(frozen=True)
class TileJob:
    z: int
    x: int
    y: int
    extension: str
    relative_path: Path


@dataclass(frozen=True)
class ChunkArgs:
    master_vrt: str
    master_pyramid: str
    jobs: tuple[TileJob, ...]
    tile_size: int
    driver: str
    resampling_name: str
    output_dtype: str
    scale_range: tuple[float, float] | None


@dataclass(frozen=True)
class ChunkResult:
    rendered: int
    failures: tuple[str, ...]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Redraw conflicting XYZ tiles from an updated EPSG:3857 master VRT."
    )
    parser.add_argument("--master-vrt", required=True, type=Path)
    parser.add_argument("--master-pyramid", required=True, type=Path)
    parser.add_argument("--conflicts", required=True, type=Path)
    parser.add_argument("--workers", default=DEFAULT_WORKERS, type=positive_int)
    parser.add_argument("--chunk-size", default=100, type=positive_int)
    parser.add_argument("--tile-size", default=256, type=positive_int)
    parser.add_argument("--extension")
    parser.add_argument("--driver", default="PNG")
    parser.add_argument("--resampling", default="bilinear", choices=RESAMPLING_NAMES)
    parser.add_argument("--output-dtype", default="auto", choices=OUTPUT_DTYPES)
    parser.add_argument("--scale-min", type=float)
    parser.add_argument("--scale-max", type=float)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be an integer") from None
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def import_rasterio():
    try:
        import rasterio
        from rasterio.crs import CRS
        from rasterio.enums import Resampling
        from rasterio.windows import from_bounds
    except ModuleNotFoundError as exc:
        raise RuntimeError("Rasterio is required for rendering conflict tiles") from exc
    return rasterio, CRS, Resampling, from_bounds


def validate_paths(args: argparse.Namespace) -> None:
    if not args.master_vrt.exists():
        raise ValueError(f"master VRT does not exist: {args.master_vrt}")
    if not args.master_vrt.is_file():
        raise ValueError(f"master VRT is not a file: {args.master_vrt}")
    if not args.master_pyramid.exists():
        raise ValueError(f"master pyramid does not exist: {args.master_pyramid}")
    if not args.master_pyramid.is_dir():
        raise ValueError(f"master pyramid is not a directory: {args.master_pyramid}")
    if not args.conflicts.exists():
        raise ValueError(f"conflict file does not exist: {args.conflicts}")
    if not args.conflicts.is_file():
        raise ValueError(f"conflicts path is not a file: {args.conflicts}")


def validate_scale_args(args: argparse.Namespace) -> None:
    if (args.scale_min is None) != (args.scale_max is None):
        raise ValueError("--scale-min and --scale-max must be provided together")
    if args.scale_min is not None and args.scale_min >= args.scale_max:
        raise ValueError("--scale-min must be less than --scale-max")


def validate_master_vrt(master_vrt: Path) -> None:
    rasterio, CRS, _, _ = import_rasterio()
    with rasterio.open(master_vrt) as src:
        if src.crs is None:
            raise ValueError("Master VRT has no CRS")
        if src.crs != CRS.from_epsg(3857):
            raise ValueError("Master VRT must be EPSG:3857")


def normalize_extension(extension: str | None) -> str | None:
    if extension is None:
        return None
    normalized = extension.strip().lstrip(".")
    if not normalized:
        raise ValueError("--extension must not be empty")
    if any(sep in normalized for sep in ("/", "\\")):
        raise ValueError("--extension must be a file extension, not a path")
    return normalized.lower()


def parse_conflict_file(conflicts: Path, extension_filter: str | None) -> tuple[list[TileJob], int]:
    jobs_by_path: dict[str, TileJob] = {}
    parsed_count = 0

    with conflicts.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                if line.startswith("# Codex Build Brief"):
                    raise ValueError(
                        f"{conflicts}:{line_number}: this looks like the build brief, "
                        "not a generated conflict tile list. Use conflicts.txt instead."
                    )
                continue

            rel = Path(line)
            if rel.is_absolute() or ".." in rel.parts:
                raise ValueError(f"{conflicts}:{line_number}: conflict path must be relative: {line}")

            parts = rel.parts
            if len(parts) != 3:
                raise ValueError(
                    f"{conflicts}:{line_number}: expected z/x/y.ext path, got: {line}"
                )

            z_str, x_str, filename = parts
            suffix = Path(filename).suffix
            stem = Path(filename).stem
            extension = suffix.lstrip(".").lower()

            if not suffix or not extension:
                raise ValueError(f"{conflicts}:{line_number}: tile path has no extension: {line}")
            if extension_filter is not None and extension != extension_filter:
                continue
            if not z_str.isdigit() or not x_str.isdigit() or not stem.isdigit():
                raise ValueError(
                    f"{conflicts}:{line_number}: z, x, and y must be integers: {line}"
                )

            job = TileJob(
                z=int(z_str),
                x=int(x_str),
                y=int(stem),
                extension=extension,
                relative_path=Path(z_str) / x_str / f"{stem}.{extension}",
            )
            parsed_count += 1
            jobs_by_path[job.relative_path.as_posix()] = job

    return list(jobs_by_path.values()), parsed_count


def xyz_tile_bounds(z: int, x: int, y: int) -> tuple[float, float, float, float]:
    num_tiles = 2**z
    if x < 0 or y < 0 or x >= num_tiles or y >= num_tiles:
        raise ValueError(f"tile coordinate out of range for z={z}: x={x}, y={y}")

    tile_span = (2 * WEB_MERCATOR_HALF_WORLD) / num_tiles
    left = -WEB_MERCATOR_HALF_WORLD + x * tile_span
    right = left + tile_span
    top = WEB_MERCATOR_HALF_WORLD - y * tile_span
    bottom = top - tile_span
    return left, bottom, right, top


def chunks(items: list[TileJob], chunk_size: int):
    for start in range(0, len(items), chunk_size):
        yield tuple(items[start : start + chunk_size])


def resolve_scale_range(args: argparse.Namespace) -> tuple[float, float] | None:
    if args.scale_min is not None and args.scale_max is not None:
        return args.scale_min, args.scale_max

    rasterio, _, _, _ = import_rasterio()
    with rasterio.open(args.master_vrt) as src:
        source_is_float = any(dtype.startswith("float") for dtype in src.dtypes)
        needs_scaling = args.output_dtype in ("uint8", "uint16") or (
            args.output_dtype == "auto"
            and args.driver.upper() == "PNG"
            and source_is_float
        )
        if not needs_scaling:
            return None

        mins: list[float] = []
        maxes: list[float] = []
        sample_height = min(src.height, 2048)
        sample_width = min(src.width, 2048)
        data = src.read(
            out_shape=(src.count, sample_height, sample_width),
            masked=True,
        )
        for band in data:
            if band.count() == 0:
                continue
            mins.append(float(band.min()))
            maxes.append(float(band.max()))

    if not mins or not maxes:
        raise ValueError("could not compute scale range from master VRT")
    scale_min = min(mins)
    scale_max = max(maxes)
    if not math.isfinite(scale_min) or not math.isfinite(scale_max) or scale_min >= scale_max:
        raise ValueError("computed scale range is invalid")
    return scale_min, scale_max


def resampling_value(resampling_name: str) -> Any:
    _, _, Resampling, _ = import_rasterio()
    return getattr(Resampling, resampling_name)


def output_dtype_for(data: Any, requested: str, driver: str, scale_range: tuple[float, float] | None) -> str:
    if requested == "source":
        return str(data.dtype)
    if requested in ("uint8", "uint16"):
        return requested
    if scale_range is not None:
        return "uint8"
    if driver.upper() == "PNG" and str(data.dtype).startswith("float"):
        return "uint8"
    return str(data.dtype)


def prepare_output_data(
    data: Any,
    driver: str,
    output_dtype: str,
    scale_range: tuple[float, float] | None,
):
    import numpy as np

    target_dtype = output_dtype_for(data, output_dtype, driver, scale_range)
    mask = np.ma.getmaskarray(data)
    if mask.ndim == 0:
        invalid = np.zeros(data.shape[1:], dtype=bool)
    elif mask.ndim == 3:
        invalid = np.any(mask, axis=0)
    else:
        invalid = mask

    if target_dtype in ("uint8", "uint16"):
        if scale_range is None and str(data.dtype).startswith("float"):
            raise ValueError("float source data needs --scale-min/--scale-max or --output-dtype source")
        if scale_range is not None:
            scale_min, scale_max = scale_range
            max_value = 255 if target_dtype == "uint8" else 65535
            scaled = (data.astype("float64") - scale_min) / (scale_max - scale_min)
            scaled = np.clip(scaled, 0.0, 1.0) * max_value
            output = np.ma.filled(scaled, 0).round().astype(target_dtype)
        else:
            output = np.ma.filled(data, 0).astype(target_dtype)
    else:
        output = np.ma.filled(data, 0).astype(target_dtype)

    if driver.upper() == "PNG" and output.shape[0] not in (2, 4):
        alpha_dtype = output.dtype
        alpha_max = np.iinfo(alpha_dtype).max if np.issubdtype(alpha_dtype, np.integer) else 1
        alpha = np.where(invalid, 0, alpha_max).astype(alpha_dtype)
        output = np.concatenate([output, alpha[np.newaxis, :, :]], axis=0)

    return output


def write_atomic(output_data: Any, output_path: Path, driver: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")

    rasterio, _, _, _ = import_rasterio()
    profile = {
        "driver": driver,
        "height": output_data.shape[1],
        "width": output_data.shape[2],
        "count": output_data.shape[0],
        "dtype": str(output_data.dtype),
    }

    try:
        with rasterio.open(tmp_path, "w", **profile) as dst:
            dst.write(output_data)
        os.replace(tmp_path, output_path)
    except Exception:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
        raise


def render_tile(
    src: Any,
    job: TileJob,
    master_pyramid: Path,
    tile_size: int,
    driver: str,
    resampling_name: str,
    output_dtype: str,
    scale_range: tuple[float, float] | None,
) -> None:
    _, _, _, from_bounds = import_rasterio()
    bounds = xyz_tile_bounds(job.z, job.x, job.y)
    window = from_bounds(*bounds, transform=src.transform)
    data = src.read(
        window=window,
        out_shape=(src.count, tile_size, tile_size),
        resampling=resampling_value(resampling_name),
        boundless=True,
        masked=True,
    )
    output_data = prepare_output_data(data, driver, output_dtype, scale_range)
    output_path = master_pyramid / str(job.z) / str(job.x) / f"{job.y}.{job.extension}"
    write_atomic(output_data, output_path, driver)


def render_tile_chunk(args: ChunkArgs) -> ChunkResult:
    rasterio, _, _, _ = import_rasterio()
    failures: list[str] = []
    rendered = 0

    try:
        with rasterio.open(args.master_vrt) as src:
            for job in args.jobs:
                try:
                    render_tile(
                        src=src,
                        job=job,
                        master_pyramid=Path(args.master_pyramid),
                        tile_size=args.tile_size,
                        driver=args.driver,
                        resampling_name=args.resampling_name,
                        output_dtype=args.output_dtype,
                        scale_range=args.scale_range,
                    )
                    rendered += 1
                except Exception as exc:
                    failures.append(f"{job.relative_path.as_posix()}: {exc}")
    except Exception as exc:
        for job in args.jobs:
            failures.append(f"{job.relative_path.as_posix()}: {exc}")

    return ChunkResult(rendered=rendered, failures=tuple(failures))


def render_with_workers(args: argparse.Namespace, chunk_args: list[ChunkArgs]) -> tuple[int, list[str]]:
    rendered = 0
    failures: list[str] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(render_tile_chunk, chunk) for chunk in chunk_args]
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as exc:
                failures.append(f"worker failed: {exc}")
                continue
            rendered += result.rendered
            failures.extend(result.failures)
            if args.verbose:
                print(
                    f"rendered chunk: {result.rendered} rendered, {len(result.failures)} failed",
                    flush=True,
                )

    return rendered, failures


def run(args: argparse.Namespace) -> tuple[int, int, int, list[str]]:
    validate_paths(args)
    validate_scale_args(args)
    extension_filter = normalize_extension(args.extension)
    jobs, parsed_count = parse_conflict_file(args.conflicts, extension_filter)

    if args.dry_run:
        return parsed_count, len(jobs), 0, []

    validate_master_vrt(args.master_vrt)
    scale_range = resolve_scale_range(args)
    chunk_args = [
        ChunkArgs(
            master_vrt=str(args.master_vrt),
            master_pyramid=str(args.master_pyramid),
            jobs=job_chunk,
            tile_size=args.tile_size,
            driver=args.driver,
            resampling_name=args.resampling,
            output_dtype=args.output_dtype,
            scale_range=scale_range,
        )
        for job_chunk in chunks(jobs, args.chunk_size)
    ]

    rendered, failures = render_with_workers(args, chunk_args)
    return parsed_count, len(jobs), rendered, failures


def print_header(args: argparse.Namespace, parsed_count: int, unique_count: int) -> None:
    print(f"Master VRT: {args.master_vrt.resolve()}")
    print(f"Master pyramid: {args.master_pyramid.resolve()}")
    print(f"Conflicts: {args.conflicts.resolve()}")
    print(f"Conflict tiles parsed: {parsed_count}")
    print(f"Unique conflict tiles: {unique_count}")
    print(f"Workers: {args.workers}")
    print(f"Chunk size: {args.chunk_size}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        parsed_count, unique_count, rendered, failures = run(args)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print_header(args, parsed_count, unique_count)
    if args.dry_run:
        print("Dry run: true")
        print("Done.")
        return 0

    print("Rendering...")
    print("Done.")
    print(f"Rendered: {rendered}")
    print(f"Failed: {len(failures)}")
    for failure in failures[:20]:
        print(f"failure: {failure}", file=sys.stderr)
    if len(failures) > 20:
        print(f"failure: ... {len(failures) - 20} more", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
