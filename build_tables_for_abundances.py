"""Convenience script to regenerate EOS/opacity tables for chosen oxygen abundances."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import saha_eos as eos
import opac


def build_tables_for_abundance(A_O_log: float,
                               output_dir: Path,
                               show_plot: bool = False) -> Tuple[Path, Path]:
    """Generate saha_eos/opacity FITS tables for a single abundance value."""

    suffix = f"{A_O_log:.2f}".replace('.', 'p')
    eos_path = output_dir / f"saha_eos_AO{suffix}.fits"
    opacity_path = output_dir / f"Ross_Planck_opac_AO{suffix}.fits"

    print(f"\n=== Building tables for log A(O) = {A_O_log:.2f} ===")
    eos.P_T_tables(None, None, savefile=str(eos_path), A_O_log=A_O_log)
    opac.generate_opacity_tables(eos_filename=str(eos_path),
                                 outfile=str(opacity_path),
                                 show_plot=show_plot,
                                 A_O_log=A_O_log)

    print(f"Saved EOS table to {eos_path}")
    print(f"Saved opacity table to {opacity_path}")
    return eos_path, opacity_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("abundances",
                        nargs='+',
                        type=float,
                        help="List of log A(O) abundances to process")
    parser.add_argument("--output-dir",
                        default=".",
                        type=Path,
                        help="Directory where FITS files will be written (default: current directory)")
    parser.add_argument("--show-plot",
                        action="store_true",
                        help="Display diagnostic plots while building the opacity table")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    for abundance in args.abundances:
        build_tables_for_abundance(abundance, output_dir, show_plot=args.show_plot)


if __name__ == "__main__":
    main()
