#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse

from gr00t.eval.progress_curve_plot import load_progress_curve_rows, write_progress_curve_plot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", required=True)
    parser.add_argument("--png-path", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--success-only", action="store_true")
    args = parser.parse_args()

    rows = load_progress_curve_rows(args.csv_path, success_only=args.success_only)
    write_progress_curve_plot(rows, png_path=args.png_path, target=args.target)


if __name__ == "__main__":
    main()
