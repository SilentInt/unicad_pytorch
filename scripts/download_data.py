#!/usr/bin/env python3
"""Download ADBench datasets from GitHub.

Usage
-----
    # Download Classical datasets (default)
    uv run python scripts/download_data.py --output data

    # Download specific categories
    uv run python scripts/download_data.py --output data \
        --categories Classical CV_by_ResNet18

    # Download all categories
    uv run python scripts/download_data.py --output data --all

    # Download a single dataset
    uv run python scripts/download_data.py --output data \
        --datasets 2_annthyroid 38_thyroid
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.error
import urllib.request

BASE_URL = "https://raw.githubusercontent.com/Minqi824/ADBench/refs/heads/main/adbench/datasets"

# Embedded dataset file list (from ADBench repo, avoids extra HTTP call)
DATASET_INDEX: dict[str, list[str]] = {
    "Classical": [
        "1_ALOI.npz",
        "2_annthyroid.npz",
        "3_backdoor.npz",
        "4_breastw.npz",
        "5_campaign.npz",
        "6_cardio.npz",
        "7_Cardiotocography.npz",
        "8_celeba.npz",
        "9_census.npz",
        "10_cover.npz",
        "11_donors.npz",
        "12_fault.npz",
        "13_fraud.npz",
        "14_glass.npz",
        "15_Hepatitis.npz",
        "16_http.npz",
        "17_InternetAds.npz",
        "18_Ionosphere.npz",
        "19_landsat.npz",
        "20_letter.npz",
        "21_Lymphography.npz",
        "22_magic.gamma.npz",
        "23_mammography.npz",
        "24_mnist.npz",
        "25_musk.npz",
        "26_optdigits.npz",
        "27_PageBlocks.npz",
        "28_pendigits.npz",
        "29_Pima.npz",
        "30_satellite.npz",
        "31_satimage-2.npz",
        "32_shuttle.npz",
        "33_skin.npz",
        "34_smtp.npz",
        "35_SpamBase.npz",
        "36_speech.npz",
        "37_Stamps.npz",
        "38_thyroid.npz",
        "39_vertebral.npz",
        "40_vowels.npz",
        "41_Waveform.npz",
        "42_WBC.npz",
        "43_WDBC.npz",
        "44_Wilt.npz",
        "45_wine.npz",
        "46_WPBC.npz",
        "47_yeast.npz",
    ],
    "CV_by_ResNet18": [
        "CIFAR10_0.npz",
        "CIFAR10_1.npz",
        "CIFAR10_2.npz",
        "CIFAR10_3.npz",
        "CIFAR10_4.npz",
        "CIFAR10_5.npz",
        "CIFAR10_6.npz",
        "CIFAR10_7.npz",
        "CIFAR10_8.npz",
        "CIFAR10_9.npz",
        "FashionMNIST_0.npz",
        "FashionMNIST_1.npz",
        "FashionMNIST_2.npz",
        "FashionMNIST_3.npz",
        "FashionMNIST_4.npz",
        "FashionMNIST_5.npz",
        "FashionMNIST_6.npz",
        "FashionMNIST_7.npz",
        "FashionMNIST_8.npz",
        "FashionMNIST_9.npz",
        "MNIST-C_brightness.npz",
        "MNIST-C_canny_edges.npz",
        "MNIST-C_dotted_line.npz",
        "MNIST-C_fog.npz",
        "MNIST-C_glass_blur.npz",
        "MNIST-C_identity.npz",
        "MNIST-C_impulse_noise.npz",
        "MNIST-C_motion_blur.npz",
        "MNIST-C_rotate.npz",
        "MNIST-C_scale.npz",
        "MNIST-C_shear.npz",
        "MNIST-C_shot_noise.npz",
        "MNIST-C_spatter.npz",
        "MNIST-C_stripe.npz",
        "MNIST-C_translate.npz",
        "MNIST-C_zigzag.npz",
        "MVTec-AD_bottle.npz",
        "MVTec-AD_cable.npz",
        "MVTec-AD_capsule.npz",
        "MVTec-AD_carpet.npz",
        "MVTec-AD_grid.npz",
        "MVTec-AD_hazelnut.npz",
        "MVTec-AD_leather.npz",
        "MVTec-AD_metal_nut.npz",
        "MVTec-AD_pill.npz",
        "MVTec-AD_screw.npz",
        "MVTec-AD_tile.npz",
        "MVTec-AD_toothbrush.npz",
        "MVTec-AD_transistor.npz",
        "MVTec-AD_wood.npz",
        "MVTec-AD_zipper.npz",
        "SVHN_0.npz",
        "SVHN_1.npz",
        "SVHN_2.npz",
        "SVHN_3.npz",
        "SVHN_4.npz",
        "SVHN_5.npz",
        "SVHN_6.npz",
        "SVHN_7.npz",
        "SVHN_8.npz",
        "SVHN_9.npz",
    ],
    "NLP_by_BERT": [
        "20news_0.npz",
        "20news_1.npz",
        "20news_2.npz",
        "20news_3.npz",
        "20news_4.npz",
        "20news_5.npz",
        "agnews_0.npz",
        "agnews_1.npz",
        "agnews_2.npz",
        "agnews_3.npz",
        "amazon.npz",
        "imdb.npz",
        "yelp.npz",
    ],
}

ALL_CATEGORIES = list(DATASET_INDEX.keys())


def download_file(url: str, dest: str) -> None:
    """Download a file from url to dest with a progress indicator."""
    try:
        urllib.request.urlretrieve(url, dest)
    except urllib.error.URLError as e:
        print(f"  FAILED: {e}")


def download_category(
    output_dir: str, category: str, files: list[str]
) -> tuple[int, int]:
    """Download all files for a category. Returns (ok, failed) counts."""
    dest_dir = os.path.join(output_dir, category)
    os.makedirs(dest_dir, exist_ok=True)

    ok, failed = 0, 0
    for fname in files:
        dest = os.path.join(dest_dir, fname)
        if os.path.isfile(dest):
            ok += 1
            continue

        url = f"{BASE_URL}/{category}/{fname}"
        print(f"  Downloading {category}/{fname} ...", end="", flush=True)
        try:
            download_file(url, dest)
            ok += 1
            print(" OK")
        except Exception as e:
            failed += 1
            print(f" FAILED ({e})")

    return ok, failed


def download_datasets(
    output_dir: str,
    categories: list[str],
    dataset_names: list[str] | None = None,
) -> None:
    """Download datasets by category or by specific name."""
    total_ok, total_fail = 0, 0

    if dataset_names:
        for name in dataset_names:
            fname = f"{name}.npz" if not name.endswith(".npz") else name
            found = False
            for cat, files in DATASET_INDEX.items():
                if fname in files:
                    ok, fail = download_category(output_dir, cat, [fname])
                    total_ok += ok
                    total_fail += fail
                    found = True
                    break
            if not found:
                print(f"  Unknown dataset: {name}")
                total_fail += 1
    else:
        for cat in categories:
            files = DATASET_INDEX.get(cat)
            if files is None:
                print(f"Unknown category: {cat}")
                print(f"Available: {', '.join(ALL_CATEGORIES)}")
                continue
            print(f"\n[{cat}] ({len(files)} datasets)")
            ok, fail = download_category(output_dir, cat, files)
            total_ok += ok
            total_fail += fail
            print(f"  {ok} downloaded, {fail} failed")

    print(f"\nDone: {total_ok} OK, {total_fail} failed")
    if total_fail > 0:
        sys.exit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download ADBench datasets from GitHub"
    )
    parser.add_argument(
        "--output",
        default="data",
        help="Output directory (default: data)",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=["Classical"],
        choices=ALL_CATEGORIES,
        help="Dataset categories to download (default: Classical)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all categories",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Download specific datasets by name (e.g. 2_annthyroid 38_thyroid)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    categories = ALL_CATEGORIES if args.all else args.categories
    download_datasets(
        output_dir=args.output,
        categories=categories,
        dataset_names=args.datasets,
    )


if __name__ == "__main__":
    main()
