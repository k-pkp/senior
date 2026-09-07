"""Segment volume from taped circumferences, by the disc model.

Each measured interval is treated as a truncated cone between the two
circumferences bounding it, and the segment is the sum of those slices. See
docs/measuring_ground_truth.md for the procedure the input comes from and for
the bias this model carries.

Input is a CSV of one row per measurement:

    subject,side,height_cm,circumference_cm
    0,left,0.0,20.5
    0,left,4.0,22.8
    ...

`height_cm` is measured up from the lower band, so the lower band is 0.0 and the
upper band is the largest height for that limb.

    python tools/limb_volume_from_tape.py measurements.csv
"""
import argparse
import csv
import math
from collections import defaultdict


def frustum_volume(lower_circumference, upper_circumference, height):
    """Volume of a truncated cone given the circumferences of its two ends.

    Working from circumference rather than radius keeps the tape's own quantity
    in the formula, with no rounding through a radius on the way:

        V = h (C1^2 + C1 C2 + C2^2) / (12 pi)
    """
    return (height
            * (lower_circumference ** 2
               + lower_circumference * upper_circumference
               + upper_circumference ** 2)
            / (12.0 * math.pi))


def segment_volume(measurements):
    """Total volume for one limb, from its (height, circumference) pairs.

    The pairs are sorted by height first, so the input file does not have to be
    in order. Returns the volume in cm3 and the number of slices summed.
    """
    ordered = sorted(measurements)
    total = 0.0
    for index in range(len(ordered) - 1):
        lower_height, lower_circumference = ordered[index]
        upper_height, upper_circumference = ordered[index + 1]
        total += frustum_volume(lower_circumference, upper_circumference,
                                upper_height - lower_height)
    return total, max(len(ordered) - 1, 0)


def read_measurements(path):
    """Group a measurement CSV into {(subject, side): [(height, girth), ...]}."""
    grouped = defaultdict(list)
    with open(path) as handle:
        for row in csv.DictReader(handle):
            key = (row["subject"].strip(), row["side"].strip())
            grouped[key].append((float(row["height_cm"]),
                                 float(row["circumference_cm"])))
    return grouped


def main():
    """Prints one volume per limb found in the input file."""
    parser = argparse.ArgumentParser(
        description="Segment volume from taped circumferences (disc model).")
    parser.add_argument("csv_path", help="measurement CSV, see module docstring")
    arguments = parser.parse_args()

    grouped = read_measurements(arguments.csv_path)
    if not grouped:
        raise SystemExit("no measurements found")

    print(f'{"subject":9s} {"side":6s} {"slices":>6s} {"span cm":>8s} '
          f'{"volume cm3":>11s}')
    print("-" * 45)
    for (subject, side) in sorted(grouped):
        pairs = grouped[(subject, side)]
        if len(pairs) < 2:
            print(f"{subject:9s} {side:6s} "
                  f"{'—':>6s} {'—':>8s} {'need 2+ heights':>11s}")
            continue
        volume, slice_count = segment_volume(pairs)
        span = max(height for height, _ in pairs) - min(height for height, _ in pairs)
        print(f"{subject:9s} {side:6s} {slice_count:6d} {span:8.1f} {volume:11.1f}")


if __name__ == "__main__":
    main()
