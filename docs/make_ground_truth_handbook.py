"""Render docs/measuring_ground_truth.md as an illustrated PDF handbook.

The diagrams are drawn here rather than photographed, so they stay correct if
the procedure changes.

The one photograph-like figure comes from a rendered run under
compare_old_vs_new/, which is generated output and not kept in the repository.
When it is absent the page still builds, without that figure; to restore it,
re-run the comparison and its renderer. The committed PDF was built while it
existed, so the figure is in there.

That figure is the cut solid from a real reconstruction
from inputs/1_left -- the segment its own cut produced, whose two ends ARE the
bands -- annotated with the heights the tape method would use.

    python docs/make_ground_truth_handbook.py
"""
import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CUT_SHEET = os.path.join(PROJECT_ROOT, "compare_old_vs_new", "renders",
                         "rework", "1_left", "05_cut.png")
OUTPUT = os.path.join(PROJECT_ROOT, "docs", "measuring_ground_truth.pdf")

PAGE = (8.27, 11.69)               # A4 portrait
INK = "#1a1a1a"
MUTED = "#5a5a5a"
ACCENT = "#0b5fa5"
WARN = "#b3261e"
GOOD = "#1e7d32"


def new_page(pdf, title, subtitle=""):
    """Start a page and return its figure, with the heading already drawn."""
    figure = plt.figure(figsize=PAGE)
    figure.text(0.08, 0.955, title, fontsize=17, weight="bold", color=INK)
    if subtitle:
        figure.text(0.08, 0.932, subtitle, fontsize=9.5, color=MUTED)
    return figure


def body_text(figure, top, lines, size=9.5, mono=False):
    """Write a block of wrapped body text, returning the y it ended at."""
    font = {"family": "monospace"} if mono else {}
    figure.text(0.08, top, "\n".join(lines), fontsize=size, color=INK,
                va="top", linespacing=1.55, **font)
    return top - 0.019 * len(lines) * (size / 9.5)


def cover(pdf):
    """Title page."""
    figure = plt.figure(figsize=PAGE)
    figure.text(0.08, 0.72, "Measuring Ground Truth", fontsize=30,
                weight="bold", color=INK)
    figure.text(0.08, 0.675, "Limb segment volume, and how to measure it so that\n"
                             "a disagreement with the pipeline means something",
                fontsize=13, color=MUTED, linespacing=1.5)
    figure.add_artist(plt.Line2D([0.08, 0.92], [0.645, 0.645],
                                 color=ACCENT, linewidth=2))
    figure.text(0.08, 0.60,
                "The short version\n\n"
                "Measure circumferences with a tape and sum truncated cones.\n"
                "Keep water displacement as a cross-check, and only with an\n"
                "overflow spout.",
                fontsize=11, color=INK, va="top", linespacing=1.7)
    figure.text(0.08, 0.10, "VGGT Volume Measurement · docs/measuring_ground_truth.md",
                fontsize=8.5, color=MUTED)
    pdf.savefig(figure)
    plt.close(figure)


def page_requirements(pdf):
    """What a usable ground truth has to satisfy."""
    figure = new_page(pdf, "What has to be true of a ground truth")
    body_text(figure, 0.90, [
        "The pipeline reports a volume. Nothing in it can tell you whether that",
        "volume is right. Only a physical measurement of the same limb can.",
        "",
        "",
        "1.  THE SAME QUANTITY THE PIPELINE REPORTS",
        "",
        "    The pipeline measures either the limb below one marker band or the",
        "    segment between two of them. Those differ by more than a litre. A",
        "    truth measured foot-to-band compared against a run cut band-to-band",
        "    is not a 30% error — it is two different numbers.",
        "",
        "",
        "2.  REPEATABLE",
        "",
        "    A method whose run-to-run spread is larger than the error being",
        "    investigated cannot resolve that error, however careful the",
        "    arithmetic afterwards.",
        "",
        "",
        "3.  A BIAS YOU CAN NAME AND SIZE",
        "",
        "    Every method is biased. A bias you can size is a correction. A bias",
        "    you cannot is an unknown that contaminates every comparison drawn",
        "    against it.",
    ], size=10)
    pdf.savefig(figure)
    plt.close(figure)


def page_disc_model(pdf):
    """The tape method, illustrated on a real reconstruction."""
    figure = new_page(pdf, "Method 1 — circumferences and the disc model",
                      "Preferred. Nothing is immersed, so none of the water "
                      "errors apply.")

    # Left: the real limb segment, with the heights the tape would use.
    #
    # The CUT solid is used rather than the whole limb because its two ends are
    # the bands themselves -- Stage 5 cut it there -- so marks spaced evenly
    # down the image are the 4 cm intervals, with no guessing where the bands
    # fall in the picture.
    axes = figure.add_axes([0.07, 0.34, 0.36, 0.53])
    if os.path.exists(CUT_SHEET):
        sheet = Image.open(CUT_SHEET)
        panel = sheet.crop((0, 26, 420, sheet.height))
        axes.imshow(panel)
        image_height, image_width = panel.height, panel.width
        # The rendered solid does not fill the panel; these are the fractions of
        # the panel its top and bottom occupy in this render.
        top_fraction, bottom_fraction = 0.06, 0.90
        for index in range(8):
            fraction = top_fraction + (bottom_fraction - top_fraction) * index / 7.0
            is_band = index in (0, 7)
            axes.plot([image_width * 0.18, image_width * 0.62],
                      [fraction * image_height] * 2,
                      color=ACCENT if is_band else MUTED,
                      linewidth=2.0 if is_band else 0.9,
                      alpha=0.95)
        axes.text(image_width * 0.64, top_fraction * image_height + 4,
                  "upper band", fontsize=8, color=ACCENT)
        axes.text(image_width * 0.64, bottom_fraction * image_height + 4,
                  "lower band", fontsize=8, color=ACCENT)
    axes.axis("off")
    axes.set_title("the segment between the bands (inputs/1_left)\n"
                   "marked every 4 cm", fontsize=9, color=MUTED)

    # Right: one frustum and its formula.
    axes = figure.add_axes([0.52, 0.44, 0.40, 0.40])
    axes.set_xlim(0, 10)
    axes.set_ylim(0, 10)
    axes.axis("off")
    frustum = Polygon([(3.1, 1.5), (6.9, 1.5), (6.2, 7.5), (3.8, 7.5)],
                      closed=True, facecolor="#e8f0f8", edgecolor=ACCENT,
                      linewidth=1.6)
    axes.add_patch(frustum)
    axes.plot([3.1, 6.9], [1.5, 1.5], color=ACCENT, linewidth=2.4)
    axes.plot([3.8, 6.2], [7.5, 7.5], color=ACCENT, linewidth=2.4)
    axes.text(7.2, 1.4, "C₁", fontsize=12, color=ACCENT)
    axes.text(6.5, 7.4, "C₂", fontsize=12, color=ACCENT)
    axes.add_patch(FancyArrowPatch((2.3, 1.5), (2.3, 7.5),
                                   arrowstyle="<->", color=MUTED, linewidth=1.2,
                                   mutation_scale=12))
    axes.text(1.5, 4.3, "h", fontsize=12, color=MUTED)
    axes.set_title("each interval is a truncated cone", fontsize=9, color=MUTED)

    figure.text(0.52, 0.40, r"$V_{slice}=\dfrac{h\,(C_1^{2}+C_1C_2+C_2^{2})}{12\pi}$",
                fontsize=15, color=INK)

    body_text(figure, 0.30, [
        "PROCEDURE",
        "",
        "  1.  Mark the limb every 4 cm from the lower band to the upper band.",
        "      Mark the skin, so the same heights can be found again.",
        "  2.  Measure the circumference at each mark. Same tape, same tension,",
        "      flat against the skin without compressing it.",
        "  3.  Record each circumference with its height above the lower band.",
        "  4.  Repeat the whole set three times. Keep all three.",
        "",
        "  tools/limb_volume_from_tape.py sums the slices from a CSV.",
        "",
        "WHAT IT GETS WRONG",
        "",
        "  The model treats each cross-section as a circle. A calf is elliptical,",
        "  and a circle of equal circumference encloses more area, so it reads",
        "  high — about 2% at the 1.3 axis ratio measured at these ankles.",
        "  That bias is constant, which is what makes it usable.",
    ], size=9)
    pdf.savefig(figure)
    plt.close(figure)


def page_displacement(pdf):
    """The water method, right and wrong, side by side."""
    figure = new_page(pdf, "Method 2 — water displacement",
                      "Cross-check only. One gram of water is 1.00 mL to within "
                      "0.3% at room temperature.")

    def draw_tub(axes, spout, title, colour):
        """A tub of water, with or without an overflow spout."""
        axes.set_xlim(0, 10)
        axes.set_ylim(0, 10)
        axes.axis("off")
        axes.add_patch(Rectangle((1.2, 1.6), 5.6, 5.2, facecolor="none",
                                 edgecolor=INK, linewidth=1.8))
        water_top = 6.3 if spout else 5.4
        axes.add_patch(Rectangle((1.25, 1.65), 5.5, water_top - 1.65,
                                 facecolor="#cfe3f5", edgecolor="none"))
        axes.plot([1.25, 6.75], [water_top, water_top], color=ACCENT, linewidth=1.6)
        if spout:
            axes.plot([6.8, 8.3], [6.3, 6.3], color=INK, linewidth=1.8)
            axes.plot([8.3, 8.3], [6.3, 5.0], color=INK, linewidth=1.8)
            axes.add_patch(Rectangle((7.6, 3.0), 1.5, 1.9, facecolor="#cfe3f5",
                                     edgecolor=INK, linewidth=1.4))
            axes.text(7.4, 2.5, "catch vessel\non the scale", fontsize=7.5,
                      color=MUTED)
            axes.text(6.9, 6.6, "spout", fontsize=8, color=INK)
        else:
            axes.annotate("", xy=(6.8, 6.8), xytext=(6.8, 5.4),
                          arrowprops=dict(arrowstyle="<->", color=WARN, lw=1.4))
            axes.text(7.0, 5.9, "unknown\nshortfall", fontsize=8, color=WARN)
        axes.set_title(title, fontsize=9.5, color=colour, weight="bold")

    draw_tub(figure.add_axes([0.07, 0.55, 0.40, 0.33]), True,
             "RIGHT — spout, catch the overflow", GOOD)
    draw_tub(figure.add_axes([0.53, 0.55, 0.40, 0.33]), False,
             "WRONG — fill to the rim, re-weigh the tub", WARN)

    body_text(figure, 0.51, [
        "PROCEDURE",
        "",
        "  1.  Use a container with an overflow spout at a fixed height.",
        "  2.  Fill until the spout stops dripping on its own. This is the only",
        "      way to put the starting level at a defined point, and it is the",
        "      step that matters most.",
        "  3.  Put the catch vessel on the scale and tare it.",
        "  4.  Lower the limb until the water surface sits at the marker band.",
        "      Hold still — muscle contraction changes limb volume.",
        "  5.  Wait for the spout to stop dripping.",
        "  6.  Weigh the caught water. Grams read as cm³.",
        "",
        "  For a segment, measure to each band separately and subtract.",
        "",
        "",
        "WHY THE CATCH VESSEL, NOT THE TUB",
        "",
        "  Re-weighing the tub counts the film of water the limb carries out:",
        "",
        "      W₁ − W₂  =  V_displaced  +  V_film",
        "",
        "  A wet calf carries 20–50 g. That is 1–2% of a 2100 cm³ measurement",
        "  to the upper band, but 7–17% of a 300 cm³ measurement to the ankle —",
        "  which is why the below-lower-band column is the least trustworthy",
        "  in inputs/groundtruth0-6.csv.",
        "",
        "  Catching the overflow avoids it: the spill happens during immersion,",
        "  the film leaves afterwards.",
    ], size=9)
    pdf.savefig(figure)
    plt.close(figure)


def page_errors(pdf):
    """The two error terms, sized."""
    figure = new_page(pdf, "The errors, sized",
                      "Both were derived from this project's own measurements "
                      "on 2026-09-04.")

    axes = figure.add_axes([0.10, 0.58, 0.80, 0.28])
    shortfall_mm = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
    lost_cm3 = shortfall_mm * 100.0
    axes.plot(shortfall_mm, lost_cm3, color=WARN, linewidth=2.2, marker="o",
              markersize=4)
    axes.axhline(2100 * 0.25, color=MUTED, linestyle="--", linewidth=1)
    axes.text(0.15, 2100 * 0.25 + 25, "25% of a 2100 cm³ reading",
              fontsize=8, color=MUTED)
    axes.set_xlabel("container filled this many mm below the overflow point",
                    fontsize=9)
    axes.set_ylabel("volume never spilled (cm³)", fontsize=9)
    axes.set_title("Fill-level shortfall — the dominant error without a spout",
                   fontsize=10, color=INK)
    axes.grid(alpha=0.25)
    for spine in ("top", "right"):
        axes.spines[spine].set_visible(False)

    body_text(figure, 0.50, [
        "A container wide enough for a leg has roughly a 1000 cm² water surface,",
        "so every millimetre below the overflow point is 100 cm³ that never",
        "spills and is invisible to the scale. Without a spout, being 5 mm out is",
        "easy: surface tension lets water stand proud of a rim before it breaks,",
        "then it runs and settles somewhere below. The starting point is both",
        "unknown and different every time.",
        "",
        "",
        "SCALE TILT — real, but too small to be the cause",
        "",
        "  A load cell reads the force component along its own axis, so a tilted",
        "  scale scales both weighings, and their difference, by about cos θ:",
        "",
        "      tilt   5°   →   reading  −0.4%",
        "      tilt  10°   →   reading  −1.5%",
        "      tilt  20°   →   reading  −6.0%",
        "      tilt  41°   →   reading   −25%",
        "",
        "  Explaining a 25% discrepancy by tilt alone needs about 41°, which you",
        "  would not miss. Level the scale and stop thinking about it.",
        "",
        "",
        "TWO CHECKS WORTH RUNNING ONCE",
        "",
        "  Weigh a known volume. Pour a measured 2000 mL from a graduated flask",
        "  into the catch vessel. If the scale reads 2000 g, the scale and its",
        "  levelling are no longer suspects. Two minutes.",
        "",
        "  Repeat one limb three times without changing anything. Close agreement",
        "  means a constant bias; scatter means the setup moves between",
        "  measurements, and that has to be fixed first.",
    ], size=9)
    pdf.savefig(figure)
    plt.close(figure)


def page_recording(pdf):
    """The CSV schema, with the two new columns argued for."""
    figure = new_page(pdf, "What to record",
                      "One row per limb, so the comparison scripts read it "
                      "directly.")
    body_text(figure, 0.90, [
        "  no                        subject number",
        "  sex                       M or W",
        "  <side>_low_delta          displaced volume to the LOWER band, cm³",
        "  <side>_up_delta           displaced volume to the UPPER band, cm³",
        "  <side>_vol                segment between bands = up_delta − low_delta",
        "  band_<side>_lower         taped girth at the lower band, cm",
        "  band_<side>_upper         taped girth at the upper band, cm",
        "  band_<side>_separation    tape distance between the bands, cm    ← new",
        "  <side>_tape_vol           disc-model volume between the bands    ← new",
    ], size=9.5, mono=True)

    body_text(figure, 0.70, [
        "Both new columns earn their place.",
        "",
        "band_<side>_separation is the one measurement that lets plane placement",
        "be checked without trusting the reconstruction: the pipeline reports",
        "where it cut, and a tape says how far apart the bands actually are.",
        "Without it, a cut in the wrong place and a limb reconstructed the wrong",
        "size look identical in the volume.",
        "",
        "<side>_tape_vol carries the disc-model result alongside the displacement",
        "one, so the two methods sit on the same limb rather than one silently",
        "replacing the other.",
        "",
        "",
        "Record the full circumference profile as well — every height, every",
        "repeat, in a separate file:",
        "",
        "    subject,side,height_cm,circumference_cm",
        "    0,left,0.0,20.5",
        "    0,left,4.0,22.8",
        "",
        "The summary row throws away exactly the part that localises error.",
    ], size=9.5)
    pdf.savefig(figure)
    plt.close(figure)


def page_open_question(pdf):
    """The unresolved discrepancy, with the ring-coverage evidence."""
    figure = new_page(pdf, "Open question, as of 2026-09-04",
                      "Two mechanisms, pointing opposite ways. Neither is "
                      "isolated yet.")

    axes = figure.add_axes([0.12, 0.60, 0.76, 0.26])
    coverage = np.array([94, 89, 72, 58, 26])
    girth_error = np.array([5.3, 11.3, 14.1, 12.8, 46.6])
    labels = ["1_left", "6_left", "1_right", "0_right", "5_right"]
    axes.scatter(coverage, girth_error, s=70, color=WARN, zorder=3)
    for ring_coverage, over_read, label in zip(coverage, girth_error, labels):
        axes.annotate(label, (ring_coverage, over_read),
                      textcoords="offset points", xytext=(6, 5),
                      fontsize=8, color=MUTED)
    axes.set_xlabel("ellipse-fit ring coverage at the upper plane (%)", fontsize=9)
    axes.set_ylabel("girth over-read (%)", fontsize=9)
    axes.set_title("The pipeline over-reads girth where the ring is incomplete",
                   fontsize=10, color=INK)
    axes.grid(alpha=0.25)
    axes.invert_xaxis()
    for spine in ("top", "right"):
        axes.spines[spine].set_visible(False)

    body_text(figure, 0.52, [
        "Fourteen captures were measured against inputs/groundtruth0-6.csv. The",
        "pipeline read +24.6% to +35.8% high on every capture that produced a",
        "sound two-plane cut. Two mechanisms have been identified:",
        "",
        "",
        "  THE PIPELINE OVER-READS GIRTH where the reconstruction ring is",
        "  incomplete. Circumference is an ellipse fitted to a ring of surface",
        "  points, and where the back of the calf did not reconstruct, the fit",
        "  extrapolates across the gap. Volume goes as girth squared, so +11%",
        "  girth is roughly +23% volume.",
        "",
        "  THE DISPLACEMENT METHOD UNDER-READS if the container is not at its",
        "  overflow point, by 100 cm³ per millimetre of shortfall.",
        "",
        "",
        "Neither has been isolated. Until one is, the difference between the two",
        "columns cannot be attributed to either side, and quoting it as a",
        "pipeline error overstates what is known.",
        "",
        "The tape profile settles it. It shares no error mechanism with either —",
        "no water, no reconstruction — so it can adjudicate at every height",
        "rather than only at the end.",
    ], size=9.5)
    pdf.savefig(figure)
    plt.close(figure)


def main():
    """Write the handbook PDF."""
    with PdfPages(OUTPUT) as pdf:
        cover(pdf)
        page_requirements(pdf)
        page_disc_model(pdf)
        page_displacement(pdf)
        page_errors(pdf)
        page_recording(pdf)
        page_open_question(pdf)
    print(OUTPUT)


if __name__ == "__main__":
    main()
