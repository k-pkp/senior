"""Open-vocabulary detection and segmentation for Stage 0.

Stage 0 has to find three things in a photograph: the reference cube, the limb
being measured, and the band that marks where the measurement stops. Only the
first is a geometry problem. The cube carries printed markers, so its silhouette
is recoverable exactly and no model is involved -- see pipeline/stages/prep.py.

The other two were colour and class rules, and both were brittle for the same
reason: they encoded one particular capture. The limb came from a COCO `person`
mask, which contains the whole body and has to be narrowed by a heuristic, and
the band came from `2G - R - B > 10`, which is tuned to one khaki band, fires on
houseplants, and silently stops working if the marker is red or blue.

Describing them in words instead removes both limitations. GroundingDINO locates
"a human leg" and "a rubber band on the leg" without knowing the classes in
advance; SAM turns the limb's box into a mask of the limb itself rather than of
the person it belongs to. Measured on inputs/small_leg the band is found on 6 of
6 frames at 0.81-0.84 confidence, including the two frames where the colour rule
failed outright.

It also makes the colour rule unnecessary rather than merely better tuned. Once
the band is located without reference to its colour, its colour can be read off
and handed to Stage 3, which is where the cut actually happens. On
inputs/small_leg that recovers an excess-green of +14 -- exactly the value
MARKER_EXG_MIN was hand-tuned to, but derived from the image.
"""
import numpy as np

DINO_MODEL = "IDEA-Research/grounding-dino-tiny"
SAM_MODEL = "facebook/sam-vit-base"

# Text queries. GroundingDINO expects lowercase phrases separated by periods.
LEG_PROMPT = "a human leg."
BAND_PROMPT = "a rubber band on the leg."
# Used only to widen the marker-derived cube box, never alone: measured
# against the face union it under-covers the true silhouette on every
# frame, by 22 to 90 px.
BOX_PROMPT = "a box."

# Detection confidence. 0.3 is comfortably below the 0.81-0.84 the band scores
# on inputs/small_leg and above the noise floor.
DETECT_THRESHOLD = 0.3

_CACHE = {}


def _models(device="cuda"):
    """Load once and reuse: a six-frame set would otherwise pay six times."""
    if "dino" not in _CACHE:
        import torch
        from transformers import (AutoProcessor,
                                  AutoModelForZeroShotObjectDetection,
                                  SamModel, SamProcessor)
        _CACHE["torch"] = torch
        _CACHE["dino_proc"] = AutoProcessor.from_pretrained(DINO_MODEL)
        _CACHE["dino"] = AutoModelForZeroShotObjectDetection.from_pretrained(
            DINO_MODEL).to(device).eval()
        _CACHE["sam_proc"] = SamProcessor.from_pretrained(SAM_MODEL)
        _CACHE["sam"] = SamModel.from_pretrained(SAM_MODEL).to(device).eval()
    return _CACHE


def release():
    """Drop the models and free their VRAM before VGGT needs it."""
    torch = _CACHE.get("torch")
    _CACHE.clear()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def detect(image_pil, prompt, threshold=DETECT_THRESHOLD, device="cuda"):
    """Best box for `prompt`, as [x0, y0, x1, y1], or None."""
    m = _models(device)
    torch = m["torch"]
    inputs = m["dino_proc"](images=image_pil, text=prompt,
                            return_tensors="pt").to(device)
    with torch.no_grad():
        out = m["dino"](**inputs)
    res = m["dino_proc"].post_process_grounded_object_detection(
        out, inputs.input_ids, threshold=threshold, text_threshold=threshold,
        target_sizes=[image_pil.size[::-1]])[0]
    if not len(res["scores"]):
        return None, 0.0
    i = int(res["scores"].argmax())
    return [float(v) for v in res["boxes"][i]], float(res["scores"][i])


def segment(image_pil, box, device="cuda"):
    """Mask of whatever `box` encloses, as a boolean array."""
    m = _models(device)
    torch = m["torch"]
    inputs = m["sam_proc"](image_pil, input_boxes=[[list(map(float, box))]],
                           return_tensors="pt").to(device)
    with torch.no_grad():
        out = m["sam"](**inputs, multimask_output=False)
    mask = m["sam_proc"].image_processor.post_process_masks(
        out.pred_masks.cpu(), inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu())[0][0][0].numpy()
    return mask.astype(bool)


def trace_band_colour(bgr, box, keep_percentile=40, dilate=3):
    """The band's own colour, traced rather than averaged.

    The band is a thin cord lying diagonally across its detected box and fills
    only a few percent of it, so any statistic over the box returns the limb --
    averaging it reports skin, and taking the pixels least like skin reports
    shadow. Both were tried and both were wrong.

    What is true of the cord is that it crosses every column of the box exactly
    once. So for each column, the pixel departing furthest from that column's
    own median is the band, and the trace follows it across regardless of what
    colour it happens to be.

    `dilate` is why the trace reports a colour the reconstruction actually
    contains. The furthest-departing pixel is the cord's most extreme one --
    its darkest, most shadowed core -- whereas a 3D point takes the colour of
    whatever cord pixel its ray happened to land on, which is a typical one.
    Measured on small_leg: the argmax trace reports RGB(37,30,9) while the
    band's points in the cloud sit at RGB(70,62,36). Calibrating the detector's
    contrast axis on the extreme then put the real band at 0.30 along an axis
    thresholded at 0.50, so 85% of it was discarded and the surviving sliver
    was too thin to fit a plane through -- the cut came out 21 degrees off
    perpendicular to the limb. Sampling a few rows either side of the traced
    cord reports its body instead, and the same cut lands 2 degrees off.
    """
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    roi = bgr[max(0, y0):y1, max(0, x0):x1].astype(np.float32)
    if roi.size == 0 or roi.shape[0] < 3 or roi.shape[1] < 3:
        return None
    column_median = np.median(roi, axis=0, keepdims=True)
    deviation = np.linalg.norm(roi - column_median, axis=2)
    rows = deviation.argmax(axis=0)
    cols = np.arange(roi.shape[1])
    strength = deviation[rows, cols]
    # Columns where the cord leaves the limb carry no real maximum; drop them.
    solid = strength > np.percentile(strength, keep_percentile)
    if int(solid.sum()) < 8:
        return None
    body = []
    for offset in range(-dilate, dilate + 1):
        rr = np.clip(rows + offset, 0, roi.shape[0] - 1)
        body.append(roi[rr, cols][solid])
    keep = np.vstack(body)
    bgr_med = np.median(keep, axis=0)
    b, g, r = (float(v) for v in bgr_med)
    # The limb's own colour, from the same columns. What separates a band from
    # a limb is the CONTRAST between them, not the band's absolute colour --
    # this khaki band sits at hue 26 and skin at hue 11-20, so a hue window
    # centred on the band admits the whole limb. Recording both lets the
    # detector learn the direction that actually separates them.
    limb_med = np.median(column_median.reshape(-1, 3), axis=0)
    lb, lg, lr = (float(v) for v in limb_med)
    import cv2
    hsv = cv2.cvtColor(np.uint8([[bgr_med]]), cv2.COLOR_BGR2HSV)[0][0]
    return {
        "limb_bgr": [round(lb, 1), round(lg, 1), round(lr, 1)],
        "limb_rgb": [round(lr, 1), round(lg, 1), round(lb, 1)],
        "bgr": [round(b, 1), round(g, 1), round(r, 1)],
        "rgb": [round(r, 1), round(g, 1), round(b, 1)],
        "hsv": [int(hsv[0]), int(hsv[1]), int(hsv[2])],
        "exg": round(2 * g - r - b, 1),
        "n_px": int(len(keep)),
    }
