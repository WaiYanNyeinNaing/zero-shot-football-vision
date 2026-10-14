"""Decisions questions and passes: classify, resolve, force labels, ball search, render, export.

Four output labels only: ball, referee, team_1_player, team_2_player. Anything
forced to choose is tagged forced=True.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict

import cv2
import numpy as np

from config_io import (CLASSES, DECISIONS_ENDPOINT, DOC_URL, BudgetExhausted, Config,
                       DecisionClient, image_part)
from detect import _iou, pitch_mask


def q_object() -> list[dict]:
    return [{"type": "choice", "name": "object_class",
             "instructions": "You were shown a TEAM 1 reference image, a TEAM 2 reference "
                             "image, and (when available) a REFEREE reference image, then the "
                             "crop to label. The subject to label is the person or object at the "
                             "CENTRE of the final crop - ignore anyone at its edges. Match the "
                             "centre subject's shirt against the reference kits. Which class "
                             "fits it?",
             "choices": [
                 {"value": "team_1_player", "description": "Outfield player whose shirt matches "
                                                           "the TEAM 1 reference image."},
                 {"value": "team_2_player", "description": "Outfield player whose shirt matches "
                                                           "the TEAM 2 reference image."},
                 {"value": "ball", "description": "The match ball is the centre subject."},
                 {"value": "referee", "description": "Match official - matches the REFEREE "
                                                "reference kit, or matches neither team kit "
                                                "and is dressed as an official."},
                 {"value": "background", "description": "Crowd, pitch, HUD or no subject."},
                 {"value": "uncertain", "description": "A subject is there but the kit or class "
                                                       "cannot be told from this crop."},
             ]}]


def q_ball_box() -> list[dict]:
    """Confirm a tight box. Deliberately NOT q_object: no team references are
    sent here, and q_object's instructions promise two reference images."""
    return [{"type": "predicate", "name": "is_ball",
             "instructions": "Is the object at the exact centre of this crop the match ball - "
                             "a small, free-standing, roughly spherical ball resting on or "
                             "moving over the grass, clearly separate from any player? "
                             "Answer no if the centre is part of a player (shirt, shorts, socks, "
                             "boots, skin or hair), a pitch marking, a goal net or post, a "
                             "shadow, or a HUD graphic. Measured failure on this clip: a white "
                             "shirt torso was wrongly accepted, so check that the centre object "
                             "is detached from any body before answering yes."}]


def q_resolve() -> list[dict]:
    """Second pass over detections the 6-way question could not settle.

    Measured cause of those: the 6-way `choice` splits probability mass so the
    true class lands around 0.4-0.5 and never clears the bar, even when its own
    probabilities already lean the right way. This asks only the four labels
    that matter, on a larger crop, so the mass concentrates. It adds a sharper
    question and better evidence - it does NOT lower the threshold.
    """
    return [{"type": "choice", "name": "resolved_class",
             "instructions": "References were shown first: TEAM 1 kit, TEAM 2 kit, and the "
                             "REFEREE kit. The final image is one player, enlarged, at the "
                             "centre. Ignore anyone at the edges. Which of these four is the "
                             "centre subject?",
             "choices": [
                 {"value": "team_1_player", "description": "Shirt matches the TEAM 1 reference."},
                 {"value": "team_2_player", "description": "Shirt matches the TEAM 2 reference."},
                 {"value": "referee", "description": "Match official - matches the REFEREE "
                                                     "reference, or matches neither team."},
                 {"value": "not_a_player", "description": "Not a player at all: crowd, dugout, "
                                                          "stadium screen, HUD or empty pitch."},
             ]}]


def q_region() -> list[dict]:
    return [{"type": "predicate", "name": "ball_present",
             "instructions": "Is the match ball visible anywhere inside this image region? "
                             "The ball may be small, blurred or partly hidden. Do not count "
                             "pitch markings, white boots, socks or HUD graphics."}]


# --------------------------------------------------------------------------
# step 6 - classify object proposals with Decisions
# --------------------------------------------------------------------------
def context_crop(img: np.ndarray, box: list[int], pad: float = 0.25,
                 min_h: int = 192) -> np.ndarray:
    """Crop with a little context, then upscale if it is tiny.

    Players are often ~20x45px here. Sending that raw asks the model to read a
    kit off a postage stamp; cubic upscaling adds no information but makes the
    colour and shirt pattern legible.
    """
    x0, y0, x1, y1 = box
    px, py = int((x1 - x0) * pad) + 4, int((y1 - y0) * pad) + 4
    h, w = img.shape[:2]
    crop = img[max(y0 - py, 0):min(y1 + py, h), max(x0 - px, 0):min(x1 + px, w)]
    if crop.size and crop.shape[0] < min_h:
        f = min_h / crop.shape[0]
        crop = cv2.resize(crop, (max(int(crop.shape[1] * f), 1), min_h),
                          interpolation=cv2.INTER_CUBIC)
    return crop


def classify_objects(cfg: Config, dc: DecisionClient, frames: list[dict],
                     proposals: list[dict], kits: dict) -> list[dict]:
    by_id = {f["frame_id"]: f for f in frames}
    refs = kits.get("refs")
    dets = []
    for p in proposals:
        crop = context_crop(by_id[p["frame_id"]]["image"], p["box"])
        if crop.size == 0:
            continue
        det = {**p, "decision_type": "object", "localization": "tight",
               "cls": "uncertain", "confidence": None, "probabilities": None,
               "uncertain_reason": None, "parent_id": None}
        if not kits.get("resolved") or not refs:
            det["uncertain_reason"] = "no frozen team reference"
            dets.append(det)
            continue
        try:
            content = [{"type": "input_text", "text": "TEAM 1 reference kit:"},
                       image_part(refs[0]),
                       {"type": "input_text", "text": "TEAM 2 reference kit:"},
                       image_part(refs[1])]
            if kits.get("referee_ref") is not None:
                content += [{"type": "input_text", "text": "REFEREE reference kit:"},
                            image_part(kits["referee_ref"])]
            content += [{"type": "input_text", "text": "Crop to label (subject at the centre):"},
                        image_part(crop)]
            r = dc.ask(content, q_object(), tag=f"obj:{p['proposal_id']}")
        except BudgetExhausted as exc:
            det["uncertain_reason"] = f"budget stop: {exc}"
            dets.append(det)
            continue
        a = next((x for x in r.get("answers", []) if x["name"] == "object_class"), None)
        if not a:
            det["uncertain_reason"] = r.get("error") or f"no answer ({r['source']})"
        elif a["type"] == "refusal":
            det["uncertain_reason"] = "model refusal"
        else:
            det["probabilities"] = a.get("probabilities")
            det["confidence"] = a.get("confidence")
            if (a.get("confidence") or 0) >= cfg.t_object_accept:
                det["cls"] = a["choice"]
            else:
                det["uncertain_reason"] = f"confidence {a.get('confidence')} < {cfg.t_object_accept}"
        det["decision_source"] = r["source"]
        dets.append(det)
    return dets


# --------------------------------------------------------------------------
# step 7 - ball: coarse regions -> refine -> localize -> validate
# --------------------------------------------------------------------------
def _ball_cue(img: np.ndarray, box: list[int], pitch: np.ndarray) -> float:
    """Small-bright-blob energy, not mean whiteness.

    Mean whiteness ranks a white SHIRT above a 5px ball, which is how this
    search originally missed balls that were plainly visible. A top-hat keeps
    only structures smaller than its kernel, so players drop out and balls stay.
    """
    x0, y0, x1, y1 = box
    patch = img[y0:y1, x0:x1]
    if patch.size == 0:
        return 0.0
    sub = pitch[y0:y1, x0:x1]
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    th = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    th[sub == 0] = 0
    return float(th.astype(np.float32).max() * (th > th.mean() + 2 * th.std()).mean())


def ball_regions(cfg: Config, frame: dict, mask: np.ndarray,
                 scales=(0.17,), stride: float = 0.5) -> tuple[list[dict], int]:
    """Overlapping grid over the PITCH, spread out rather than cue-clustered.

    Two lessons are baked in here. (1) Restrict to the pitch - crowd regions can
    only contribute false positives. (2) Pure cue ranking clusters every pick
    into one corner; measured on earlier test footage, the ball's own location was simply
    never offered to the model in 2 of 5 frames. So we greedily take the best
    remaining region that does not mostly overlap what we already took, which
    buys coverage instead of redundancy.

    YOLO and colour cues only *prioritize*. They never gate the search.
    """
    img = frame["image"]
    pitch = pitch_mask(img, mask)
    frame["pitch_fraction"] = round(float((pitch > 0).mean()), 4)
    h, w = img.shape[:2]
    cands = []
    for sc in scales:
        side = int(sc * h)
        step = max(int(side * stride), 1)
        for y in range(0, max(h - side, 0) + 1, step):
            for x in range(0, max(w - side, 0) + 1, step):
                box = [x, y, min(x + side, w), min(y + side, h)]
                sub = pitch[box[1]:box[3], box[0]:box[2]]
                if sub.size == 0 or (sub > 0).mean() < 0.35:
                    continue
                cands.append({"box": box, "scale": round(sc, 3),
                              "cue": round(_ball_cue(img, box, pitch), 4)})
    cands.sort(key=lambda c: -c["cue"])

    kept: list[dict] = []
    for c in cands:                      # greedy spread: cue order, overlap-limited
        if len(kept) >= cfg.max_ball_regions_per_frame:
            break
        if any(_iou(c["box"], k["box"]) > 0.35 for k in kept):
            continue
        kept.append(c)
    for c in cands:                      # top up with whatever is left
        if len(kept) >= cfg.max_ball_regions_per_frame:
            break
        if c not in kept:
            kept.append(c)
    for i, c in enumerate(kept):
        c.update({"region_id": f"{frame['frame_id']}_r{i:02d}", "frame_id": frame["frame_id"],
                  "depth": 0, "parent_id": None, "proposal_source": "pitch_grid+tophat_cue"})
    return kept, len(cands) - len(kept)


def _split(region: dict) -> list[dict]:
    x0, y0, x1, y1 = region["box"]
    mw, mh = (x1 - x0) // 2, (y1 - y0) // 2
    ox, oy = mw // 4, mh // 4                      # overlap so a ball on a seam survives
    out = []
    for j, (ax, ay) in enumerate([(0, 0), (1, 0), (0, 1), (1, 1)]):
        bx0 = max(x0 + ax * mw - ox, x0)
        by0 = max(y0 + ay * mh - oy, y0)
        out.append({"box": [bx0, by0, min(bx0 + mw + ox, x1), min(by0 + mh + oy, y1)],
                    "region_id": f"{region['region_id']}_{j}", "frame_id": region["frame_id"],
                    "depth": region["depth"] + 1, "parent_id": region["region_id"],
                    "scale": region["scale"], "cue": region["cue"],
                    "proposal_source": "subdivision"})
    return out


def ball_candidates(img: np.ndarray, region: dict, top: int = 2) -> list[list[int]]:
    """Tight box from local image processing inside a positive region.

    A fixed "white" threshold does not work: on earlier test footage the ball's V runs
    ~128-214 while the surrounding grass sits at ~113, so any global cutoff
    either misses it or floods. We threshold **relative to this patch** -
    brighter than median + 1.2 sigma and less saturated than the local median -
    then keep the roundest, best-filled blob of plausible size.

    The region decision never produced coordinates; this is where they come from.
    """
    x0, y0, x1, y1 = region["box"]
    patch = img[y0:y1, x0:x1]
    if patch.size == 0 or patch.shape[0] < 6 or patch.shape[1] < 6:
        return None
    npx = patch.shape[0] * patch.shape[1]

    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    V = hsv[:, :, 2].astype(np.float32)
    S = hsv[:, :, 1].astype(np.float32)
    m = ((V > np.median(V) + 1.2 * V.std()) & (S < np.median(S))).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    scored = []
    for c in cnts:
        a = cv2.contourArea(c)
        if a < 6 or a > 0.25 * npx:            # too small to be real, too big to be a ball
            continue
        bx, by, bw, bh = cv2.boundingRect(c)
        if bw == 0 or bh == 0:
            continue
        ar = min(bw, bh) / max(bw, bh)         # roundness
        fill = a / (bw * bh)                   # solidity
        if ar > 0.55 and fill > 0.40:
            scored.append((ar * fill,
                           [x0 + bx, y0 + by, x0 + bx + bw, y0 + by + bh]))
    scored.sort(key=lambda t: -t[0])
    return [b for _, b in scored[:top]]


def find_ball(cfg: Config, dc: DecisionClient, frames: list[dict],
              mask: np.ndarray) -> tuple[list[dict], list[dict]]:
    """Coarse region -> localize inside it -> confirm the tight box.

    Subdivision is the *fallback*, not the default. Measured on earlier test footage: a
    coarse 121px region scored ball_present=0.99, but its own 75px child
    containing that same ball scored 0.19 - cropping tighter removes the scale
    context the model needs to tell a 5px ball from a speck. So when a region
    is positive we localize inside it immediately, and only subdivide when
    local image processing offers no candidate.
    """
    dets, regions_log = [], []
    for f in frames:
        img = f["image"]
        queue, omitted = ball_regions(cfg, f, mask)
        f["ball_regions_dropped_by_budget"] = omitted
        while queue:
            reg = queue.pop(0)
            crop = img[reg["box"][1]:reg["box"][3], reg["box"][0]:reg["box"][2]]
            if crop.size == 0:
                continue
            try:
                r = dc.ask([{"type": "input_text",
                             "text": "Image region from a football video frame."},
                            image_part(crop)], q_region(), tag=f"region:{reg['region_id']}")
            except BudgetExhausted as exc:
                reg.update({"verdict": "unsearched", "note": str(exc)})
                regions_log.append(reg)
                regions_log.extend({**q, "verdict": "unsearched"} for q in queue)
                break
            a = next((x for x in r.get("answers", []) if x["name"] == "ball_present"), None)
            p = a.get("probability") if a and a["type"] == "predicate" else None
            reg.update({"ball_p": p, "decision_type": "region", "decision_source": r["source"],
                        "error": r.get("error")})
            # unknown != absent: anything we could not settle stays unknown
            reg["verdict"] = ("ball_present" if p is not None and p >= cfg.t_region_ball
                              else "unknown" if p is None else "ball_absent")
            regions_log.append(reg)
            if reg["verdict"] != "ball_present":
                continue

            det_base = {"frame_id": f["frame_id"], "proposal_id": reg["region_id"],
                        "parent_id": reg["parent_id"], "proposal_source": reg["proposal_source"],
                        "decision_type": "region->object", "yolo_label": None,
                        "yolo_score": None}
            confirmed = False
            try:
                for cand in ball_candidates(img, reg, top=2):
                    r2 = dc.ask([{"type": "input_text",
                                  "text": "Candidate ball, centred, with surrounding pitch."},
                                 image_part(context_crop(img, cand, pad=3.0, min_h=160))],
                                q_ball_box(), tag=f"ballbox:{reg['region_id']}")
                    a2 = next((x for x in r2.get("answers", []) if x["name"] == "is_ball"), None)
                    conf = a2.get("probability") if a2 and a2["type"] == "predicate" else None
                    if conf is not None and conf >= cfg.t_object_accept:
                        dets.append({**det_base, "box": cand, "cls": "ball",
                                     "localization": "tight", "confidence": conf,
                                     "probabilities": None, "uncertain_reason": None,
                                     "decision_source": r2["source"]})
                        confirmed = True
                        break
            except BudgetExhausted as exc:
                reg["note"] = str(exc)
            if confirmed:
                continue

            # nothing confirmed here: subdivide if we still can, else say so plainly
            if reg["depth"] < cfg.refine_depth:
                queue = _split(reg) + queue
            else:
                dets.append({**det_base, "box": reg["box"], "cls": "ball",
                             "localization": "ball_present_unlocalized",
                             "confidence": reg["ball_p"], "probabilities": None,
                             "uncertain_reason": "presence supported; no tight box confirmed",
                             "decision_source": reg["decision_source"]})
    return dets, regions_log


# --------------------------------------------------------------------------
# step 7b - second pass over everything the 6-way question left uncertain
# --------------------------------------------------------------------------
RESOLVE_MAP = {"team_1_player": "team_1_player", "team_2_player": "team_2_player",
               "referee": "referee", "not_a_player": "background"}


def resolve_uncertain(cfg: Config, dc: DecisionClient, frames: list[dict],
                      dets: list[dict], kits: dict) -> tuple[list[dict], dict]:
    """Re-ask only the undecided detections, with a narrower question.

    Honest framing: this is allowed to change a label only because it adds
    evidence (a bigger crop, a referee reference) and narrows the question from
    six options to four. It uses the SAME accept threshold. Lowering the bar
    instead would just be relabelling noise.
    """
    by_id = {f["frame_id"]: f for f in frames}
    refs = kits.get("refs")
    stats = {"attempted": 0, "resolved": 0, "still_uncertain": 0,
             "by_class": {}, "skipped": 0}
    if not kits.get("resolved") or not refs:
        stats["skipped"] = sum(1 for d in dets if d["cls"] == "uncertain")
        return dets, stats

    out = []
    for d in dets:
        if d["cls"] != "uncertain" or d["decision_type"] != "object":
            out.append(d)
            continue
        stats["attempted"] += 1
        crop = context_crop(by_id[d["frame_id"]]["image"], d["box"], pad=0.2, min_h=288)
        content = [{"type": "input_text", "text": "TEAM 1 reference kit:"},
                   image_part(refs[0]),
                   {"type": "input_text", "text": "TEAM 2 reference kit:"},
                   image_part(refs[1])]
        if kits.get("referee_ref") is not None:
            content += [{"type": "input_text", "text": "REFEREE reference kit:"},
                        image_part(kits["referee_ref"])]
        content += [{"type": "input_text", "text": "Subject to resolve, enlarged:"},
                    image_part(crop)]
        try:
            r = dc.ask(content, q_resolve(), tag=f"resolve:{d['proposal_id']}")
        except BudgetExhausted as exc:
            out.append({**d, "uncertain_reason": f"{d['uncertain_reason']}; resolve skipped: {exc}"})
            continue
        a = next((x for x in r.get("answers", []) if x["name"] == "resolved_class"), None)
        if not a or a["type"] == "refusal":
            stats["still_uncertain"] += 1
            out.append({**d, "pass2_confidence": None,
                        "uncertain_reason": f"{d['uncertain_reason']}; pass2 no answer"})
            continue
        conf = a.get("confidence")
        if (conf or 0) >= cfg.t_object_accept:
            cls = RESOLVE_MAP.get(a["choice"], "uncertain")
            stats["resolved"] += 1
            stats["by_class"][cls] = stats["by_class"].get(cls, 0) + 1
            out.append({**d, "cls": cls, "confidence": conf,
                        "probabilities": a.get("probabilities"),
                        "decision_type": "object_pass2", "uncertain_reason": None,
                        "pass1_confidence": d["confidence"], "decision_source": r["source"]})
        else:
            stats["still_uncertain"] += 1
            out.append({**d, "pass2_confidence": conf, "probabilities": a.get("probabilities"),
                        "uncertain_reason": f"pass1={d['confidence']}, pass2={conf}, "
                                            f"both < {cfg.t_object_accept}"})
    return out, stats


def q_force() -> list[dict]:
    """Final forced choice over PERSON proposals - no escape hatch.

    `ball` is deliberately absent. These candidates all came from person
    proposals, and offering `ball` here produced exactly the failure you would
    predict: a dark crowd patch in the top-right corner was forced to `ball` at
    0.40 confidence. Balls come from the dedicated ball search, which starts
    from pitch regions and confirms a tight box.
    """
    return [{"type": "choice", "name": "forced_class",
             "instructions": "References were shown first: TEAM 1 kit, TEAM 2 kit, REFEREE kit. "
                             "The final image shows one person, enlarged, at the centre. You "
                             "MUST pick the single best of these three - there is no 'unsure' "
                             "option. If the shirt is unclear, judge on whatever is visible: "
                             "shorts, socks, goalkeeper gloves, position on the pitch. A "
                             "goalkeeper wears a kit unlike both outfield teams but still "
                             "belongs to one of them - pick the likelier side.",
             "choices": [
                 {"value": "team_1_player", "description": "Belongs to TEAM 1 (outfield player "
                                                           "or that side's goalkeeper)."},
                 {"value": "team_2_player", "description": "Belongs to TEAM 2 (outfield player "
                                                           "or that side's goalkeeper)."},
                 {"value": "referee", "description": "A match official, not on either team."},
             ]}]


def force_labels(cfg: Config, dc: DecisionClient, frames: list[dict],
                 dets: list[dict], kits: dict) -> tuple[list[dict], dict]:
    """Resolve every remaining `uncertain` into one of the four output labels.

    By request, the delivered output carries only ball / referee / team_1 /
    team_2. Two honesty guards survive that:

      * Anything the earlier pass judged `not_a_player` (banners, screens,
        crowd) is dropped from the output rather than forced into a team -
        inventing players would corrupt any downstream shape analysis.
      * The argmax is accepted regardless of confidence, so `forced=True` and
        `forced_confidence` are recorded on each one. Filter on those to get
        the cautious view back.
    """
    by_id = {f["frame_id"]: f for f in frames}
    refs = kits.get("refs")
    stats = {"attempted": 0, "forced": 0, "dropped_background": 0, "by_class": {},
             "low_confidence": 0}
    out = []
    for d in dets:
        if d["cls"] == "background":
            stats["dropped_background"] += 1
            continue
        if d["cls"] != "uncertain":
            out.append(d)
            continue
        if not kits.get("resolved") or not refs:
            out.append(d)
            continue
        stats["attempted"] += 1
        crop = context_crop(by_id[d["frame_id"]]["image"], d["box"], pad=0.2, min_h=288)
        content = [{"type": "input_text", "text": "TEAM 1 reference kit:"}, image_part(refs[0]),
                   {"type": "input_text", "text": "TEAM 2 reference kit:"}, image_part(refs[1])]
        if kits.get("referee_ref") is not None:
            content += [{"type": "input_text", "text": "REFEREE reference kit:"},
                        image_part(kits["referee_ref"])]
        content += [{"type": "input_text", "text": "Subject - you must choose one of four:"},
                    image_part(crop)]
        try:
            r = dc.ask(content, q_force(), tag=f"force:{d['proposal_id']}")
        except BudgetExhausted as exc:
            out.append({**d, "uncertain_reason": f"{d['uncertain_reason']}; force skipped: {exc}"})
            continue
        a = next((x for x in r.get("answers", []) if x["name"] == "forced_class"), None)
        if not a or a["type"] == "refusal":
            out.append(d)
            continue
        conf = a.get("confidence")
        stats["forced"] += 1
        stats["by_class"][a["choice"]] = stats["by_class"].get(a["choice"], 0) + 1
        if (conf or 0) < cfg.t_object_accept:
            stats["low_confidence"] += 1
        out.append({**d, "cls": a["choice"], "confidence": conf,
                    "probabilities": a.get("probabilities"), "decision_type": "object_forced",
                    "forced": True, "forced_confidence": conf, "uncertain_reason": None,
                    "pass1_confidence": d.get("pass1_confidence", d["confidence"]),
                    "decision_source": r["source"]})
    return out, stats


COLOURS = {"team_1_player": (255, 120, 0), "team_2_player": (0, 215, 255),
           "ball": (0, 255, 0), "referee": (255, 0, 255),
           "background": (120, 120, 120), "uncertain": (0, 0, 255)}


def render(frame: dict, dets: list[dict], regions: list[dict] | None = None) -> np.ndarray:
    img = frame["image"].copy()
    for r in (regions or []):
        if r.get("verdict") != "ball_present":
            continue
        x0, y0, x1, y1 = r["box"]
        for x in range(x0, x1, 14):                 # dashed = search region, not a box
            cv2.line(img, (x, y0), (min(x + 7, x1), y0), (0, 180, 0), 1)
            cv2.line(img, (x, y1 - 1), (min(x + 7, x1), y1 - 1), (0, 180, 0), 1)
    for d in dets:
        x0, y0, x1, y1 = d["box"]
        c = COLOURS.get(d["cls"], (255, 255, 255))
        thick = 1 if d["localization"] != "tight" else 2
        cv2.rectangle(img, (x0, y0), (x1, y1), c, thick)
        conf = f" {d['confidence']:.2f}" if d.get("confidence") is not None else ""
        tag = d["cls"].replace("_player", "")
        if d["localization"] != "tight":
            tag += "?"
        if d.get("forced"):
            tag += "*"            # label was forced, not confidently earned
        if d.get("track_id") is not None:
            tag += f" #{d['track_id']}"   # identity from the tracker, see frame_tracks.py
        cv2.putText(img, f"{tag}{conf}", (x0, max(y0 - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, c, 1, cv2.LINE_AA)
    return img


def contact_sheet(images: list[np.ndarray], cols: int = 3, width: int = 520) -> np.ndarray:
    tiles = []
    for im in images:
        s = width / im.shape[1]
        tiles.append(cv2.resize(im, (width, int(im.shape[0] * s))))
    th = max(t.shape[0] for t in tiles)
    rows = []
    for i in range(0, len(tiles), cols):
        row = [np.pad(t, ((0, th - t.shape[0]), (0, 0), (0, 0))) for t in tiles[i:i + cols]]
        while len(row) < cols:
            row.append(np.zeros_like(row[0]))
        rows.append(np.hstack(row))
    return np.vstack(rows)


def export(cfg: Config, frames: list[dict], dets: list[dict], regions: list[dict],
           kits: dict, hud: dict, dc: DecisionClient, dropped: list[dict]) -> dict:
    meta = {
        "video": cfg.video, "config": {k: v for k, v in asdict(cfg).items()},
        "api": {"endpoint": DECISIONS_ENDPOINT, "model": cfg.model, "docs": DOC_URL,
                "sdk": __import__("openai").__version__},
        "frames": [{k: v for k, v in f.items() if k != "image"} for f in frames],
        "hud_mask": hud,
        "team_mapping": {k: v for k, v in kits.items() if k not in ("board", "groups")},
        "kit_groups": [{k: v for k, v in g.items() if k != "members"}
                       for g in kits.get("groups", [])],
        "decisions": dc.stats(),
        "call_log": dc.log,
        "counts_per_frame": {
            f["frame_id"]: {
                **{c: sum(1 for d in dets if d["frame_id"] == f["frame_id"] and d["cls"] == c)
                   for c in CLASSES},
                "ball_unlocalized": sum(1 for d in dets if d["frame_id"] == f["frame_id"]
                                        and d["cls"] == "ball" and d["localization"] != "tight"),
                "regions_searched": sum(1 for r in regions if r["frame_id"] == f["frame_id"]
                                        and r.get("verdict") != "unsearched"),
                "regions_unsearched": sum(1 for r in regions if r["frame_id"] == f["frame_id"]
                                          and r.get("verdict") == "unsearched"),
                "proposals_dropped_by_budget": f.get("proposals_dropped_by_budget", 0),
                "ball_regions_dropped_by_budget": f.get("ball_regions_dropped_by_budget", 0),
            } for f in frames},
        "suppressed": len(dropped),
    }
    cfg.path("results.json").write_text(json.dumps(
        {"meta": meta, "detections": dets, "regions": regions}, indent=1, default=str))

    ts = {f["frame_id"]: f["timestamp_s"] for f in frames}
    dims = {f["frame_id"]: (f["width"], f["height"]) for f in frames}
    cols = ["frame_id", "timestamp_s", "frame_w", "frame_h", "proposal_id", "parent_id",
            "proposal_source", "decision_type", "cls", "confidence", "localization",
            "x0", "y0", "x1", "y1", "yolo_label", "yolo_score", "uncertain_reason",
            "decision_source", "track_id", "track_iou", "track_label", "track_conflict"]
    with cfg.path("detections.csv").open("w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=cols)
        wr.writeheader()
        for d in dets:
            w, h = dims[d["frame_id"]]
            wr.writerow({"frame_id": d["frame_id"], "timestamp_s": ts[d["frame_id"]],
                         "frame_w": w, "frame_h": h,
                         **{k: d.get(k) for k in ("proposal_id", "parent_id", "proposal_source",
                                                  "decision_type", "cls", "confidence",
                                                  "localization", "yolo_label", "yolo_score",
                                                  "uncertain_reason", "decision_source",
                                                  "track_id", "track_iou", "track_label",
                                                  "track_conflict")},
                         "x0": d["box"][0], "y0": d["box"][1],
                         "x1": d["box"][2], "y1": d["box"][3]})
    with cfg.path("regions.csv").open("w", newline="") as fh:
        rc = ["frame_id", "region_id", "parent_id", "depth", "scale", "cue", "ball_p",
              "verdict", "x0", "y0", "x1", "y1", "proposal_source", "decision_source"]
        wr = csv.DictWriter(fh, fieldnames=rc, extrasaction="ignore")
        wr.writeheader()
        for r in regions:
            wr.writerow({**r, "x0": r["box"][0], "y0": r["box"][1],
                         "x1": r["box"][2], "y1": r["box"][3]})
    return meta


