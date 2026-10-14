"""track.py - track people once, ask Decisions once per track, find the ball without inventing it.

run_clip: HUD mask -> ByteTrack the window (local, no API) -> kit discovery -> one Decisions vote per track
from ~3 crops (a track whose crops disagree confidently is SPLIT in time, never given one label) -> YOLO
"sports ball" paths, Decisions confirms checkpoints, template-match growth.

Rules kept:
  * Decisions is the classifier; YOLO only proposes boxes and track ids.
  * Nothing is fabricated. A confirmed ball is a tight box Decisions accepted. Frames between two accepted
    checkpoints of one path, and frames reached by growth, are localization="tracked": a visual estimate,
    never counted as confirmed. A rejected checkpoint ends the stretch; nothing is interpolated across it.
    Frames no path covers are ball_searched=False, not "no ball".
  * Confidences are scores, not calibrated probabilities. Forced labels carry forced=True.
  * dry_run=True (the default) makes ZERO API calls: no client is created and ask() refuses to call out.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config_io import (MAX_REQUEST_BYTES, BudgetExhausted, Config, DecisionClient,  # noqa: E402
                       estimate_hud_mask, image_part)
from decide import RESOLVE_MAP, context_crop, q_ball_box, q_force, q_resolve  # noqa: E402
from detect import _iou, _kit_colour, _torso, kit_hypotheses, pitch_mask, validate_kits  # noqa: E402

OUTPUT_LABELS = ("ball", "referee", "team_1_player", "team_2_player")
PERSON_CLASSES = ("team_1_player", "team_2_player", "referee")
CROP_BUCKET = 15        # frames: at most one candidate crop per track per half second
BALL_CLASS = 32         # COCO "sports ball"
GROW_NCC = 0.70         # template-match floor for growth (no Decisions re-check follows those frames)


@dataclass
class Opts:
    """Everything that changes what a run does."""
    yolo_weights: str = "models/yolo11s.pt"
    conf: float = 0.15
    imgsz: int = 0                    # 0 = ultralytics default (640)
    hud_mode: str = "auto"            # "auto" or "none"
    device: str = ""
    min_track_len: int = 10
    crops_per_track: int = 3
    workers: int = 12
    max_calls: int = 3000
    kit_samples: int = 20
    kit_clusters: int = 5
    kit_core_frac: float = 0.5
    hud_samples: int = 8
    ball_paths: bool = True
    ball_path_every: int = 8          # ask Decisions about a path every N frames (1 call each)
    ball_path_min_conf: float = 0.40  # skip paths whose mean YOLO ball score is lower (a score, not a probability)
    ball_grow: int = 6                # template-match growth, frames each way from a confirmed box (0 = off)


class ConcurrentDecisionClient(DecisionClient):
    """DecisionClient a thread pool can share: bookkeeping locked, the HTTP call is not. Saves the cache
    every `save_every` new answers so a crash loses at most that much spend."""

    def __init__(self, cfg: Config, save_every: int = 25):
        super().__init__(cfg)
        self._lock, self._save_every, self._since_save = threading.Lock(), save_every, 0
        self.tag_counts: dict[str, int] = {}
        self.budget_stop = False

    def prime(self) -> None:
        if not self.cfg.dry_run:
            self._lazy()                      # build the HTTP client before threads race on it

    def _bump(self, tag: str) -> None:
        self.tag_counts[tag.split(":")[0]] = self.tag_counts.get(tag.split(":")[0], 0) + 1

    def ask(self, content: list[dict], questions: list[dict], tag: str) -> dict:
        size = len(json.dumps({"input": [{"role": "user", "content": content}], "questions": questions}).encode())
        if size > MAX_REQUEST_BYTES:
            return {"answers": [], "source": "error", "tag": tag, "error": f"payload {size}B over ceiling"}
        ck = self._key(content, questions, self.cfg.model)
        with self._lock:
            if ck in self.cache:
                self.cache_hits += 1
                self._bump(tag)
                return {**self.cache[ck], "source": "cache"}
            if self.cfg.dry_run:              # enforced here too: a dry run can never reach the network
                return {"answers": [], "source": "dry_run", "payload_bytes": size, "tag": tag}
        last_err = None
        for attempt in range(self.cfg.max_retries + 1):
            with self._lock:
                if self.calls >= self.cfg.max_api_calls:
                    self.budget_stop = True
                    raise BudgetExhausted(f"hit max_api_calls={self.cfg.max_api_calls}")
                self.calls += 1
            t0 = time.time()
            try:
                d = self._client.decisions.create(model=self.cfg.model, questions=questions,
                                                  input=[{"role": "user", "content": content}])
                res = {"answers": [a.model_dump() for a in d.answers], "tag": tag, "payload_bytes": size,
                       "usage": getattr(d, "usage", None) and d.usage.model_dump(),
                       "latency_s": round(time.time() - t0, 3)}
                with self._lock:
                    self.cache[ck] = res
                    self.log.append({"tag": tag, "ok": True, "attempt": attempt, "latency_s": res["latency_s"]})
                    self._bump(tag)
                    self._since_save += 1
                    due, self._since_save = self._since_save >= self._save_every, self._since_save % self._save_every
                if due:
                    self.save()
                return {**res, "source": "live"}
            except Exception as exc:          # recorded, never silenced
                last_err = f"{type(exc).__name__}: {exc}"
                with self._lock:
                    self.log.append({"tag": tag, "ok": False, "attempt": attempt, "error": last_err})
        return {"answers": [], "source": "error", "error": last_err, "tag": tag}

    def save(self) -> None:
        """Atomic: Ctrl-C mid-write must not leave a truncated cache."""
        with self._lock:
            blob = json.dumps(self.cache, indent=1)
        tmp = self.cache_path.with_suffix(".json.tmp")
        tmp.write_text(blob)
        tmp.replace(self.cache_path)


def _upscale(crop: np.ndarray, min_h: int) -> np.ndarray:
    """Cubic upscale at send time; crops are stored at source resolution so a whole clip fits in memory."""
    if crop.size == 0 or crop.shape[0] >= min_h:
        return crop
    return cv2.resize(crop, (max(int(crop.shape[1] * min_h / crop.shape[0]), 1), min_h), interpolation=cv2.INTER_CUBIC)


def _answer(res: dict, name: str) -> dict | None:
    a = next((x for x in res.get("answers", []) if x["name"] == name), None)
    return None if (a is None or a.get("type") == "refusal") else a


def cam_h(g0: np.ndarray, g1: np.ndarray) -> np.ndarray:
    """Camera motion g0 -> g1 as a homography (RANSAC on tracked corners); identity if unsure."""
    p = cv2.goodFeaturesToTrack(g0, 1200, 0.005, 6)
    if p is None:
        return np.eye(3)
    q, st, _ = cv2.calcOpticalFlowPyrLK(g0, g1, p, None, winSize=(31, 31), maxLevel=4)
    ok = st[:, 0] > 0
    if ok.sum() < 40:
        return np.eye(3)
    H, _ = cv2.findHomography(p[ok, 0], q[ok, 0], cv2.RANSAC, 2.5)
    return H if H is not None else np.eye(3)


# --------------------------------------------------------------------------
# HUD + ByteTrack over the window (no API calls)
# --------------------------------------------------------------------------
def _window(video: str, start: float, duration: float) -> tuple[float, int, int]:
    """(fps, first frame, frame count) of the requested window, validated."""
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video}")
    fps, total = cap.get(cv2.CAP_PROP_FPS) or 0.0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    first = int(round(start * fps))
    if fps <= 0 or first >= total:
        raise RuntimeError(f"no usable fps, or start {start}s is past the end of the video")
    n = min(int(round(duration * fps)), total - first)
    if n < 2:
        raise RuntimeError("window shorter than 2 frames")
    return fps, first, n


def hud_mask(video: str, first: int, n_frames: int, samples: int = 8, mode: str = "auto"):
    """Sample frames far apart (overlay edges stay, scene edges move) and estimate the HUD mask."""
    cap = cv2.VideoCapture(video)
    frames = []
    for idx in sorted({int(round(x)) for x in np.linspace(first, first + n_frames - 1, max(samples, 2))}):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, img = cap.read()
        if ok:
            frames.append({"frame_id": f"h{len(frames)}", "image": img})
    cap.release()
    if len(frames) < 2:
        raise RuntimeError(f"could not decode 2 frames from {video} at frame {first}")
    return estimate_hud_mask(frames, mode=mode)


def track_clip(video: str, start: float, duration: float, cfg: Config, hud: np.ndarray, o: Opts) -> dict:
    """Decode the window once, ByteTrack every frame, keep small crops (never the frames). No API calls.

    Per frame: tracked boxes (track_id, box, yolo score, feet-on-pitch). Per track: at most one candidate
    torso crop per half second (gated on feet on the pitch and a torso that is >=35% not grass).
    """
    from ultralytics import YOLO
    model = YOLO(cfg.yolo_weights)
    fps, first, n_frames = _window(video, start, duration)
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    landed = int(round(cap.get(cv2.CAP_PROP_POS_FRAMES)))
    kit_set = {int(round(x)) for x in np.linspace(0, n_frames - 1, max(o.kit_samples, 2))}
    frames_meta, tracks, kit_frames, kit_props = [], {}, [], []
    mg, t0, decoded = cfg.pitch_margin_px, time.time(), 0
    for rel in range(n_frames):
        ok, img = cap.read()
        if not ok:
            break
        decoded += 1
        src, (h, w) = first + rel, img.shape[:2]
        # tolerance grows sideways and DOWNWARD only: growing upward re-admits spectators
        grown = cv2.dilate(pitch_mask(img, hud), np.ones((mg + 1, 2 * mg + 1), np.uint8), anchor=(mg, mg))
        kw = {"imgsz": cfg.yolo_imgsz} if cfg.yolo_imgsz else {}   # unset: ultralytics default
        res = model.track(img, persist=True, conf=cfg.yolo_conf, classes=[0], tracker="bytetrack.yaml",
                          verbose=False, device=o.device or None, **kw)[0]
        ids = res.boxes.id.int().tolist() if (res.boxes is not None and res.boxes.id is not None) else []
        boxes = []
        for b, tid in (zip(res.boxes, ids) if ids and len(ids) == len(res.boxes) else []):
            x0, y0, x1, y1 = (int(v) for v in b.xyxy[0].tolist())
            on_pitch = bool(grown[min(max(y1 - 2, 0), h - 1), min(max((x0 + x1) // 2, 0), w - 1)])  # the FEET
            score = round(float(b.conf[0]), 4)
            boxes.append({"track_id": tid, "box": [x0, y0, x1, y1], "yolo_score": score, "on_pitch": on_pitch})
            t = tracks.setdefault(tid, {"track_id": tid, "first_rel": rel, "n_frames": 0, "on_pitch_frames": 0,
                                        "crops": {}, "yolo_scores": []})
            t["last_rel"] = rel
            t["n_frames"] += 1
            t["on_pitch_frames"] += int(on_pitch)
            t["yolo_scores"].append(score)
            if not (on_pitch and (y1 - y0) >= 28 and (x1 - x0) >= 10):
                continue
            # _kit_colour doubles as a quality gate: None when the torso is mostly grass, plus a sharpness
            col, sharp = _kit_colour(_torso(img, [x0, y0, x1, y1]))
            q, bk = float(sharp) * float((y1 - y0) * (x1 - x0)) ** 0.5, rel // CROP_BUCKET
            if col is not None and sharp >= 8.0 and (bk not in t["crops"] or q > t["crops"][bk]["score"]):
                t["crops"][bk] = {"rel": rel, "source_index": src, "score": round(q, 1), "sharpness": round(float(sharp), 1),
                                  "box": [x0, y0, x1, y1], "crop": context_crop(img, [x0, y0, x1, y1], pad=0.2, min_h=1)}
        if rel in kit_set:
            fid = f"s{len(kit_frames)}"
            kit_frames.append({"frame_id": fid, "image": img.copy(), "source_index": src,
                               "timestamp_s": round(src / fps, 4), "width": w, "height": h})
            kit_props += [{"frame_id": fid, "box": bx["box"], "proposal_source": "yolo+bytetrack+on_pitch",
                           "yolo_label": "person", "yolo_score": bx["yolo_score"],  # metadata ONLY
                           "proposal_id": f"{fid}_p{i:02d}"} for i, bx in enumerate(x for x in boxes if x["on_pitch"])]
        frames_meta.append({"rel": rel, "source_index": src, "timestamp_s": round(src / fps, 4),
                            "boxes": boxes, "ball_searched": False})
        if rel and rel % 150 == 0:
            print(f"    tracked {rel}/{n_frames} frames, {len(tracks)} track ids", flush=True)
    cap.release()
    for t in tracks.values():
        t["on_pitch_frac"] = round(t["on_pitch_frames"] / max(t["n_frames"], 1), 3)
        t["mean_yolo_score"] = round(float(np.mean(t.pop("yolo_scores"))), 4)
    return {"fps": fps, "first": first, "landed": landed, "n_frames": decoded, "frames": frames_meta,
            "tracks": tracks, "kit_frames": kit_frames, "kit_proposals": kit_props,
            "track_seconds": round(time.time() - t0, 1)}


def track_window(cfg: Config, hud: np.ndarray, o: Opts | None = None) -> dict:
    """Notebook entry: ByteTrack over cfg's window. Same tracker as run_clip, zero API calls.

    Public helper for video input; not used by the sample notebook (which reads still frames).
    """
    o = o or Opts(yolo_weights=cfg.yolo_weights, conf=cfg.yolo_conf, kit_samples=2)
    return track_clip(cfg.video, cfg.clip_start_s, cfg.clip_len_s, cfg, hud, o)


def link_tracks(dets: list[dict], frames: list[dict], tr: dict, min_iou: float = 0.6) -> dict:
    """Public helper for video input; not used by the sample notebook.

    Bridge: attach `track_id` and `track_iou` to each judged person box by best IoU (>= min_iou).

    `frames` are the notebook's judged frames (frame_id, source_index). Ball boxes are not tracked, labels
    stay Decisions' own, and nothing is painted onto frames that were not judged.
    """
    by_src = {m["source_index"]: m["boxes"] for m in tr["frames"]}
    src_of = {f["frame_id"]: f["source_index"] for f in frames}
    linked = unlinked = 0
    for d in dets:
        d["track_id"], d["track_iou"] = None, None
        if d["cls"] not in PERSON_CLASSES:
            continue
        best, best_iou = None, min_iou
        for tb in by_src.get(src_of[d["frame_id"]], []):
            if (v := _iou(d["box"], tb["box"])) >= best_iou:
                best, best_iou = tb, v
        if best is None:
            unlinked += 1
        else:
            d["track_id"], d["track_iou"] = best["track_id"], round(best_iou, 3)
            linked += 1
    return {"linked": linked, "unlinked": unlinked, "min_iou": min_iou}


# --------------------------------------------------------------------------
# kits, then one Decisions vote per track
# --------------------------------------------------------------------------
def qualify_tracks(tracks: dict, o: Opts) -> tuple[list[dict], dict]:
    keep, skipped = [], {}
    for t in sorted(tracks.values(), key=lambda x: (x["first_rel"], x["track_id"])):
        why = (f"short track ({t['n_frames']} < {o.min_track_len})" if t["n_frames"] < o.min_track_len else
               f"off pitch ({t['on_pitch_frac']:.2f} of frames)" if t["on_pitch_frac"] < 0.5 else
               "no usable torso crop (too small, blurred or grass)" if not t["crops"] else None)
        if why:
            skipped[t["track_id"]] = why
        else:
            keep.append(t)
    return keep, skipped


def pick_crops(t: dict, k: int) -> list[dict]:
    """Best crop from each of k equal slices of the track's life: early/middle/late crops expose an ID switch."""
    buckets = [c for _, c in sorted(t["crops"].items())]
    if len(buckets) <= k:
        return buckets
    return [max((buckets[i] for i in g), key=lambda c: c["score"]) for g in np.array_split(np.arange(len(buckets)), k)]


def discover_kits(cfg: Config, dc: ConcurrentDecisionClient, tr: dict, o: Opts) -> dict:
    """k-means torso-colour clusters over sampled frames, each trimmed to its colour core, judged by Decisions.

    Judging a cluster on its core, not its four sharpest crops, stops one mis-clustered player sinking the
    group (measured: same_kit_p 0.26 -> 0.86). It trims the evidence; it does not lower the threshold.
    """
    hyp = kit_hypotheses(tr["kit_frames"], tr["kit_proposals"], k=cfg.k_clusters)
    by_id, groups = {f["frame_id"]: f for f in tr["kit_frames"]}, []
    for g in hyp.get("groups", []):
        pairs = [(c, m) for m in g["members"]
                 if (c := _kit_colour(_torso(by_id[m["frame_id"]]["image"], m["box"]))[0]) is not None]
        if o.kit_core_frac >= 1.0 or len(pairs) < 4:
            groups.append(g)
            continue
        cols = np.asarray([c for c, _ in pairs])
        d = np.linalg.norm(cols - np.median(cols, axis=0), axis=1)
        kept = [pairs[i][1] for i in sorted(np.argsort(d)[:max(4, int(round(len(pairs) * o.kit_core_frac)))])]
        groups.append({**g, "members": kept, "size": len(kept), "members_before_trim": len(pairs)})
    kits = validate_kits(cfg, dc, tr["kit_frames"], {**hyp, "groups": groups, "core_trim": o.kit_core_frac})
    kits["core_trim"] = o.kit_core_frac
    return kits


def _reference_content(kits: dict) -> list[dict]:
    """Separate labelled reference images, never one captioned montage (that biased everything to team_1)."""
    content = [{"type": "input_text", "text": "TEAM 1 reference kit:"}, image_part(kits["refs"][0]),
               {"type": "input_text", "text": "TEAM 2 reference kit:"}, image_part(kits["refs"][1])]
    if kits.get("referee_ref") is not None:
        content += [{"type": "input_text", "text": "REFEREE reference kit:"}, image_part(kits["referee_ref"])]
    return content


def _segments(t: dict, votes: list[dict]) -> list[dict]:
    """One span per confident vote, cut at the midpoints between them (the identity-switch split)."""
    pts, merged = sorted(votes, key=lambda v: v["rel"]), []
    for i, v in enumerate(pts):
        s = {"start_rel": t["first_rel"] if i == 0 else (pts[i - 1]["rel"] + v["rel"]) // 2 + 1,
             "end_rel": t["last_rel"] if i == len(pts) - 1 else (v["rel"] + pts[i + 1]["rel"]) // 2,
             "label": v["cls"], "confidence": v["conf"], "forced": False, "from_rel": v["rel"]}
        if merged and merged[-1]["label"] == s["label"]:      # fold identical neighbours
            merged[-1]["end_rel"] = s["end_rel"]
            merged[-1]["confidence"] = round((merged[-1]["confidence"] + s["confidence"]) / 2, 4)
        else:
            merged.append(s)
    return merged


def classify_tracks(cfg: Config, dc: ConcurrentDecisionClient, kits: dict, tracks: list[dict],
                    o: Opts) -> tuple[dict, dict]:
    """One decision per crop (`q_resolve`: four labels, no ball), pooled by confidence-weighted vote.

    Confident votes for >=2 person classes -> the track is split in time. Only tracks with no confident
    vote at all pay for the forced pass, and those are tagged forced=True.
    """
    base = _reference_content(kits)
    stats = {"tracks": len(tracks), "confident": 0, "forced": 0, "segmented": 0,
             "dropped_not_a_player": 0, "unresolved": 0, "budget_skipped": 0}

    def ask_crop(crop: dict, text: str, q: list, tag: str) -> dict:
        return dc.ask(base + [{"type": "input_text", "text": text}, image_part(_upscale(crop["crop"], 288))], q, tag=tag)

    def work(t: dict) -> dict:
        tid = t["track_id"]
        dec = {"track_id": tid, "label": None, "confidence": None, "forced": False, "segmented": False,
               "votes": [], "vote_weights": {}, "calls": 0, "segments": [], "note": None,
               "lifetime": [t["first_rel"], t["last_rel"]], "n_frames": t["n_frames"], "on_pitch_frac": t["on_pitch_frac"]}
        if dc.budget_stop:
            dec["note"] = "skipped: call budget already exhausted"
            return dec
        picks = pick_crops(t, o.crops_per_track)
        try:
            for c in picks:
                r = ask_crop(c, "Subject to label, enlarged (centre):", q_resolve(), f"track:{tid}")
                dec["calls"] += 1
                a = _answer(r, "resolved_class")
                v = {"rel": c["rel"], "cls": None, "choice": None, "conf": None, "source": r["source"]}
                if a:
                    v.update(choice=a["choice"], cls=RESOLVE_MAP.get(a["choice"], "uncertain"),
                             conf=a.get("confidence"), probabilities=a.get("probabilities"))
                else:
                    v["error"] = r.get("error") or f"no answer ({r['source']})"
                dec["votes"].append(v)
        except BudgetExhausted as exc:
            dec["note"] = f"budget stop: {exc}"
        answered = [v for v in dec["votes"] if v["cls"]]
        confident = [v for v in answered if (v["conf"] or 0) >= cfg.t_object_accept]
        people = [v for v in confident if v["cls"] in PERSON_CLASSES]
        w: dict[str, float] = {}
        for v in answered:
            w[v["cls"]] = round(w.get(v["cls"], 0.0) + (v["conf"] or 0.0), 4)
        dec["vote_weights"] = w
        if people:
            classes = {v["cls"] for v in people}
            best = max(classes, key=lambda c: w.get(c, 0.0))
            sup = [v for v in people if v["cls"] == best]
            if len(classes) >= 2:
                dec.update(segmented=True, segments=_segments(t, people), label=best,
                           confidence=round(float(np.mean([v["conf"] for v in people])), 4),
                           note="crops disagree confidently - likely ID switch; track split in time")
            else:
                conf = round(float(np.mean([v["conf"] for v in sup])), 4)
                dec.update(label=best, confidence=conf, segments=[
                    {"start_rel": t["first_rel"], "end_rel": t["last_rel"], "label": best,
                     "confidence": conf, "forced": False, "from_rel": None}])
            return dec
        if any(v["cls"] == "background" for v in confident):
            dec["note"] = "confidently not a player (crowd, dugout, screen or HUD) - dropped, not forced"
            return dec
        if dc.budget_stop or dec["note"] or not picks:
            return dec
        best_crop = max(picks, key=lambda c: c["score"])   # nothing confident: force on the sharpest crop
        try:
            r = ask_crop(best_crop, "Subject - you must choose one of three:", q_force(), f"force:{tid}")
        except BudgetExhausted as exc:
            dec["note"] = f"budget stop before forced pass: {exc}"
            return dec
        dec["calls"] += 1
        a = _answer(r, "forced_class")
        if not a:
            dec["note"] = f"unresolved: forced pass gave no answer ({r.get('error') or r['source']})"
            return dec
        dec.update(label=a["choice"], confidence=a.get("confidence"), forced=True,
                   probabilities=a.get("probabilities"), note="label forced: no crop cleared the accept threshold",
                   segments=[{"start_rel": t["first_rel"], "end_rel": t["last_rel"], "label": a["choice"],
                              "confidence": a.get("confidence"), "forced": True, "from_rel": best_crop["rel"]}])
        return dec

    out: dict[int, dict] = {}
    with ThreadPoolExecutor(max_workers=o.workers) as ex:
        for fut in as_completed([ex.submit(work, t) for t in tracks]):
            dec = fut.result()
            out[dec["track_id"]] = dec
            note = dec["note"] or ""
            stats["forced" if dec["forced"] else "confident" if dec["label"] else "dropped_not_a_player"
                  if "not a player" in note else "budget_skipped" if "budget" in note else "unresolved"] += 1
            stats["segmented"] += int(dec["segmented"])
    return out, stats


def label_for(dec: dict, rel: int) -> tuple[str | None, float | None, bool]:
    """A track's label at this frame, honouring an identity-switch split."""
    segs = dec["segments"]
    if not segs:
        return None, None, False
    s = next((s for s in segs if s["start_rel"] <= rel <= s["end_rel"]),
             min(segs, key=lambda s: min(abs(rel - s["start_rel"]), abs(rel - s["end_rel"]))))
    return s["label"], s["confidence"], bool(s["forced"])


# --------------------------------------------------------------------------
# ball: YOLO "sports ball" paths, Decisions confirms checkpoints, template-match growth
# --------------------------------------------------------------------------
TOPN, MAX_STEP, NULL_COST, SPEED_W, EDGE_COST, MIN_LEN = 8, 80.0, 2.2, 0.045, 0.8, 4
R_T, SEARCH, NCC_MIN = 8, 34, 0.55   # template half-size, search radius, match floor (similarity, not a probability)


def _centre(b) -> np.ndarray:
    return np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])


def yolo_ball_candidates(video: str, first: int, n: int, weights: str, hud: np.ndarray,
                         conf: float = 0.03) -> tuple[list, np.ndarray]:
    """Per frame: pitch-gated YOLO ball boxes [x0,y0,x1,y1,score], best first; and camera homographies t -> t+1."""
    from ultralytics import YOLO
    model = YOLO(weights)
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    cands, Hs, prev = [], [], None
    for _ in range(n):
        ok, img = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if prev is not None:
            Hs.append(cam_h(prev, g))
        prev = g
        pm = cv2.dilate(pitch_mask(img, hud), np.ones((25, 25), np.uint8))
        r = model(img, verbose=False, conf=conf, classes=[BALL_CLASS], imgsz=max(img.shape[:2]))[0]
        keep = []
        for b, c in sorted(zip(r.boxes.xyxy.cpu().numpy().tolist(), r.boxes.conf.cpu().numpy().tolist()), key=lambda x: -x[1]):
            cx, cy = int((b[0] + b[2]) / 2), int((b[1] + b[3]) / 2)
            if 0 <= cy < pm.shape[0] and 0 <= cx < pm.shape[1] and pm[cy, cx] > 0:
                keep.append([int(b[0]), int(b[1]), int(b[2]), int(b[3]), round(float(c), 4)])
        cands.append(keep[:TOPN])
    cap.release()
    Hs.append(np.eye(3))
    return cands, np.stack(Hs)


def link_ball(cands: list, Hs: np.ndarray) -> list[int]:
    """Viterbi: cheapest sequence, 0 = no ball, j >= 1 = j-th candidate. A path can break; a step costs more
    the faster the ball moves in the camera-compensated image and the less confident YOLO is."""
    n = len(cands)
    cost = [np.array([NULL_COST] + [-np.log(max(c[4], 1e-3)) for c in cs]) for cs in cands]
    back, best = [None] * n, cost[0].copy()
    for t in range(1, n):
        warped = [None] + [cv2.perspectiveTransform(_centre(c).reshape(1, 1, 2).astype(np.float32), Hs[t - 1])[0, 0]
                           for c in cands[t - 1]]
        new, bk = np.full(len(cands[t]) + 1, np.inf), np.zeros(len(cands[t]) + 1, int)
        for j in range(len(cands[t]) + 1):
            for i in range(len(cands[t - 1]) + 1):
                if i == 0 or j == 0:
                    step = 0.0 if (i == 0 and j == 0) else EDGE_COST
                else:
                    d = float(np.linalg.norm(warped[i] - _centre(cands[t][j - 1])))
                    if d > MAX_STEP:
                        continue
                    step = SPEED_W * d
                if (v := best[i] + step + cost[t][j]) < new[j]:
                    new[j], bk[j] = v, i
        best, back[t] = new, bk
    j, seq = int(np.argmin(best)), [0] * n
    for t in range(n - 1, -1, -1):
        seq[t] = j
        j = int(back[t][j]) if t else j
    return seq


def build_paths(video: str, first: int, n: int, weights: str, hud: np.ndarray, cache: Path | None = None) -> dict:
    """{'chosen': {rel: [x0,y0,x1,y1,score]}, 'paths': [[rel, ...], ...]}; JSON-cached when `cache` is given."""
    if cache and cache.exists():
        d = json.loads(cache.read_text())
        if (d.get("n"), d.get("first"), d.get("weights")) == (n, first, weights):
            return {"chosen": {int(k): v for k, v in d["chosen"].items()}, "paths": d["paths"]}
    cands, Hs = yolo_ball_candidates(video, first, n, weights, hud)
    chosen = {t: cands[t][s - 1] for t, s in enumerate(link_ball(cands, Hs)) if s}
    paths, cur = [], []
    for t in range(len(cands) + 1):                  # the extra index flushes the last path
        if t in chosen:
            cur.append(t)
            continue
        if len(cur) >= MIN_LEN:
            paths.append(cur)
        cur = []
    if cache:
        cache.write_text(json.dumps({"n": n, "first": first, "weights": weights, "paths": paths,
                                     "chosen": {str(k): v for k, v in chosen.items()}}))
    return {"chosen": chosen, "paths": paths}


def checkpoints(path: list[int], every: int) -> list[int]:
    """Frames of a path Decisions is asked about: every `every`-th, plus the last."""
    cps = path[::max(every, 1)]
    return cps if path[-1] in cps else cps + [path[-1]]


def usable_paths(paths: dict, o: Opts) -> list[list[int]]:
    return [p for p in paths["paths"] if float(np.mean([paths["chosen"][t][4] for t in p])) >= o.ball_path_min_conf]


def confirm_ball_paths(cfg: Config, dc: ConcurrentDecisionClient, o: Opts, video: str, tr: dict,
                       paths: dict) -> tuple[dict, dict, dict]:
    """Decisions confirms checkpoints along each YOLO path (`is_ball`, one call each).

    An accepted checkpoint is a confirmed tight box. Frames between two consecutive ACCEPTED checkpoints of one
    path are 'tracked': the linked YOLO box, a visual estimate. A rejected checkpoint ends that stretch;
    nothing is carried across it and nothing is interpolated.
    """
    chosen, good = paths["chosen"], {id(p) for p in usable_paths(paths, o)}
    stats = {"paths": len(paths["paths"]), "paths_used": len(good), "checkpoint_calls": 0,
             "checkpoints_accepted": 0, "checkpoints_rejected": 0, "budget_skipped": 0}
    ball_by_rel, tracked = {}, {}
    cap = cv2.VideoCapture(video)
    for path in (p for p in paths["paths"] if id(p) in good):
        cps, accepted = checkpoints(path, o.ball_path_every), {}
        try:
            for rel in cps:
                box = chosen[rel][:4]
                cap.set(cv2.CAP_PROP_POS_FRAMES, tr["first"] + rel)
                ok, im = cap.read()
                if not ok:
                    continue
                r = dc.ask([{"type": "input_text", "text": "Candidate ball, centred, with surrounding pitch."},
                            image_part(_upscale(context_crop(im, box, pad=3.0, min_h=1), 160))],
                           q_ball_box(), tag=f"path:{rel}")
                stats["checkpoint_calls"] += 1
                a = _answer(r, "is_ball")
                p = a.get("probability") if a and a["type"] == "predicate" else None
                if p is not None and p >= cfg.t_object_accept:
                    accepted[rel] = p
                    ball_by_rel[rel] = {"rel": rel, "source_index": tr["first"] + rel, "ball": {
                        "box": [int(v) for v in box], "confidence": p, "localization": "tight",
                        "region_id": f"path:{rel}", "decision_source": r["source"]}}
                else:
                    stats["checkpoints_rejected"] += 1
        except BudgetExhausted:
            stats["budget_skipped"] += 1
        stats["checkpoints_accepted"] += len(accepted)
        for a_rel, b_rel in zip(cps, cps[1:]):
            if a_rel in accepted and b_rel in accepted:
                for rel in (r for r in path if a_rel < r < b_rel):
                    near = a_rel if rel - a_rel <= b_rel - rel else b_rel
                    tracked[rel] = {"box": [int(v) for v in chosen[rel][:4]], "ncc": chosen[rel][4],
                                    "seed_rel": near, "step": abs(rel - near), "source": "yolo_path",
                                    "confirmed_p": min(accepted[a_rel], accepted[b_rel])}
        for rel in path:
            tr["frames"][rel]["ball_searched"] = True
    cap.release()
    stats.update(tracked_frames=len(tracked), frames_with_ball=len(ball_by_rel),
                 frames_searched=sum(1 for f in tr["frames"] if f["ball_searched"]))
    return ball_by_rel, tracked, stats


def follow(frames: list, seed_i: int, box: list, direction: int, steps: int) -> list:
    """Template-follow the ball from grey `frames[seed_i]`, one frame at a time, camera-compensated.

    Returns [(index into frames, cx, cy, ncc)] excluding the seed. Stops at the first weak match or when it no
    longer resembles the ORIGINAL seed (so the template cannot drift onto a sock).
    """
    cx, cy = _centre(box)
    tpl = frames[seed_i][int(cy) - R_T:int(cy) + R_T + 1, int(cx) - R_T:int(cx) + R_T + 1].copy()
    if tpl.shape != (2 * R_T + 1, 2 * R_T + 1):
        return []
    seed_tpl, vel, out, i = tpl.copy(), np.zeros(2), [], seed_i
    for _ in range(steps):
        j = i + direction
        if not 0 <= j < len(frames):
            break
        warped = cv2.perspectiveTransform(np.float32([[[cx, cy]]]), cam_h(frames[i], frames[j]))[0, 0]
        pred = warped + vel
        x0, y0 = max(int(pred[0]) - SEARCH - R_T, 0), max(int(pred[1]) - SEARCH - R_T, 0)
        win = frames[j][y0:int(pred[1]) + SEARCH + R_T + 1, x0:int(pred[0]) + SEARCH + R_T + 1]
        if win.shape[0] <= tpl.shape[0] or win.shape[1] <= tpl.shape[1]:
            break
        _, mx, _, loc = cv2.minMaxLoc(cv2.matchTemplate(win, tpl, cv2.TM_CCOEFF_NORMED))
        nx, ny = x0 + loc[0] + R_T, y0 + loc[1] + R_T
        if ny - R_T < 0 or nx - R_T < 0:
            break
        patch = frames[j][ny - R_T:ny + R_T + 1, nx - R_T:nx + R_T + 1]
        if mx < NCC_MIN or float(cv2.matchTemplate(patch, seed_tpl, cv2.TM_CCOEFF_NORMED)[0, 0]) < NCC_MIN - 0.15:
            break
        vel, cx, cy = np.array([nx, ny]) - warped, float(nx), float(ny)   # camera-compensated displacement
        tpl = (0.8 * tpl + 0.2 * patch).astype(np.uint8)
        out.append((j, cx, cy, round(float(mx), 3)))
        i = j
    return out


def _body_attached(tr: dict, rel0: int, box: list, lo: int, all_steps: dict) -> bool:
    """True if a 'ball' just rides on a player (sock/boot). A real ball moves 14-16 px/frame relative to the
    nearest player, a sock 3.5. A seed within 8 px of a player box whose foot-offset changes < 8 px/frame on
    average (either direction) is body-attached. Not applied when no player is that close."""
    cx, cy = _centre(box)
    gaps = [(float(np.hypot(max(b["box"][0] - cx, 0, cx - b["box"][2]), max(b["box"][1] - cy, 0, cy - b["box"][3]))),
             b["track_id"]) for b in tr["frames"][rel0]["boxes"]]
    gaps = [g for g in gaps if g[0] <= 8.0]
    if not gaps:
        return False
    tid = min(gaps)[1]

    def foot(rel):
        return next((((b["box"][0] + b["box"][2]) / 2, b["box"][3]) for b in tr["frames"][rel]["boxes"]
                     if b["track_id"] == tid), None)

    f0 = foot(rel0)
    for steps in ([] if f0 is None else all_steps.values()):
        offs = [(cx - f0[0], cy - f0[1])]
        for j, x, y, _ in steps:
            if (f := foot(lo + j)) is None:
                break
            offs.append((x - f[0], y - f[1]))
        if len(offs) >= 4 and np.mean([np.hypot(a[0] - b[0], a[1] - b[1]) for a, b in zip(offs[1:], offs)]) < 8.0:
            return True
    return False


def _on_player_body(tr: dict, rel: int, cx: float, cy: float) -> bool:
    """A ball is at the feet, not on a torso or thigh: inside a player box above its lowest 20% is rejected."""
    return any(x0 <= cx <= x1 and y0 <= cy <= y0 + 0.8 * (y1 - y0)
               for x0, y0, x1, y1 in (b["box"] for b in tr["frames"][rel]["boxes"]))


def grow_ball(video: str, o: Opts, tr: dict, ball_by_rel: dict, tracked: dict) -> tuple[dict, dict]:
    """Extend confirmed boxes outward by template matching into frames nothing else covers. No API calls.

    Written 'tracked', never 'confirmed'. A step stops at NCC < GROW_NCC or on a player's torso/thigh; a
    body-attached seed is skipped. Never fills a frame already confirmed or already tracked.
    """
    n = o.ball_grow
    seeds = {rel: v["ball"]["box"] for rel, v in ball_by_rel.items()}
    stats = {"grow_seeds": len(seeds), "grow_frames": 0, "grow_body_attached": 0, "grow_window": n}
    added: dict[int, dict] = {}
    if n <= 0 or not seeds:
        return added, stats
    cap = cv2.VideoCapture(video)
    for rel0, box in sorted(seeds.items()):
        lo, hi = max(rel0 - n, 0), min(rel0 + n, tr["n_frames"] - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, tr["first"] + lo)
        gray = []
        for _ in range(lo, hi + 1):
            ok, im = cap.read()
            if not ok:
                break
            gray.append(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
        bw, bh = box[2] - box[0], box[3] - box[1]
        all_steps = {d: follow(gray, rel0 - lo, box, d, n) for d in (1, -1)}
        if _body_attached(tr, rel0, box, lo, all_steps):
            stats["grow_body_attached"] += 1
            continue
        for d, steps in all_steps.items():
            for k, (j, cx, cy, ncc) in enumerate(steps, 1):
                rel = lo + j
                if ncc < GROW_NCC or _on_player_body(tr, rel, cx, cy):
                    break
                if rel in ball_by_rel or rel in tracked or (rel in added and added[rel]["step"] <= k):
                    continue
                added[rel] = {"box": [int(cx - bw / 2), int(cy - bh / 2), int(cx + bw / 2) + 1, int(cy + bh / 2) + 1],
                              "ncc": ncc, "seed_rel": rel0, "step": k, "source": "grow", "confirmed_p": None}
    cap.release()
    stats["grow_frames"] = len(added)
    return added, stats


# --------------------------------------------------------------------------
# assemble + run
# --------------------------------------------------------------------------
def assemble(tr: dict, decisions: dict, ball_by_rel: dict, tracked: dict) -> dict:
    """frame rel -> {dets, counts, ball_tracked, ball_searched}. Only evidence we actually have."""
    out = {}
    for f in tr["frames"]:
        rel, dets = f["rel"], []
        for bx in f["boxes"]:
            dec = decisions.get(bx["track_id"])
            lab, conf, forced = label_for(dec, rel) if dec and dec["label"] else (None, None, False)
            if lab in OUTPUT_LABELS:
                dets.append({"box": bx["box"], "cls": lab, "confidence": conf, "localization": "tight",
                             "forced": forced, "track_id": bx["track_id"], "decision_type": "track_vote",
                             "yolo_score": bx["yolo_score"], "on_pitch": bx["on_pitch"], "segmented": dec["segmented"]})
        b = ball_by_rel.get(rel)
        if b:
            dets.append({"box": b["ball"]["box"], "cls": "ball", "confidence": b["ball"]["confidence"],
                         "localization": "tight", "forced": False, "track_id": None,
                         "decision_type": "ball_path_checkpoint", "yolo_score": None, "on_pitch": True, "segmented": False})
        bt = tracked.get(rel) if not b else None
        if bt:   # visual estimate: `ncc` is a template/YOLO score, not a probability
            bt = {**bt, "cls": "ball", "localization": "tracked", "decision_type": "ball_tracker", "forced": False}
        out[rel] = {"dets": dets, "ball_tracked": bt, "ball_searched": f["ball_searched"],
                    "counts": {c: sum(1 for d in dets if d["cls"] == c) for c in OUTPUT_LABELS}}
    return out


def project(tr: dict, qualified: list[dict], o: Opts, ball_calls: int) -> dict:
    """Calls a live run would spend (0.199 s mean latency measured earlier), shown before spending."""
    crops = sum(len(pick_crops(t, o.crops_per_track)) for t in qualified)
    forced = int(round(0.25 * len(qualified)))      # assumed: 25% of tracks need the forced pass
    total = 8 + crops + forced + ball_calls
    return {"frames": tr["n_frames"], "track_ids": len(tr["tracks"]), "tracks_classified": len(qualified),
            "kit_calls_est": 8, "track_crop_calls": crops, "forced_calls_est": forced, "ball_path_calls": ball_calls,
            "total_calls_est": total, "concurrent_seconds": round(total * 0.199 / max(o.workers, 1), 1)}


def run_clip(video: str, start: float, duration: float, out: str, dry_run: bool = True, **kw) -> dict:
    """Track a clip, vote once per track, find the ball. `kw` are `Opts` fields. Returns the result dict.

    dry_run=True (default): tracks and builds ball paths locally, writes dry_run.json with the projected call
    count, and makes ZERO API calls (no client is created). A live re-run replays answers from
    <out>/results/decisions_cache.json for free.
    """
    o = Opts(**kw)
    res_dir = Path(out) / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    cfg = Config(video=video, clip_start_s=start, clip_len_s=duration, out_dir=str(res_dir),
                 yolo_weights=o.yolo_weights, yolo_conf=o.conf, yolo_imgsz=o.imgsz, hud_mode=o.hud_mode, max_api_calls=o.max_calls,
                 k_clusters=o.kit_clusters, refine_depth=0, dry_run=dry_run)
    _, first, n_req = _window(video, start, duration)
    print(f"run_clip: {video} t={start}-{start + duration}s -> {out}  dry_run={dry_run}")
    hud, hud_rec = hud_mask(video, first, n_req, o.hud_samples, o.hud_mode)
    tr = track_clip(video, start, duration, cfg, hud, o)
    qualified, skipped = qualify_tracks(tr["tracks"], o)
    print(f"[1] {tr['n_frames']} frames, {len(tr['tracks'])} track ids, {len(qualified)} qualify, "
          f"{len(skipped)} skipped ({tr['track_seconds']}s)")
    bpaths, ncp = None, 0
    if o.ball_paths:
        bpaths = build_paths(video, tr["first"], tr["n_frames"], o.yolo_weights, hud, res_dir / "ball_paths.json")
        ncp = sum(len(checkpoints(p, o.ball_path_every)) for p in usable_paths(bpaths, o))
        print(f"[2] {len(bpaths['paths'])} ball paths, {len(usable_paths(bpaths, o))} usable, {ncp} checkpoint calls")
    projection = project(tr, qualified, o, ncp)
    if dry_run:
        print(f"[dry run] ZERO API calls. Projected ~{projection['total_calls_est']} calls, "
              f"~{projection['concurrent_seconds'] / 60:.1f} min concurrent")
        (res_dir / "dry_run.json").write_text(json.dumps(
            {"projection": projection, "hud_mask": hud_rec, "skipped_tracks": skipped}, indent=1, default=str))
        return {"dry_run": True, "projection": projection, "skipped_tracks": skipped, "api_calls": 0}

    dc = ConcurrentDecisionClient(cfg)
    if not dc.key_present:
        raise RuntimeError("OPENAI_API_KEY not found in the environment or ./.env - stopping rather than inventing results")
    dc.prime()
    ball_by_rel, tracked, ball_stats = {}, {}, {}
    try:
        kits = discover_kits(cfg, dc, tr, o)
        if not kits.get("resolved"):
            dc.save()
            raise RuntimeError(f"kits not resolved ({kits.get('reason')}): without a frozen team reference "
                               "every label would be a guess; nothing is fabricated")
        decisions, cls_stats = classify_tracks(cfg, dc, kits, qualified, o)
        print(f"[3] kits {kits['mapping']}; tracks {cls_stats}; {dc.calls} live calls")
        if bpaths is not None:
            ball_by_rel, tracked, ball_stats = confirm_ball_paths(cfg, dc, o, video, tr, bpaths)
            grown, gstats = grow_ball(video, o, tr, ball_by_rel, tracked)
            tracked.update(grown)
            ball_stats.update(gstats, tracked_frames=len(tracked))
            print(f"[4] ball: {len(ball_by_rel)} confirmed + {len(tracked)} tracked frames; {dc.calls} live calls")
    except KeyboardInterrupt:
        dc.save()
        print("interrupted - decisions cache saved; re-run the same command to resume", file=sys.stderr)
        raise
    dc.save()
    result = {"video": video, "start": start, "duration": duration, "fps": tr["fps"], "first_frame": tr["first"],
              "n_frames": tr["n_frames"], "projection": projection, "skipped_tracks": skipped,
              "kits": {"mapping": kits["mapping"], "referee_group": kits.get("referee_group")},
              "track_decisions": decisions, "track_stats": cls_stats, "ball_stats": ball_stats,
              "frames": assemble(tr, decisions, ball_by_rel, tracked), "api_calls": dc.calls,
              "cache_hits": dc.cache_hits, "calls_by_tag": dc.tag_counts, "budget_stop": dc.budget_stop,
              "failures": sum(1 for e in dc.log if not e.get("ok"))}
    (res_dir / "results.json").write_text(json.dumps(result, indent=1, default=str))
    print(f"live calls {dc.calls}, cache hits {dc.cache_hits}; wrote {res_dir / 'results.json'}")
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Track-based Decisions analysis of one clip. Pass --dry-run first "
                                 "(zero API calls); without it the run spends calls.")
    ap.add_argument("--video", required=True)
    ap.add_argument("--start", type=float, required=True)
    ap.add_argument("--duration", type=float, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry-run", action="store_true", help="track locally, project cost, make ZERO API calls")
    ap.add_argument("--yolo-weights", default=Opts.yolo_weights)
    ap.add_argument("--max-calls", type=int, default=Opts.max_calls)
    ap.add_argument("--ball-grow", type=int, default=Opts.ball_grow, help="growth frames each way (0 = off)")
    ap.add_argument("--no-ball", action="store_true", help="skip the ball finder")
    a = ap.parse_args(argv)
    try:
        run_clip(a.video, a.start, a.duration, a.out, dry_run=a.dry_run, yolo_weights=a.yolo_weights,
                 max_calls=a.max_calls, ball_grow=a.ball_grow, ball_paths=not a.no_ball)
    except KeyboardInterrupt:
        return 130
    except (RuntimeError, FileNotFoundError) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
