"""Config, key loading, frame extraction, HUD mask and the cached, capped Decisions client.

Rules kept from the original lab: Decisions is the classifier, nothing is
fabricated, every API call is cached, counted and capped.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# --------------------------------------------------------------------------
# verified API facts (https://developers.openai.com/api/docs/guides/decisions)
# --------------------------------------------------------------------------
DECISIONS_MODEL = "gpt-6-luna"          # model name used for all Decisions calls
DECISIONS_ENDPOINT = "POST /v1/decisions"
DOC_URL = "https://developers.openai.com/api/docs/guides/decisions"
MAX_REQUEST_BYTES = 8 * 1024 * 1024     # our own ceiling; no official one published

CLASSES = ["team_1_player", "team_2_player", "ball", "referee", "background", "uncertain"]



# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------
@dataclass
class Config:
    video: str = "/path/to/match.mp4"   # placeholder: set to your own video
    clip_start_s: float = 60.0          # placeholder for your own video: window start inside a long video
    clip_len_s: float = 5.0
    n_frames: int = 5
    out_dir: str = "out"

    yolo_weights: str = "models/yolo11s.pt"
    yolo_conf: float = 0.15
    yolo_imgsz: int = 0                 # 0 = ultralytics default (640). Raise it for small players (30-50 px tall)
    hud_mode: str = "auto"              # "auto" or "none" (see estimate_hud_mask)

    # budgets (hard stops, not suggestions)
    # Measured on earlier test footage, not included in this repo: 12 regions/frame
    # covers only ~16% of the pitch and often never shows the model the ball.
    # 48 -> ~44% coverage, 96 -> ~72%. Coverage, not cleverness, drives recall.
    max_api_calls: int = 600
    max_proposals_per_frame: int = 80
    max_ball_regions_per_frame: int = 48
    refine_depth: int = 2
    max_retries: int = 2

    k_clusters: int = 5          # 3 merged referees into a team cluster; 5 separates them
    pitch_margin_px: int = 28    # tolerance so touchline players survive the pitch filter

    # experimental thresholds - NOT calibrated probabilities
    t_object_accept: float = 0.55
    t_region_ball: float = 0.50
    t_kit_group: float = 0.50
    t_kit_different: float = 0.50   # two outfield clusters count as different team kits at or above this

    dry_run: bool = True
    model: str = DECISIONS_MODEL

    def path(self, *parts: str) -> Path:
        p = Path(self.out_dir).joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p


# --------------------------------------------------------------------------
# step 1 - frames
# --------------------------------------------------------------------------
def extract_frames(cfg: Config) -> list[dict]:
    """Exactly cfg.n_frames distinct, decoded source frames, evenly spaced.

    Raises if the clip cannot supply that many distinct frames. Never
    duplicates a frame to hit the count.
    """
    cap = cv2.VideoCapture(cfg.video)
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {cfg.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0:
        cap.release()
        raise RuntimeError("video reports no usable fps")

    first = int(round(cfg.clip_start_s * fps))
    last = min(total - 1, int(round((cfg.clip_start_s + cfg.clip_len_s) * fps)) - 1)
    if last - first + 1 < cfg.n_frames:
        cap.release()
        raise RuntimeError(f"window {first}..{last} too short for {cfg.n_frames} frames")

    targets = sorted({int(round(x)) for x in np.linspace(first, last, cfg.n_frames)})
    if len(targets) < cfg.n_frames:
        cap.release()
        raise RuntimeError("evenly spaced targets collapsed to duplicates")

    # sequential decode: frame-accurate, unlike seeking into a GOP
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    frames, idx, want = [], first, list(targets)
    while want and idx <= last:
        ok, img = cap.read()
        if not ok:
            break
        if idx == want[0]:
            want.pop(0)
            name = f"frame_{len(frames):02d}_src{idx}.png"
            cv2.imwrite(str(cfg.path("frames", name)), img)
            frames.append({
                "frame_id": f"f{len(frames)}",
                "source_index": idx,
                "timestamp_s": round(idx / fps, 4),
                "width": img.shape[1],
                "height": img.shape[0],
                "file": str(cfg.path("frames", name)),
                "image": img,
            })
        idx += 1
    cap.release()

    if len(frames) < cfg.n_frames:
        raise RuntimeError(
            f"only {len(frames)} distinct frames decoded, need {cfg.n_frames} - "
            "no duplication applied")
    return frames


# --------------------------------------------------------------------------
# step 2 - conservative HUD mask (automatic, overridable)
# --------------------------------------------------------------------------
def estimate_hud_mask(frames: list[dict], override: dict | None = None, mode: str = "auto") -> tuple[np.ndarray, dict]:
    """Find the HUD automatically. Returns (mask, record); 255 = searchable.

    The naive tests fail on overlay-style broadcast HUDs: such a HUD is semi-transparent, so the panels read
    as *green* (the pitch shows through) and as *non-static* (the camera pans
    beneath them). What does hold is that overlay **edges land on identical
    pixels in every frame** while real scene edges move. So: keep edges present
    in all frames, restrict to the perimeter bands where HUD lives, then promote
    each persistent-edge cluster to its bounding rectangle - HUD panels are
    rectangles.

    Still deliberately timid: a greedy mask hides players, which is worse than
    searching a little HUD. Oversized or near-empty clusters are rejected, and if
    the result swallows >30% of the frame it is discarded and recorded instead.
    """
    h, w = frames[0]["image"].shape[:2]
    mask = np.full((h, w), 255, np.uint8)
    if mode == "none":   # static-edge detection needs a moving camera; with a fixed one the whole scene is "static"
        return mask, {"method": "none", "overridden": False, "panels": [], "rejected": [], "playable_fraction": 1.0}
    rec: dict = {"method": "persistent_edges+perimeter+bbox", "overridden": bool(override),
                 "panels": [], "rejected": []}

    if override:
        for (x0, y0, x1, y1) in override.get("rects", []):
            mask[y0:y1, x0:x1] = 0
        rec["rects"] = override.get("rects", [])
        rec["playable_fraction"] = round(float((mask > 0).mean()), 4)
        return mask, rec

    # edges that survive in every frame at the same pixel
    acc = np.full((h, w), 255, np.uint8)
    for f in frames:
        e = cv2.Canny(cv2.cvtColor(f["image"], cv2.COLOR_BGR2GRAY), 40, 110)
        acc = cv2.bitwise_and(acc, cv2.dilate(e, np.ones((3, 3), np.uint8)))

    band = np.zeros((h, w), np.uint8)
    band[: int(0.14 * h), :] = 255
    band[int(0.76 * h):, :] = 255
    band[:, : int(0.08 * w)] = 255
    band[:, int(0.92 * w):] = 255

    cand = cv2.morphologyEx(cv2.bitwise_and(acc, band), cv2.MORPH_CLOSE,
                            np.ones((13, 13), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(cand, 8)
    for i in range(1, n):
        x, y, bw, bh, area = (int(v) for v in stats[i])
        if area < 200:
            continue
        if bw * bh > 0.14 * h * w:
            rec["rejected"].append({"bbox": [x, y, bw, bh], "why": "oversized"})
            continue
        if area / (bw * bh) < 0.04:
            rec["rejected"].append({"bbox": [x, y, bw, bh], "why": "too sparse"})
            continue
        mask[y:y + bh, x:x + bw] = 0
        rec["panels"].append({"bbox": [x, y, bw, bh], "edge_px": area,
                              "fill": round(area / (bw * bh), 2)})

    if float((mask > 0).mean()) < 0.70:
        mask[:] = 255
        rec["backed_off"] = True
        rec["limitation"] = "mask covered >30% of the frame; disabled rather than hide players"
    rec["playable_fraction"] = round(float((mask > 0).mean()), 4)
    return mask, rec


# --------------------------------------------------------------------------
# Decisions client: cached, counted, capped
# --------------------------------------------------------------------------
def load_key(dotenv: str = ".env") -> str | None:
    """Process env wins; otherwise read only OPENAI_API_KEY from a local .env."""
    if os.environ.get("OPENAI_API_KEY"):
        return os.environ["OPENAI_API_KEY"]
    p = Path(dotenv)
    if not p.is_file():
        return None
    for line in p.read_text().splitlines():
        line = line.strip().removeprefix("export ").strip()
        if line.startswith("OPENAI_API_KEY="):
            return line.split("=", 1)[1].strip().strip("\"'") or None
    return None


def png_b64(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("png encode failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def image_part(img: np.ndarray) -> dict:
    return {"type": "input_image", "image_url": f"data:image/png;base64,{png_b64(img)}"}


class BudgetExhausted(RuntimeError):
    pass


class DecisionClient:
    """Thin wrapper over client.decisions.create with cache + hard caps."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.calls = 0
        self.cache_hits = 0
        self.log: list[dict] = []
        self.cache_path = cfg.path("decisions_cache.json")
        self.cache = json.loads(self.cache_path.read_text()) if self.cache_path.is_file() else {}
        self._client = None
        self.key_present = bool(load_key())

    def _lazy(self):
        if self._client is None:
            key = load_key()
            if not key:
                raise RuntimeError(
                    "OPENAI_API_KEY not found in the environment or ./.env - "
                    "stopping at the API step rather than inventing results.")
            from openai import OpenAI
            self._client = OpenAI(api_key=key)
        return self._client

    @staticmethod
    def _key(content: list[dict], questions: list[dict], model: str) -> str:
        blob = json.dumps({"c": content, "q": questions, "m": model}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    def ask(self, content: list[dict], questions: list[dict], tag: str) -> dict:
        """Returns {'answers': [...], 'source': cache|live|dry_run|error}."""
        payload = json.dumps({"input": [{"role": "user", "content": content}],
                              "questions": questions})
        size = len(payload.encode())
        if size > MAX_REQUEST_BYTES:
            return {"answers": [], "source": "error",
                    "error": f"payload {size}B over ceiling {MAX_REQUEST_BYTES}B - "
                             "subdivide instead of shrinking the evidence"}

        ck = self._key(content, questions, self.cfg.model)
        if ck in self.cache:
            self.cache_hits += 1
            return {**self.cache[ck], "source": "cache"}

        if self.cfg.dry_run:
            return {"answers": [], "source": "dry_run", "payload_bytes": size, "tag": tag}

        if self.calls >= self.cfg.max_api_calls:
            raise BudgetExhausted(f"hit max_api_calls={self.cfg.max_api_calls}")

        client = self._lazy()   # a missing key stops here, not inside the retry loop
        last_err = None
        for attempt in range(self.cfg.max_retries + 1):
            self.calls += 1
            t0 = time.time()
            try:
                d = client.decisions.create(
                    model=self.cfg.model,
                    input=[{"role": "user", "content": content}],
                    questions=questions,
                )
                dt = time.time() - t0
                res = {"answers": [a.model_dump() for a in d.answers],
                       "usage": getattr(d, "usage", None) and d.usage.model_dump(),
                       "latency_s": round(dt, 3), "tag": tag, "payload_bytes": size}
                self.cache[ck] = res
                self.save()     # a paid answer reaches disk at once, so an interrupted run loses nothing
                self.log.append({"tag": tag, "latency_s": res["latency_s"],
                                 "attempt": attempt, "ok": True, "bytes": size})
                return {**res, "source": "live"}
            except Exception as exc:                        # recorded, never silenced
                last_err = f"{type(exc).__name__}: {exc}"
                self.log.append({"tag": tag, "attempt": attempt, "ok": False,
                                 "error": last_err,
                                 "latency_s": round(time.time() - t0, 3)})
                if self.calls >= self.cfg.max_api_calls:
                    break
        return {"answers": [], "source": "error", "error": last_err, "tag": tag}

    def save(self) -> None:
        tmp = self.cache_path.with_name(self.cache_path.name + ".tmp")
        tmp.write_text(json.dumps(self.cache, indent=1))
        os.replace(tmp, self.cache_path)       # atomic: a crash mid-write cannot corrupt the cache

    def stats(self) -> dict:
        ok = [e for e in self.log if e.get("ok")]
        return {"live_calls": self.calls, "cache_hits": self.cache_hits,
                "failures": len([e for e in self.log if not e.get("ok")]),
                "mean_latency_s": round(float(np.mean([e["latency_s"] for e in ok])), 3) if ok else None,
                "budget": self.cfg.max_api_calls, "dry_run": self.cfg.dry_run}


