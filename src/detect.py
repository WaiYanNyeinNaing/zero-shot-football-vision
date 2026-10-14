"""YOLO proposals gated to the pitch, kit colour clusters, kit validation and class-aware NMS."""

from __future__ import annotations

import cv2
import numpy as np

from config_io import Config, DecisionClient, image_part


def _iou(a: list[int], b: list[int]) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def pitch_mask(img: np.ndarray, hud: np.ndarray) -> np.ndarray:
    """Largest green region = the pitch. The ball lives here.

    Searching the stands only buys false positives: crowd texture produces far
    more small bright blobs than the pitch does. Limitation: a ball lofted above
    the stands line would fall outside this mask.
    """
    # Saturation >= 60 and a SMALL closing kernel both matter. Stadium
    # screens have bright green borders; a generous threshold plus a 41x41
    # close bridged them into the pitch component, dragging the detected top
    # edge from y~170 up to y~72 and letting stand figures and on-screen player
    # cards through as valid candidates.
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    g = cv2.inRange(hsv, (30, 60, 30), (95, 255, 255))
    g = cv2.morphologyEx(g, cv2.MORPH_CLOSE, np.ones((17, 17), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(g, 8)
    if n > 1:
        big = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
        g = (lab == big).astype(np.uint8) * 255
    g = cv2.morphologyEx(g, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    g = cv2.morphologyEx(g, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    g[hud == 0] = 0
    return g


def propose_objects(cfg: Config, frames: list[dict], mask: np.ndarray) -> list[dict]:
    """YOLO boxes, filtered to people actually standing on the pitch.

    Test the FEET, not the box: a player's head and shoulders rise above the
    grass, so requiring the whole box to sit inside the pitch would delete every
    player. The bottom-centre point is where they touch the ground.

    This is what removes spectators, dugout staff and stand figures. Without it
    they reach the classifier as genuine candidates, and a forced label turns a
    dark crowd patch into a player - or, once, into a ball.
    """
    from ultralytics import YOLO
    model = YOLO(cfg.yolo_weights)
    out: list[dict] = []
    for f in frames:
        pitch = pitch_mask(f["image"], mask)
        # Tolerance must be ASYMMETRIC. Growing the mask upward reaches into the
        # stands, which let stadium screens and dugout figures through as valid
        # candidates - a FUT player-card graphic was forced into a team that
        # way. Players stand on or below the pitch edge, never above it, so the
        # margin goes sideways and down only.
        mg = cfg.pitch_margin_px
        # anchor at the kernel's BOTTOM edge so output[y] = max(input[y-mg..y]),
        # i.e. the mask grows downward. Anchoring at the top does the opposite
        # and silently re-admits the stands.
        grown = cv2.dilate(pitch, np.ones((mg + 1, 2 * mg + 1), np.uint8), anchor=(mg, mg))
        h, w = grown.shape
        kw = {"imgsz": cfg.yolo_imgsz} if cfg.yolo_imgsz else {}   # unset: ultralytics default
        res = model(f["image"], verbose=False, conf=cfg.yolo_conf, **kw)[0]
        names = res.names
        rows, off_pitch = [], 0
        for b in res.boxes:
            x0, y0, x1, y1 = (int(v) for v in b.xyxy[0].tolist())
            cx = min(max((x0 + x1) // 2, 0), w - 1)
            feet = min(max(y1 - 2, 0), h - 1)
            if grown[feet, cx] == 0:
                off_pitch += 1
                continue
            rows.append({
                "frame_id": f["frame_id"],
                "box": [x0, y0, x1, y1],
                "proposal_source": "yolo+on_pitch",
                "yolo_label": names.get(int(b.cls[0]), "?"),   # metadata ONLY
                "yolo_score": round(float(b.conf[0]), 4),
            })
        rows.sort(key=lambda r: -r["yolo_score"])
        kept = rows[: cfg.max_proposals_per_frame]
        for i, r in enumerate(kept):
            r["proposal_id"] = f"{f['frame_id']}_p{i:02d}"
        out.extend(kept)
        f["proposals_dropped_by_budget"] = len(rows) - len(kept)
        f["proposals_off_pitch"] = off_pitch
    return out


# --------------------------------------------------------------------------
# step 4 - kit hypotheses from colour (bootstrap only)
# --------------------------------------------------------------------------
def _torso(img: np.ndarray, box: list[int]) -> np.ndarray:
    x0, y0, x1, y1 = box
    h = y1 - y0
    ty0, ty1 = y0 + int(0.18 * h), y0 + int(0.52 * h)
    return img[max(ty0, 0):max(ty1, ty0 + 1), x0:x1]


def _kit_colour(patch: np.ndarray, min_nongrass: float = 0.35) -> tuple[np.ndarray | None, float]:
    if patch.size == 0 or patch.shape[0] < 4 or patch.shape[1] < 3:
        return None, 0.0
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    not_grass = cv2.inRange(hsv, (30, 40, 20), (95, 255, 255)) == 0
    if not_grass.mean() < min_nongrass:
        return None, 0.0
    lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB).reshape(-1, 3)[not_grass.reshape(-1)]
    sharp = cv2.Laplacian(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
    return np.median(lab, axis=0).astype(np.float32), float(sharp)


def kit_hypotheses(frames: list[dict], proposals: list[dict], k: int = 5) -> dict:
    """cv2.kmeans over torso colours. A HYPOTHESIS, not a team assignment."""
    by_id = {f["frame_id"]: f for f in frames}
    feats, meta = [], []
    for p in proposals:
        if p["yolo_label"] != "person":
            continue
        x0, y0, x1, y1 = p["box"]
        if (y1 - y0) < 34 or (x1 - x0) < 12:
            continue
        col, sharp = _kit_colour(_torso(by_id[p["frame_id"]]["image"], p["box"]))
        if col is None or sharp < 12.0:
            continue
        feats.append(col)
        meta.append({**{x: p[x] for x in ("frame_id", "box", "proposal_id")},
                     "sharpness": round(sharp, 1)})
    if len(feats) < 4:
        return {"ok": False, "reason": f"only {len(feats)} usable torso crops", "groups": []}

    data = np.asarray(feats, np.float32)
    k = min(k, len(feats))
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    _, labels, centers = cv2.kmeans(data, k, None, crit, 10, cv2.KMEANS_PP_CENTERS)
    labels = labels.ravel()

    groups = []
    for g in range(k):
        members = [meta[i] for i in np.where(labels == g)[0]]
        if not members:
            continue
        members.sort(key=lambda m: (m["frame_id"], m["box"][0]))     # deterministic
        groups.append({
            "group_id": f"g{g}",
            "size": len(members),
            "lab_center": [round(float(v), 1) for v in centers[g]],
            "members": members,
            "earliest": (members[0]["frame_id"], members[0]["box"][0]),
        })
    groups = _merge_close(groups, lab_dist=25.0)
    groups.sort(key=lambda g: g["earliest"])                         # stable ordering
    return {"ok": True, "groups": groups, "usable_crops": len(feats)}


def _merge_close(groups: list[dict], lab_dist: float = 25.0) -> list[dict]:
    """Fold clusters whose kit colours are near-identical back together.

    k=5 is needed to split the red officials out of a team cluster, but it also
    splits one team in two (lit vs shaded side of the pitch). A fragmented team
    makes a weak reference montage, which measurably raised the uncertain count.
    Merging by Lab distance keeps the officials separate while restoring whole
    teams.
    """
    merged: list[dict] = []
    for g in sorted(groups, key=lambda x: -x["size"]):
        host = next((m for m in merged
                     if float(np.linalg.norm(np.array(m["lab_center"]) -
                                             np.array(g["lab_center"]))) < lab_dist), None)
        if host is None:
            merged.append({**g, "merged_from": [g["group_id"]]})
            continue
        n_h, n_g = host["size"], g["size"]
        host["lab_center"] = [round((a * n_h + b * n_g) / (n_h + n_g), 1)
                              for a, b in zip(host["lab_center"], g["lab_center"])]
        host["members"] = sorted(host["members"] + g["members"],
                                 key=lambda m: (m["frame_id"], m["box"][0]))
        host["size"] = n_h + n_g
        host["earliest"] = (host["members"][0]["frame_id"], host["members"][0]["box"][0])
        host["merged_from"].append(g["group_id"])
    return merged


def q_kit_group() -> list[dict]:
    return [
        {"type": "predicate", "name": "same_kit",
         "instructions": "Do all the shirt crops in this montage show players wearing the "
                         "same team kit? Ignore lighting, blur and compression."},
        {"type": "choice", "name": "group_role",
         "instructions": "What do these crops mostly show?",
         "choices": [
             {"value": "outfield_team", "description": "Outfield players of one team."},
             {"value": "goalkeeper", "description": "A goalkeeper in a distinct kit."},
             {"value": "official", "description": "Referee or assistant referee."},
             {"value": "mixed", "description": "More than one kit, or not players at all."},
         ]},
    ]


def q_kit_pair() -> list[dict]:
    return [{"type": "predicate", "name": "different_kits",
             "instructions": "The first image is kit A, the second is kit B. Are they two DIFFERENT "
                             "team kits (clearly different shirt colour or design), rather than the "
                             "same kit seen in different light, distance or blur? Answer true only "
                             "if the two kits are different."}]


# --------------------------------------------------------------------------
# step 5 - validate kit groups, then freeze the team mapping
# --------------------------------------------------------------------------
def montage(frames: list[dict], members: list[dict], n: int = 4, cell: int = 96) -> np.ndarray:
    by_id = {f["frame_id"]: f for f in frames}
    picks = sorted(members, key=lambda m: -m["sharpness"])[:n]
    cells = []
    for m in picks:
        x0, y0, x1, y1 = m["box"]
        crop = by_id[m["frame_id"]]["image"][y0:y1, x0:x1]
        if crop.size:
            cells.append(cv2.resize(crop, (cell, cell * 2)))
    if not cells:
        return np.zeros((cell * 2, cell, 3), np.uint8)
    return np.hstack(cells)


def validate_kits(cfg: Config, dc: DecisionClient, frames: list[dict], hyp: dict) -> dict:
    """Decisions judges each colour cluster; we keep the best outfield group plus the best one of a different kit."""
    if not hyp.get("ok"):
        return {"resolved": False, "reason": hyp.get("reason"), "groups": []}

    scored = []
    for g in hyp["groups"]:
        board = montage(frames, g["members"])
        cv2.imwrite(str(cfg.path("kits", f"{g['group_id']}_montage.png")), board)
        r = dc.ask([{"type": "input_text",
                     "text": f"Shirt crops from one colour cluster ({g['size']} members)."},
                    image_part(board)], q_kit_group(), tag=f"kit:{g['group_id']}")
        ans = {a["name"]: a for a in r.get("answers", [])}
        same = ans.get("same_kit", {}).get("probability")
        role = ans.get("group_role", {})
        scored.append({**{x: g[x] for x in ("group_id", "size", "lab_center", "earliest")},
                       "members": g["members"],
                       "same_kit_p": same,
                       "role": role.get("choice"), "role_conf": role.get("confidence"),
                       "source": r["source"], "error": r.get("error"),
                       "montage": str(cfg.path("kits", f"{g['group_id']}_montage.png"))})

    outfield = [g for g in scored
                if g["role"] == "outfield_team"
                and (g["same_kit_p"] or 0) >= cfg.t_kit_group]
    if len(outfield) < 2:
        return {"resolved": False, "groups": scored,
                "reason": f"{len(outfield)} validated outfield group(s); player team "
                          "assignment stays uncertain. Ball search continues.",
                "mapping": {}}

    # deterministic, documented rule: rank by evidence; the top cluster is a team (size
    # counts here). Every other validated cluster is asked "same kit or different kit" against
    # it (colour clustering can split one kit in two: near and far players). Of the clusters
    # judged DIFFERENT, team 2 is the one with the cleanest reference, highest
    # same_kit_p * role_conf; a tie goes to the larger cluster. The two teams are then ordered
    # by earliest (frame_id, box x) so the mapping never flips.
    ranked = sorted(outfield, key=lambda g: -(g["same_kit_p"] * (g["role_conf"] or 1) * g["size"]))
    first, second, pair_checks, different_groups = ranked[0], None, [], []
    a_img = cv2.resize(montage(frames, first["members"], 3), (288, 192), interpolation=cv2.INTER_CUBIC)
    for g in ranked[1:]:
        b_img = cv2.resize(montage(frames, g["members"], 3), (288, 192), interpolation=cv2.INTER_CUBIC)
        r = dc.ask([{"type": "input_text", "text": "Kit A:"}, image_part(a_img),
                    {"type": "input_text", "text": f"Kit B (a group of {g['size']} people):"},
                    image_part(b_img)], q_kit_pair(), tag=f"kitpair:{first['group_id']}:{g['group_id']}")
        a = next((x for x in r.get("answers", []) if x["name"] == "different_kits"), None)
        p = a.get("probability") if a and a["type"] == "predicate" else None
        different = (p or 0) >= cfg.t_kit_different
        pair_checks.append({"a": first["group_id"], "b": g["group_id"], "different_p": p,
                            "different": different, "source": r["source"], "error": r.get("error")})
        if different:
            different_groups.append(g)
        elif p is not None:     # judged the same kit: recorded, reference board unchanged
            g["merged_into"] = first["group_id"]
    if different_groups:
        second = max(different_groups,
                     key=lambda g: (g["same_kit_p"] * (g["role_conf"] or 1), g["size"]))
        for g in different_groups:
            if g is not second:
                g["outcome"] = "different_kit_not_chosen"
    if second is None:
        unanswered = any(c["different_p"] is None for c in pair_checks)
        return {"resolved": False, "groups": scored, "pair_checks": pair_checks, "mapping": {},
                "reason": "no second kit: " + (
                    "a same-kit/different-kit check got no answer (dry run, error or refusal)"
                    if unanswered else f"all {len(pair_checks)} other outfield group(s) were judged the "
                    "same kit as the top group") + "; player team assignment stays uncertain. "
                    "Ball search continues."}
    best = sorted([first, second], key=lambda g: g["earliest"])
    mapping = {best[0]["group_id"]: "team_1_player", best[1]["group_id"]: "team_2_player"}
    refs = []
    for slot, g in enumerate(best):
        ref = cv2.resize(montage(frames, g["members"], 3), (288, 192),
                         interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(str(cfg.path("kits", f"reference_team_{slot + 1}.png")), ref)
        refs.append(ref)
    # Officials cannot be told from shirt crops alone - red is a plausible team
    # kit, and the model duly called the referee cluster "outfield_team" with
    # 0.95 confidence. Only once the two TEAMS are fixed does the question make
    # sense, so we ask it here, with both team references in hand.
    ref_group = None
    chosen = {g["group_id"] for g in best}
    leftovers = [g for g in scored
                 if g["group_id"] not in chosen and not g.get("merged_into")
                 and (g["same_kit_p"] or 0) >= cfg.t_kit_group]
    for g in sorted(leftovers, key=lambda x: -(x["same_kit_p"] or 0)):
        rr = dc.ask([{"type": "input_text", "text": "TEAM 1 reference kit:"},
                     image_part(refs[0]),
                     {"type": "input_text", "text": "TEAM 2 reference kit:"},
                     image_part(refs[1]),
                     {"type": "input_text",
                      "text": f"A third group of {g['size']} people from the same match:"},
                     image_part(montage(frames, g["members"], 4))],
                    [{"type": "predicate", "name": "are_officials",
                      "instructions": "Are the people in the third image match officials "
                                      "(referee or assistants) rather than players of either "
                                      "team shown in the two reference images? Officials wear a "
                                      "kit distinct from both teams and there are only a few of "
                                      "them on the pitch."}],
                    tag=f"officials:{g['group_id']}")
        a = next((x for x in rr.get("answers", []) if x["name"] == "are_officials"), None)
        g["are_officials_p"] = a.get("probability") if a and a["type"] == "predicate" else None
        if (g["are_officials_p"] or 0) >= cfg.t_kit_group:
            ref_group = g
            break

    ref_img = None
    if ref_group:
        ref_img = cv2.resize(montage(frames, ref_group["members"], 3), (288, 192),
                             interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(str(cfg.path("kits", "reference_referee.png")), ref_img)
    parts = [refs[0], np.full((192, 8, 3), 255, np.uint8), refs[1]]
    if ref_img is not None:
        parts += [np.full((192, 8, 3), 255, np.uint8), ref_img]
    board = np.hstack(parts)
    cv2.imwrite(str(cfg.path("kits", "reference_board.png")), board)
    return {"resolved": True, "groups": scored, "mapping": mapping, "refs": refs,
            "pair_checks": pair_checks,
            "referee_ref": ref_img,
            "referee_group": ref_group and ref_group["group_id"],
            "rule": "rank by same_kit_p * role_conf * size; top = a team; every other group is asked same/different "
                    "kit against it; of those judged different, team 2 = highest same_kit_p * role_conf "
                    "(ties: larger group); order the two by earliest (frame_id, box x)",
            "note": "team_1/team_2 are arbitrary visual groups - not clubs, not home/away",
            "board": board, "board_path": str(cfg.path("kits", "reference_board.png"))}


def nms_class_aware(dets: list[dict], thr: float = 0.55) -> list[dict]:
    """Suppress duplicates within a class only; overlaps across classes survive.

    One extra rule: a coarse `ball_present_unlocalized` region is redundant once a
    confirmed tight ball has been found inside it in the same frame.
    """
    tight_balls = [d for d in dets if d["cls"] == "ball" and d["localization"] == "tight"]
    survivors = []
    for d in dets:
        if d["cls"] == "ball" and d["localization"] != "tight":
            inside = any(t["frame_id"] == d["frame_id"]
                         and d["box"][0] <= (t["box"][0] + t["box"][2]) / 2 <= d["box"][2]
                         and d["box"][1] <= (t["box"][1] + t["box"][3]) / 2 <= d["box"][3]
                         for t in tight_balls)
            if inside:
                continue
        survivors.append(d)
    dets = survivors

    keep, dropped = [], []
    for d in sorted(dets, key=lambda d: -(d.get("confidence") or 0)):
        rival = next((k for k in keep if k["frame_id"] == d["frame_id"]
                      and k["cls"] == d["cls"] and _iou(k["box"], d["box"]) > thr), None)
        if rival:
            dropped.append({**d, "suppressed_by": rival["proposal_id"]})
        else:
            keep.append(d)
    return keep, dropped


