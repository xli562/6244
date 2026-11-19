"""Revisit a tied role using only the original words and the current public scene."""
import re
from copy import deepcopy
import numpy as np

from binding import TOL
from skill_library import SkillLibrary

VERB = {
    "place": "put|place|set|stack|move|transfer",
    "open": "open|pull open",
    "close": "close|shut",
    "stroke": "push|pull|turn|switch",
}
ALIASES = {"rear": "back", "center": "middle", "centre": "middle",
           "upper": "top", "lower": "bottom", "leftmost": "left", "rightmost": "right"}
DIRECTIONS = {"left", "right", "front", "back", "middle", "top", "bottom"}
STOP = set("the a an that at in of on to by from and both it them is its side region init".split())


def words(s):
    return {ALIASES.get(w, w) for w in re.findall("[a-z]+", s.lower())} - STOP


def role_spans(instruction, op, role):
    """Conservative grammar. Keep the source qualifier separate from the destination."""
    chunks = re.split(r"\s*(?:,|\bthen\b|\band\b(?=\s+(?:pick|put|place|set|stack|move|transfer|open|close|shut|push|pull|turn|switch)\b))\s*", instruction.lower())
    spans, antecedent = [], ""
    for chunk in chunks:
        pickup = re.match(r"pick (?:up )?(.+)", chunk)
        if pickup:
            antecedent = pickup[1]
        m = re.match(r"(?:please )?(?:" + VERB[op] + r")\s+(.+)", chunk)
        if not m:
            continue
        body = m[1]
        if op == "place":
            # The final placement preposition allows e.g. "the piece in the middle on the tray".
            preps = list(re.finditer(r"\s+(?:onto|into|inside|on|in|to|under|between|next to)\s+", body))
            if not preps:
                continue
            if len(preps) > 1:
                # Accept earlier source qualifiers only when their boundaries are explicit.
                earlier = [body[a.start():b.start()].strip() for a,b in zip(preps, preps[1:])]
                if not all(re.fullmatch(r"(?:in|on|to) (?:the )?(?:left|right|middle|back|front)(?: side)?", q) for q in earlier):
                    continue
            split = preps[-1]
            source = body[:split.start()].strip()
            dest = body[split.end():].strip()
            if source in {"it", "them"}:
                source = antecedent
            body = source if role == "item" else dest
        if body:
            spans.append(body)
    return list(dict.fromkeys(spans))


def revisit(instruction, step, role, original, catalog, pos, reader=None):
    tied = original.get("finalists", [])
    if len(tied) < 2:
        return original
    spans = role_spans(instruction, step["op"], role)
    quote = ""
    if reader is not None:
        try:
            proposed = reader(instruction, deepcopy(step), role)
            quote = proposed.lower().strip() if isinstance(proposed, str) else ""
        except Exception:
            pass
    # Cross-check even a model-supplied quote against independently scoped original words.
    selected = [s for s in spans if quote and re.search(r"(?<!\w)" + re.escape(quote) + r"(?!\w)", s)]
    if len(selected) == 1:
        phrase, via = selected[0], "agent"
    elif len(spans) == 1:
        phrase, via = spans[0], "parser"
    else:
        return dict(original, revisit={"reason": "no unique instruction span", "quote": quote})
    why = dict(phrase=phrase, quote=quote, via=via)
    def result(reason, name=None):
        return dict(original, revisit=dict(why, reason=reason), resolved=name,
                    status="unresolved" if name is None else
                    "confirmed" if name == original["pick"] else "corrected")

    tokens = words(phrase)
    if re.search(r"\b(?:left|right|front|back|side) of\b|\b(?:both|between|and|other|than)\b", phrase):
        return result("relative reference or multiple entities")
    labels = {n: words(n) for n in tied}
    relative = re.search(r"\b(?:beside|near|next to) (.+)$", phrase)
    if relative:
        anchor_terms = words(relative[1])
        landmarks = [n for n,v in catalog.items() if n not in tied and v["kind"] != "region"
                     and anchor_terms and anchor_terms <= words(n)]
        why["anchors"] = sorted(landmarks)
        if len(landmarks) != 1 or (words(phrase[:relative.start()]) & DIRECTIONS):
            return result("relative landmark or extra qualifier ambiguous")
        head = words(phrase[:relative.start()])
        extra = {w for w in head if any(w in v for v in labels.values())}
        eligible = [n for n in tied if extra <= labels[n]]
        dist = {n:float(np.linalg.norm(pos(n)[:2]-pos(landmarks[0])[:2])) for n in eligible}
        why["metres"] = {n:round(v,5) for n,v in dist.items()}
        closest = [n for n,d in dist.items() if np.isfinite(d) and d <= min(dist.values())+TOL]
        return result("unique nearest landmark" if len(closest)==1 else "still tied", closest[0] if len(closest)==1 else None)
    # All distinguishing literal words present in the role must agree on the same candidate.
    extra = {w for w in tokens if any(w in v for v in labels.values()) and not all(w in v for v in labels.values())}
    remaining = [n for n in tied if extra <= labels[n]]
    direction = tokens & DIRECTIONS
    why.update(terms=sorted(extra), direction=sorted(direction))
    if not remaining:
        return result("incompatible observed labels")
    if direction and not all(direction <= labels[n] for n in remaining):
        # A named spatial landmark must describe these entities, not just the direction.
        nouns = (set.intersection(*(labels[n] for n in remaining)) & tokens) - DIRECTIONS
        anchors = [n for n,v in catalog.items() if v["kind"] == "region" and n not in tied
                   and nouns and nouns | direction <= words(n)]
        why["anchors"] = sorted(anchors)
        if len(anchors) != 1:
            return result("no unique matching landmark")
        anchor = pos(anchors[0])[:2]
        distances = {n: float(np.linalg.norm(pos(n)[:2] - anchor)) for n in remaining}
        if not all(np.isfinite(v) for v in distances.values()):
            return result("missing position")
        why["metres"] = {n: round(v, 5) for n,v in distances.items()}
        remaining = [n for n,d in distances.items() if d <= min(distances.values()) + TOL]
    if len(remaining) == 1:
        return result("unique observed match", remaining[0])
    return result("still tied")


class ReviewLibrary(SkillLibrary):
    def __init__(self, sensors, mode, reader=None):
        super().__init__(sensors, mode in ("recover", "stop"))
        self.mode = mode
        self.reader = reader
        self.reviews = []

    def preview(self, step):
        evidence = super().preview(step)
        if not evidence:
            return evidence
        for role, ev in list(evidence.items()):
            if self.mode == "recover" and len(ev.get("finalists", [])) > 1:
                evidence[role] = revisit(self.instruction, step, role, ev, self.catalog(), self.pos, self.reader)
        if (step["op"] == "place" and evidence["item"]["resolved"] is not None
                and evidence["item"]["resolved"] == evidence["target"]["resolved"]):
            evidence["target"].update(resolved=None, status="unresolved", reason="Source equals destination")
        event = {"at": self.remaining(), "op": step["op"], "roles": evidence}
        if not self.reviews or event != self.reviews[-1]:
            self.reviews.append(event)
        return evidence
