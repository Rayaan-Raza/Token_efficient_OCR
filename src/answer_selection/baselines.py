"""Equal-cost rule baselines for RAVEN-Select (no gold at inference)."""

from __future__ import annotations

from collections import Counter
from typing import Sequence

from src.routing import normalize_pred, predict_question_type
from src.routing.ensembles import pick_majority, pick_shortest_nonempty


def pick_resize_default(preds: dict[str, str], *, default: str = "resize") -> str:
    return default if default in preds else next(iter(preds))


def pick_bm25_default(preds: dict[str, str], *, default: str = "bm25") -> str:
    if default in preds and normalize_pred(preds.get(default, "")):
        return default
    return pick_resize_default(preds)


def pick_consensus_first(preds: dict[str, str], *, default: str = "resize") -> str:
    """Prefer majority agreement; fall back to shortest_nonempty."""
    norms = {m: normalize_pred(p) for m, p in preds.items()}
    nonempty = {m: n for m, n in norms.items() if n}
    if not nonempty:
        return default
    counts = Counter(nonempty.values())
    best_text, best_n = counts.most_common(1)[0]
    if best_n >= 2:
        for m in (default, *nonempty):
            if nonempty.get(m) == best_text:
                return m
    return pick_shortest_nonempty(preds, default=default)


def pick_answer_type_shortest(preds: dict[str, str], question: str, *, default: str = "resize") -> str:
    """Among type-matching nonempty preds, pick shortest; else shortest_nonempty."""
    from src.answer_selection.features import _pred_type, _type_match

    qtype = predict_question_type(question)
    matching = []
    for m, p in preds.items():
        n = normalize_pred(p)
        if not n:
            continue
        if _type_match(qtype, _pred_type(p)) > 0.5:
            matching.append((m, n))
    if matching:
        from src.answer_selection.method_spec import tie_rank

        matching.sort(key=lambda x: (len(x[1]), tie_rank(x[0])))
        return matching[0][0]
    return pick_shortest_nonempty(preds, default=default)


def pick_ocr_present_shortest(
    preds: dict[str, str],
    ocr_flags: dict[str, bool],
    *,
    question: str = "",
    default: str = "resize",
    use_type: bool = False,
    use_length: bool = True,
) -> str:
    """Prefer nonempty preds present in OCR; among them shortest (optional type filter)."""
    from src.answer_selection.features import _pred_type, _type_match

    present = []
    for m, p in preds.items():
        n = normalize_pred(p)
        if not n:
            continue
        if ocr_flags.get(m, False):
            present.append((m, n, p))
    if not present:
        return pick_shortest_nonempty(preds, default=default)

    pool = present
    if use_type and question:
        qtype = predict_question_type(question)
        typed = [(m, n, p) for m, n, p in present if _type_match(qtype, _pred_type(p)) > 0.5]
        if typed:
            pool = typed

    from src.answer_selection.method_spec import tie_rank

    if use_length:
        pool = sorted(pool, key=lambda x: (len(x[1]), tie_rank(x[0])))
        return pool[0][0]
    # No length: deterministic route priority (resize → BM25 → LER-BOPS).
    pool = sorted(pool, key=lambda x: tie_rank(x[0]))
    return pool[0][0]


def pick_raven_select_primary(
    preds: dict[str, str],
    question: str,
    ocr_flags: dict[str, bool],
    *,
    default: str = "resize",
    use_ocr: bool = True,
    use_answer_type: bool = False,
    use_length: bool = True,
    use_consensus: bool = False,
) -> str:
    """Primary RAVEN-Select rule: OCR-present + shortest (type/consensus off by default).

    On DocVQA n=500 this beats shortest_nonempty with CI lower > 0 vs resize and
    shortest. Optional type/consensus flags are for ablations / variants.
    """
    if not use_ocr:
        if use_answer_type:
            return pick_answer_type_shortest(preds, question, default=default)
        return pick_shortest_nonempty(preds, default=default)

    if use_consensus:
        norms = {m: normalize_pred(p) for m, p in preds.items()}
        ocr_nonempty = {m: n for m, n in norms.items() if n and ocr_flags.get(m, False)}
        if ocr_nonempty:
            counts = Counter(ocr_nonempty.values())
            best_text, best_n = counts.most_common(1)[0]
            if best_n >= 2:
                cands = [(m, n) for m, n in ocr_nonempty.items() if n == best_text]
                from src.answer_selection.method_spec import tie_rank

                if use_length:
                    cands.sort(key=lambda x: (len(x[1]), tie_rank(x[0])))
                else:
                    cands.sort(key=lambda x: tie_rank(x[0]))
                return cands[0][0]

    return pick_ocr_present_shortest(
        preds,
        ocr_flags,
        question=question,
        default=default,
        use_type=use_answer_type,
        use_length=use_length,
    )


def pick_raven_select_rule(*args, **kwargs) -> str:
    """Alias for ``pick_raven_select_primary`` (legacy name)."""
    return pick_raven_select_primary(*args, **kwargs)


BASELINE_NAMES = [
    "resize",
    "bm25",
    "ler_bops",
    "shortest_nonempty",
    "majority",
    "resize_default",
    "bm25_default",
    "answer_type_shortest",
    "ocr_present_shortest",
    "consensus_first",
    "raven_select_rule",
]


def choose_baseline(
    name: str,
    preds: dict[str, str],
    *,
    question: str = "",
    ocr_flags: dict[str, bool] | None = None,
    default: str = "resize",
    rule_flags: dict[str, bool] | None = None,
) -> str:
    if name in ("resize", "bm25", "ler_bops"):
        return name if name in preds else default
    if name == "shortest_nonempty":
        return pick_shortest_nonempty(preds, default=default)
    if name == "majority":
        return pick_majority(preds, default=default)
    if name == "resize_default":
        return pick_resize_default(preds, default="resize")
    if name == "bm25_default":
        return pick_bm25_default(preds, default="bm25")
    if name == "answer_type_shortest":
        return pick_answer_type_shortest(preds, question, default=default)
    if name == "ocr_present_shortest":
        return pick_ocr_present_shortest(preds, ocr_flags or {}, question=question, default=default)
    if name == "consensus_first":
        return pick_consensus_first(preds, default=default)
    if name in ("raven_select_rule", "raven_select"):
        flags = rule_flags or {}
        return pick_raven_select_primary(
            preds,
            question,
            ocr_flags or {},
            default=default,
            use_ocr=flags.get("use_ocr", True),
            use_answer_type=flags.get("use_answer_type", False),
            use_length=flags.get("use_length", True),
            use_consensus=flags.get("use_consensus", False),
        )
    raise ValueError(f"Unknown baseline: {name}")


def evaluate_baselines(
    data,
    methods: Sequence[str],
    *,
    ocr_presence=None,
    default: str = "resize",
    rule_flags: dict[str, bool] | None = None,
) -> dict[str, dict]:
    """Score each baseline; return name -> {anls, em, route_counts, anls_vec, em_vec}."""
    import numpy as np

    ids = list(data.index)
    results: dict[str, dict] = {}
    for name in BASELINE_NAMES:
        anls_list = []
        em_list = []
        counts: dict[str, int] = {m: 0 for m in methods}
        for iid in ids:
            preds = {m: str(data.loc[iid, f"pred__{m}"]) for m in methods}
            ocr_flags = {}
            if ocr_presence is not None:
                for m in methods:
                    key = (iid, m)
                    if hasattr(ocr_presence, "index") and key in ocr_presence.index:
                        row = ocr_presence.loc[key]
                        ocr_flags[m] = bool(row.get("pred_in_full_ocr", False) or row.get("pred_in_patch_ocr", False))
            pick = choose_baseline(
                name,
                preds,
                question=str(data.loc[iid, "question"]),
                ocr_flags=ocr_flags,
                default=default,
                rule_flags=rule_flags,
            )
            if pick not in methods:
                pick = default
            counts[pick] = counts.get(pick, 0) + 1
            anls_list.append(float(data.loc[iid, f"anls__{pick}"]))
            em_list.append(float(data.loc[iid, f"em__{pick}"]))
        results[name] = {
            "anls": float(np.mean(anls_list)),
            "em": float(np.mean(em_list)),
            "route_counts": counts,
            "anls_vec": anls_list,
            "em_vec": em_list,
        }
    return results
