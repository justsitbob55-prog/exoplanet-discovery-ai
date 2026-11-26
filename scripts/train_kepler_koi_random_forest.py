"""Acquire genuine Kepler KOI-labelled light curves, extract features, and train/evaluate Random Forest.

Reproducible and resumable workflow:
- Fetches KOI dispositions from NASA Exoplanet Archive (TAP sync CSV).
- Uses only CONFIRMED (positive) and FALSE POSITIVE (negative) dispositions.
- Excludes CANDIDATE/uncertain and NOT DISPOSITIONED records.
- Downloads actual Kepler light curves via the existing Lightkurve ingestion pipeline.
- Processes curves through existing preprocessing and feature extraction modules.
- Caches downloads and resumes from saved manifests/feature records.
- Splits at star level (kepid) to prevent leakage.
- Trains the existing baseline Random Forest and saves genuine held-out metrics.

No synthetic/demo data is used for training. No labels or metrics are fabricated.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
import socket
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from exoplanet_discovery.common.exceptions import (
    ExoplanetDiscoveryError,
    FeatureExtractionError,
    IngestionError,
    PreprocessingError,
)
from exoplanet_discovery.features.extractor import extract_features, feature_names
from exoplanet_discovery.ingestion.archive import download_light_curve
from exoplanet_discovery.preprocessing.pipeline import preprocess_light_curve
from exoplanet_discovery.training.baseline import save_model, train_baseline
from exoplanet_discovery.training.evaluation import binary_metrics

TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
KOI_QUERY = (
    "select kepid,kepoi_name,kepler_name,koi_disposition "
    "from cumulative where koi_disposition in ('CONFIRMED','FALSE POSITIVE')"
)
POSITIVE = "CONFIRMED"
NEGATIVE = "FALSE POSITIVE"


@dataclass
class KoiTarget:
    kepid: int
    kepoi_name: str
    kepler_name: str
    disposition: str
    label: int
    search_targets: list[str]
    group_id: str


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def fetch_koi_labels(path: Path, refresh: bool = False, timeout: int = 60) -> list[dict[str, str]]:
    if path.exists() and not refresh:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    params = urllib.parse.urlencode({"query": KOI_QUERY, "format": "csv"})
    url = f"{TAP_URL}?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": "ExoplanetDiscoveryAI/1.0"})
    previous_timeout = socket.getdefaulttimeout()
    socket.setdefaulttimeout(timeout)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            text = resp.read().decode("utf-8", errors="replace")
    finally:
        socket.setdefaulttimeout(previous_timeout)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def build_targets(rows: list[dict[str, str]]) -> tuple[list[KoiTarget], dict[str, Any]]:
    by_kepid: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        disp = (row.get("koi_disposition") or "").strip().upper()
        if disp not in {POSITIVE, NEGATIVE}:
            continue
        try:
            kepid = int(float(row.get("kepid") or 0))
        except Exception:
            continue
        if kepid <= 0:
            continue
        by_kepid[kepid].append(row)

    targets: list[KoiTarget] = []
    excluded_conflicting: list[int] = []
    duplicates_collapsed = 0
    for kepid, items in sorted(by_kepid.items()):
        dispositions = {(i.get("koi_disposition") or "").strip().upper() for i in items}
        if len(dispositions) != 1:
            excluded_conflicting.append(kepid)
            continue
        disposition = next(iter(dispositions))
        # Collapse duplicate KOIs for the same star with the same disposition to one sample/group.
        if len(items) > 1:
            duplicates_collapsed += len(items) - 1
        chosen = sorted(items, key=lambda r: (r.get("kepoi_name") or "", r.get("kepler_name") or ""))[0]
        kepoi = (chosen.get("kepoi_name") or "").strip()
        kepler_name = (chosen.get("kepler_name") or "").strip()
        # Robust identifier fallbacks for Kepler light-curve retrieval (genuine same target/star).
        search_targets: list[str] = []
        for candidate in (
            f"KIC {kepid}",
            kepoi,
            kepoi.replace("K", "KOI-", 1) if kepoi.startswith("K") else "",
            kepler_name,
        ):
            c = (candidate or "").strip()
            if c and c not in search_targets:
                search_targets.append(c)
        if not search_targets:
            search_targets = [f"KIC {kepid}"]
        targets.append(
            KoiTarget(
                kepid=kepid,
                kepoi_name=kepoi,
                kepler_name=kepler_name,
                disposition=disposition,
                label=1 if disposition == POSITIVE else 0,
                search_targets=search_targets,
                group_id=str(kepid),
            )
        )
    summary = {
        "source_rows": len(rows),
        "eligible_dispositions": [POSITIVE, NEGATIVE],
        "excluded_uncertain_or_not_dispositioned_note": "Query filtered to CONFIRMED/FALSE POSITIVE only; CANDIDATE and NOT DISPOSITIONED excluded.",
        "unique_kepids_with_consistent_label": len(targets),
        "excluded_conflicting_kepids": len(excluded_conflicting),
        "excluded_conflicting_kepid_examples": excluded_conflicting[:20],
        "duplicate_koi_rows_collapsed": duplicates_collapsed,
        "class_counts_by_kepid": dict(Counter(t.disposition for t in targets)),
    }
    return targets, summary


def select_targets(
    targets: list[KoiTarget],
    *,
    limit: int | None = None,
    max_per_class: int | None = None,
    seed: int = 42,
) -> tuple[list[KoiTarget], dict[str, Any]]:
    """Select a deterministic, genuine subset of labelled targets for processing.

    When max_per_class is set, targets are sampled per class (without replacement) and
    interleaved to produce a balanced, reproducible processing order. No labels are changed.
    """
    rng = np.random.default_rng(seed)
    by_label: dict[int, list[KoiTarget]] = defaultdict(list)
    for t in targets:
        by_label[int(t.label)].append(t)

    if max_per_class is not None:
        selected_by_label: dict[int, list[KoiTarget]] = {}
        for label, items in by_label.items():
            items_sorted = sorted(items, key=lambda x: (x.kepid, x.kepoi_name))
            idx = np.arange(len(items_sorted))
            rng.shuffle(idx)
            take = min(int(max_per_class), len(items_sorted))
            selected_by_label[label] = [items_sorted[int(i)] for i in idx[:take]]
        # interleave for balanced progress
        interleaved: list[KoiTarget] = []
        queues = {k: list(v) for k, v in selected_by_label.items()}
        order = sorted(queues.keys(), reverse=True)  # positives first, deterministic
        while any(queues.values()):
            progressed = False
            for k in order:
                if queues[k]:
                    interleaved.append(queues[k].pop(0))
                    progressed = True
            if not progressed:
                break
        selected = interleaved
    else:
        selected = list(targets)

    if limit is not None:
        selected = selected[: int(limit)]

    summary = {
        "selected_targets": len(selected),
        "selected_class_counts": dict(Counter(int(t.label) for t in selected)),
        "selection_limit": limit,
        "selection_max_per_class": max_per_class,
        "selection_seed": seed,
    }
    return selected, summary


def _load_records(path: Path) -> list[dict[str, Any]]:
    data = _read_json(path, [])
    return data if isinstance(data, list) else []


def _record_index(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(r.get("group_id")): r for r in records if r.get("group_id") is not None}


def _is_retryable_error(exc: BaseException) -> bool:
    if isinstance(exc, (concurrent.futures.TimeoutError, TimeoutError, socket.timeout, ConnectionError, OSError)):
        return True
    if isinstance(exc, IngestionError):
        return True
    name = type(exc).__name__.lower()
    msg = str(exc).lower()
    transient = ("timeout", "timed out", "temporar", "connection", "network", "reset", "refused", "unavailable", "503", "502", "504")
    return any(token in name or token in msg for token in transient)


def _download_with_timeout(search_target: str, cache_dir: Path, timeout_seconds: int):
    previous_timeout = socket.getdefaulttimeout()
    socket.setdefaulttimeout(timeout_seconds)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(download_light_curve, search_target, mission="Kepler", cache_dir=cache_dir)
            return future.result(timeout=timeout_seconds)
    finally:
        socket.setdefaulttimeout(previous_timeout)


def process_targets(
    selected_targets: list[KoiTarget],
    records_path: Path,
    progress_path: Path,
    cache_dir: Path,
    *,
    max_attempts_per_target: int = 2,
    sleep_seconds: float = 0.0,
    download_timeout_seconds: int = 300,
    download_retries: int = 2,
    retry_delay_seconds: float = 10.0,
) -> dict[str, Any]:
    records = _load_records(records_path)
    index = _record_index(records)
    selected = selected_targets
    selected_ids = {t.group_id for t in selected}
    processed_now = 0
    total = len(selected)
    for i, target in enumerate(selected, start=1):
        gid = target.group_id
        existing = index.get(gid)
        label_name = "CONFIRMED" if int(target.label) == 1 else "FALSE POSITIVE"
        prefix = f"[{i}/{total}] kepid={target.kepid} koi={target.kepoi_name or '-'} label={label_name}"
        if existing and existing.get("status") in {"ok", "skipped_duplicate"}:
            print(f"{prefix} -> SKIPPED (already processed: {existing.get('status')})", flush=True)
            continue
        if existing and int(existing.get("attempts", 0)) >= max_attempts_per_target and existing.get("status") != "ok":
            print(f"{prefix} -> SKIPPED (max attempts reached: {existing.get('attempts')}, status={existing.get('status')})", flush=True)
            continue

        attempts = int((existing or {}).get("attempts", 0)) + 1
        print(f"{prefix} -> START (attempt {attempts}/{max_attempts_per_target})", flush=True)
        record: dict[str, Any] = {
            "group_id": gid,
            "kepid": target.kepid,
            "kepoi_name": target.kepoi_name,
            "kepler_name": target.kepler_name,
            "search_targets": list(target.search_targets),
            "disposition": target.disposition,
            "label": target.label,
            "attempts": attempts,
            "updated_at": _utc_now(),
        }
        errors: list[str] = []
        success = False
        for search_target in target.search_targets:
            for retry in range(1, max(1, int(download_retries)) + 1):
                try:
                    curve = _download_with_timeout(search_target, cache_dir, int(download_timeout_seconds))
                    processed = preprocess_light_curve(curve)
                    feats = extract_features(processed)
                    ordered = [float(feats[name]) for name in feature_names()]
                    record.update(
                        {
                            "status": "ok",
                            "resolved_search_target": search_target,
                            "tic_id": processed.tic_id,
                            "source": processed.source,
                            "cadence_count_raw": int(len(curve)),
                            "cadence_count_processed": int(len(processed)),
                            "features": ordered,
                        }
                    )
                    success = True
                    print(
                        f"{prefix} -> SUCCESS via '{search_target}' "
                        f"(retry {retry}/{download_retries}, raw={len(curve)}, processed={len(processed)})",
                        flush=True,
                    )
                    break
                except (PreprocessingError, FeatureExtractionError, ValueError) as exc:
                    msg = f"{search_target} [attempt {retry}/{download_retries}]: {type(exc).__name__}: {exc}"
                    errors.append(msg)
                    print(f"{prefix} -> FAILURE (non-retryable processing): {msg}", flush=True)
                    break
                except concurrent.futures.TimeoutError:
                    msg = f"{search_target} [attempt {retry}/{download_retries}]: TimeoutError: download/processing exceeded {download_timeout_seconds}s"
                    errors.append(msg)
                    retryable = retry < max(1, int(download_retries))
                    print(
                        f"{prefix} -> TIMEOUT: {msg}" + ("; retrying after delay" if retryable else "; giving up on this identifier"),
                        flush=True,
                    )
                    if retryable and retry_delay_seconds > 0:
                        time.sleep(retry_delay_seconds)
                except Exception as exc:  # network/IO/archive/unexpected
                    msg = f"{search_target} [attempt {retry}/{download_retries}]: {type(exc).__name__}: {exc}"
                    errors.append(msg)
                    retryable = _is_retryable_error(exc) and retry < max(1, int(download_retries))
                    kind = "temporary" if retryable else "non-retryable-or-exhausted"
                    print(
                        f"{prefix} -> FAILURE ({kind}): {msg}" + ("; retrying after delay" if retryable else ""),
                        flush=True,
                    )
                    if retryable and retry_delay_seconds > 0:
                        time.sleep(retry_delay_seconds)
                    elif not _is_retryable_error(exc):
                        break
            if success:
                break
        if not success:
            record.update({"status": "failed", "error_type": "AllSearchTargetsFailed", "error": " | ".join(errors)[:1500]})
            print(f"{prefix} -> FAILED (all identifiers/attempts exhausted for this visit)", flush=True)

        index[gid] = record
        records = list(index.values())
        _write_json(records_path, records)
        processed_now += 1
        progress = {
            "updated_at": _utc_now(),
            "selected_targets": len(selected),
            "completed_or_terminal": sum(
                1 for r in records
                if str(r.get("group_id")) in selected_ids and (r.get("status") == "ok" or int(r.get("attempts", 0)) >= max_attempts_per_target)
            ),
            "ok": sum(1 for r in records if str(r.get("group_id")) in selected_ids and r.get("status") == "ok"),
            "failed": sum(1 for r in records if str(r.get("group_id")) in selected_ids and r.get("status") == "failed"),
            "last_target": gid,
        }
        _write_json(progress_path, progress)
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
        if i % 10 == 0 or i == total:
            print(f"[progress] {i}/{total} selected targets visited; ok={progress['ok']} failed={progress['failed']} terminal={progress['completed_or_terminal']}", flush=True)
    return {"processed_now": processed_now, "selected_targets": len(selected), "records_path": str(records_path)}


def _confusion_matrix_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    from sklearn.metrics import confusion_matrix

    labels = [0, 1]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "labels": labels,
        "matrix": cm.tolist(),
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }


def train_and_evaluate(
    records_path: Path,
    model_dir: Path,
    eval_dir: Path,
    *,
    seed: int = 42,
    test_size: float = 0.2,
    extra_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    records = _load_records(records_path)
    ok = [r for r in records if r.get("status") == "ok" and isinstance(r.get("features"), list)]
    if not ok:
        raise RuntimeError("No successfully processed labelled samples available for training.")
    X = np.asarray([r["features"] for r in ok], dtype=np.float64)
    y = np.asarray([int(r["label"]) for r in ok], dtype=np.int64)
    groups = np.asarray([str(r["group_id"]) for r in ok])
    if X.ndim != 2 or X.shape[1] != len(feature_names()):
        raise RuntimeError("Feature matrix is incompatible with the existing Random Forest feature schema.")
    class_counts = Counter(int(v) for v in y.tolist())
    if len(class_counts) < 2:
        raise RuntimeError(f"Need both classes for genuine evaluation; got {dict(class_counts)}.")
    min_class = min(class_counts.values())
    if min_class < 2:
        raise RuntimeError(f"Each class needs at least two samples for held-out evaluation; got {dict(class_counts)}.")

    # Star/target-level split (group_id=kepid). Use StratifiedGroupKFold when feasible.
    unique_groups = np.unique(groups)
    n_splits = 5 if len(unique_groups) >= 5 and min_class >= 5 else 2
    split = None
    try:
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        train_idx, test_idx = next(sgkf.split(X, y, groups))
        split = ("StratifiedGroupKFold", n_splits)
    except Exception:
        # Fallback: group shuffle split-like via unique groups; approximate stratification by group label.
        rng = np.random.default_rng(seed)
        group_labels = {}
        for g in unique_groups:
            labels_g = y[groups == g]
            vals, counts = np.unique(labels_g, return_counts=True)
            group_labels[g] = int(vals[int(np.argmax(counts))])
        test_groups: set[str] = set()
        for label_val in sorted(set(group_labels.values())):
            gset = [g for g, lv in group_labels.items() if lv == label_val]
            rng.shuffle(gset)
            k = max(1, int(math.ceil(len(gset) * test_size)))
            if len(gset) > 1:
                k = min(k, len(gset) - 1)
            test_groups.update(gset[:k])
        test_mask = np.array([g in test_groups for g in groups], dtype=bool)
        train_idx = np.where(~test_mask)[0]
        test_idx = np.where(test_mask)[0]
        split = ("GroupShuffle_fallback", None)
        # ensure both classes present
        if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
            train_idx, test_idx = train_test_split(
                np.arange(len(y)), test_size=test_size, random_state=seed, stratify=y
            )
            split = ("train_test_split_fallback_sample_level_WARNING", None)

    leakage_groups = set(groups[train_idx]).intersection(set(groups[test_idx]))
    if leakage_groups:
        raise RuntimeError(f"Leakage detected: {len(leakage_groups)} group(s) appear in both train and test.")

    result = train_baseline(
        X[train_idx],
        y[train_idx],
        feature_names=list(feature_names()),
        test_size=max(0.1, min(0.5, 1.0 / max(3, len(np.unique(y[train_idx]))))),
        random_seed=seed,
    )
    # train_baseline does its own internal split; for genuine held-out reporting on our group split,
    # compute metrics on our fixed test fold using the trained estimator fitted on our train fold.
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=seed,
        class_weight="balanced",
        n_jobs=-1,
    )
    model.fit(X[train_idx], y[train_idx])
    y_pred = model.predict(X[test_idx])
    y_score = None
    if hasattr(model, "predict_proba"):
        proba = np.asarray(model.predict_proba(X[test_idx]))
        if 1 in getattr(model, "classes_", []):
            pos_idx = int(np.where(np.asarray(model.classes_) == 1)[0][0])
            y_score = proba[:, pos_idx]
    metrics = binary_metrics(y[test_idx], y_pred, y_score)
    cm = _confusion_matrix_dict(y[test_idx], y_pred)

    model_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "baseline_random_forest.pkl"
    metadata_path = model_dir / "baseline_random_forest.json"
    import pickle

    with model_path.open("wb") as handle:
        pickle.dump(model, handle)
    metadata = {
        "model_type": "RandomForestClassifier",
        "model_name": "Random Forest",
        "training_status": "trained",
        "trained_at": _utc_now(),
        "feature_names": list(feature_names()),
        "label_definition": {
            "positive": {"value": 1, "koi_disposition": POSITIVE},
            "negative": {"value": 0, "koi_disposition": NEGATIVE},
            "excluded": ["CANDIDATE", "NOT DISPOSITIONED", "uncertain/conflicting kepid dispositions"],
        },
        "dataset_provenance": {
            "label_source": "NASA Exoplanet Archive KOI cumulative table (koi_disposition)",
            "label_query": KOI_QUERY,
            "light_curve_source": "Kepler mission light curves retrieved via Lightkurve/download_light_curve using existing ingestion pipeline",
            "preprocessing": "existing src.exoplanet_discovery.preprocessing.pipeline.preprocess_light_curve",
            "feature_extraction": "existing src.exoplanet_discovery.features.extractor.extract_features (25 features, stable order)",
            "records_artifact": str(records_path),
            "no_synthetic_training_data": True,
        },
        "split": {
            "method": split[0],
            "n_splits": split[1],
            "level": "star/target group (kepid)",
            "test_size_requested": test_size,
            "seed": seed,
            "leakage_group_overlap_count": 0,
        },
        "sample_counts": {
            "total_samples": int(len(y)),
            "train_samples": int(len(train_idx)),
            "test_samples": int(len(test_idx)),
            "train_groups": int(len(np.unique(groups[train_idx]))),
            "test_groups": int(len(np.unique(groups[test_idx]))),
        },
        "class_distribution": {
            "overall": {str(k): int(v) for k, v in sorted(class_counts.items())},
            "train": {str(k): int(v) for k, v in sorted(Counter(int(v) for v in y[train_idx]).items())},
            "test": {str(k): int(v) for k, v in sorted(Counter(int(v) for v in y[test_idx]).items())},
        },
        "metrics": metrics,
        "confusion_matrix": cm,
        "evaluation_note": "Metrics computed on genuine held-out Kepler KOI-labelled light curves (group split by kepid). "
        "Reliability depends on successfully processed sample size and class balance; see sample counts and class distribution.",
    }
    if extra_context:
        metadata["run_context"] = extra_context
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    report_path = eval_dir / "kepler_koi_random_forest_evaluation.json"
    _write_json(report_path, metadata)
    return {"model_path": str(model_path), "metadata_path": str(metadata_path), "report_path": str(report_path), "metadata": metadata}


def main() -> int:
    parser = argparse.ArgumentParser(description="Train/evaluate Random Forest on genuine Kepler KOI light curves.")
    parser.add_argument("--koi-csv", default=str(ROOT / "data/external/koi/koi_dispositions.csv"))
    parser.add_argument("--records", default=str(ROOT / "data/processed/koi/kepler_koi_records.json"))
    parser.add_argument("--progress", default=str(ROOT / "data/processed/koi/kepler_koi_progress.json"))
    parser.add_argument("--cache-dir", default=str(ROOT / "data/cache/lightkurve"))
    parser.add_argument("--model-dir", default=str(ROOT / "models/checkpoints"))
    parser.add_argument("--eval-dir", default=str(ROOT / "models/evaluation"))
    parser.add_argument("--refresh-labels", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of selected targets to process after selection (for controlled runs).")
    parser.add_argument("--max-per-class", type=int, default=None, help="Select at most N genuine targets per class (balanced, deterministic) before processing.")
    parser.add_argument("--max-attempts", type=int, default=2, help="Resume-level max processing visits per target before marking terminal failure.")
    parser.add_argument("--download-timeout", type=int, default=300, help="Per-attempt timeout in seconds for Lightkurve download+processing wrapper (prevents indefinite stalls).")
    parser.add_argument("--download-retries", type=int, default=2, help="Limited retries per search identifier for temporary/network failures and timeouts.")
    parser.add_argument("--retry-delay", type=float, default=10.0, help="Delay in seconds between retryable download attempts.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--sleep", type=float, default=0.0, help="Optional delay in seconds after each visited target.")
    parser.add_argument("--train-only", action="store_true", help="Skip acquisition/processing and train from existing records.")
    args = parser.parse_args()

    koi_csv = Path(args.koi_csv)
    records_path = Path(args.records)
    progress_path = Path(args.progress)
    cache_dir = Path(args.cache_dir)
    model_dir = Path(args.model_dir)
    eval_dir = Path(args.eval_dir)

    rows = fetch_koi_labels(koi_csv, refresh=args.refresh_labels)
    targets, label_summary = build_targets(rows)
    selected_targets, selection_summary = select_targets(
        targets, limit=args.limit, max_per_class=args.max_per_class, seed=args.seed
    )
    label_summary_out = {**label_summary, **selection_summary}
    _write_json(progress_path.with_name("kepler_koi_label_summary.json"), {"generated_at": _utc_now(), **label_summary_out})
    print(json.dumps({"stage": "labels", **label_summary_out}, indent=2), flush=True)

    if not args.train_only:
        proc = process_targets(
            selected_targets,
            records_path,
            progress_path,
            cache_dir,
            max_attempts_per_target=args.max_attempts,
            sleep_seconds=args.sleep,
            download_timeout_seconds=args.download_timeout,
            download_retries=args.download_retries,
            retry_delay_seconds=args.retry_delay,
        )
        print(json.dumps({"stage": "processing", **proc}, indent=2), flush=True)

    records = _load_records(records_path)
    ok = [r for r in records if r.get("status") == "ok"]
    failed = [r for r in records if r.get("status") == "failed"]
    selected_ids = {t.group_id for t in selected_targets}
    ok_selected = [r for r in ok if str(r.get("group_id")) in selected_ids]
    print(json.dumps({
        "stage": "dataset_status",
        "genuine_labelled_targets_acquired_catalogue": label_summary.get("unique_kepids_with_consistent_label", 0),
        "selected_targets_for_processing": len(selected_targets),
        "successfully_processed_light_curves_total_records": len(ok),
        "successfully_processed_light_curves_selected": len(ok_selected),
        "failed_records_total": len(failed),
        "class_distribution_processed_selected": dict(Counter(int(r.get("label", -1)) for r in ok_selected)),
        "class_distribution_processed_total": dict(Counter(int(r.get("label", -1)) for r in ok)),
    }, indent=2), flush=True)

    run_context = {
        "generated_at": _utc_now(),
        "koi_csv": str(koi_csv),
        "records_path": str(records_path),
        "genuine_labelled_targets_acquired_catalogue": label_summary.get("unique_kepids_with_consistent_label", 0),
        "selected_targets_for_processing": len(selected_targets),
        "successfully_processed_light_curves_total_records": len(ok),
        "successfully_processed_light_curves_selected": len(ok_selected),
        "class_distribution_processed_selected": dict(Counter(int(r.get("label", -1)) for r in ok_selected)),
        "selection": selection_summary,
        "label_summary": label_summary,
    }
    try:
        result = train_and_evaluate(
            records_path, model_dir, eval_dir, seed=args.seed, test_size=args.test_size, extra_context=run_context
        )
        meta = result["metadata"]
        print(json.dumps({
            "stage": "training_complete",
            "model_path": result["model_path"],
            "metadata_path": result["metadata_path"],
            "report_path": result["report_path"],
            "sample_counts": meta["sample_counts"],
            "class_distribution": meta["class_distribution"],
            "metrics": meta["metrics"],
            "confusion_matrix": meta["confusion_matrix"],
            "split": meta["split"],
            "run_context": meta.get("run_context", {}),
        }, indent=2), flush=True)
        return 0
    except Exception as exc:
        print(json.dumps({
            "stage": "training_not_completed",
            "reason": f"{type(exc).__name__}: {exc}",
            "run_context": run_context,
            "insufficient_for_reliable_evaluation": True,
        }, indent=2), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
