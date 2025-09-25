# loaders.py
import json
import re
from typing import Any, Dict, List, Tuple, Optional
from datetime import datetime
import pandas as pd


# ---------- Utilities ----------
def parse_ts(ts: Any) -> Optional[datetime]:
    if ts is None:
        return None
    if isinstance(ts, (datetime, pd.Timestamp)):
        try:
            return pd.to_datetime(ts).to_pydatetime()
        except Exception:
            return None
    s = str(ts).strip()
    fmts = [
        "%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%d %H:%M:%S%z", "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d"
    ]
    for f in fmts:
        try:
            return datetime.strptime(s, f)
        except Exception:
            continue
    try:
        dt = pd.to_datetime(s, errors="coerce")
        return None if pd.isna(dt) else dt.to_pydatetime()
    except Exception:
        return None


def extract_user_id_and_audience(query: str) -> Tuple[Optional[int], str]:
    m = re.search(r'user[_\s]?id\s*[:=]?\s*(\d+)', query, flags=re.IGNORECASE)
    user_id = int(m.group(1)) if m else None
    audience = "parent"
    if re.search(r'\bteacher\b', query, flags=re.IGNORECASE):
        audience = "teacher"
    return user_id, audience


# ---------- JSON loading ----------
def load_json_raw(path_or_bytes) -> Any:
    if hasattr(path_or_bytes, "read"):
        return json.load(path_or_bytes)
    if isinstance(path_or_bytes, (bytes, bytearray)):
        return json.loads(path_or_bytes.decode("utf-8"))
    if isinstance(path_or_bytes, str):
        try:
            with open(path_or_bytes, encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return json.loads(path_or_bytes)
    return json.loads(path_or_bytes)


def _ensure_lists(d: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "user", "users",
        "enrollment", "daily_activity_log", "topic_session",
        "chapter_session", "activity_performance", "lesson_session",
        "topics", "Topics", "messages"
    ]
    out = dict(d)
    for k in keys:
        if k in out and not isinstance(out[k], list):
            out[k] = [out[k]]
    return out


def normalize_union(raw: Any) -> Dict[str, Any]:
    canon_keys = [
        "user", "enrollment", "daily_activity_log", "topic_session",
        "chapter_session", "activity_performance", "lesson_session",
        "topics", "messages"
    ]

    buckets: Dict[str, List[Any]] = {k: [] for k in canon_keys}

    def _ingest_dict(d: Dict[str, Any]):
        d = _ensure_lists(d)
        if "users" in d:
            buckets["user"].extend(d.get("users", []))
        if "user" in d:
            buckets["user"].extend(d.get("user", []))
        if "Topics" in d:
            buckets["topics"].extend(d.get("Topics", []))
        if "topics" in d:
            buckets["topics"].extend(d.get("topics", []))
        for k in ["enrollment", "daily_activity_log", "topic_session",
                  "chapter_session", "activity_performance", "lesson_session",
                  "messages"]:
            if k in d:
                buckets[k].extend(d.get(k, []))
        for v in d.values():
            if isinstance(v, dict):
                _ingest_dict(v)

    if isinstance(raw, dict):
        if "students" in raw and isinstance(raw["students"], list):
            for s in raw["students"]:
                if isinstance(s, dict):
                    _ingest_dict(s)
        else:
            _ingest_dict(raw)
    elif isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                _ingest_dict(item)

    for k in canon_keys:
        buckets.setdefault(k, [])
    return buckets


def load_json(path_or_bytes) -> Dict[str, Any]:
    raw = load_json_raw(path_or_bytes)
    return normalize_union(raw)


# ---------- NEW: merge multiple normalized datasets ----------
def merge_data(*datasets: Dict[str, Any]) -> Dict[str, Any]:
    if not datasets:
        return {}
    merged: Dict[str, List[Any]] = {}
    # union of all keys
    keys = set().union(*[set(ds.keys()) for ds in datasets])
    for k in keys:
        merged[k] = []
        for ds in datasets:
            merged[k].extend(ds.get(k, []))
    return merged
