# utils_time.py
import pandas as pd
from datetime import datetime, date
from typing import Any, Dict, List, Optional

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

def pick_first_ts(record: Dict[str, Any], keys: List[str]) -> Optional[datetime]:
    for k in keys:
        if k in record and record[k] not in (None, "", "Unknown"):
            dt = parse_ts(record[k])
            if dt:
                return dt
    return None

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def pct(n: float, d: float) -> float:
    return round(100 * n / d, 1) if d else 0.0

def get_time_spent(r: Dict[str, Any]) -> float:
    v = r.get("total_time_spent", None)
    if v is None:
        v = r.get("time_spent", 0)
    try:
        return float(v or 0.0)
    except Exception:
        return 0.0

def compute_age_from_dob(dob_str: str) -> str:
    if not dob_str:
        return "—"
    dob = None
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f"):
        try:
            dob = datetime.strptime(dob_str, fmt).date()
            break
        except Exception:
            pass
    if dob is None:
        try:
            dob = datetime.fromisoformat(dob_str).date()
        except Exception:
            return "—"
    today = date.today()
    years = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
    return str(years)
