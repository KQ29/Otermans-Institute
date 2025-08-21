import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import json
import pandas as pd
import re
import sqlite3
from calendar import monthrange
from dateutil import parser as dateutil_parser

# ─── Helper: serialize nested structures so SQLite accepts them ─────────────
def sanitize_df_for_sql(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if df[col].apply(lambda v: isinstance(v, (list, dict))).any():
            df[col] = df[col].apply(lambda v: json.dumps(v) if isinstance(v, (list, dict)) else v)
    return df

# ─── 1) Load & cache the dataset ───────────────────────────────────────────
@st.cache_data
def load_data():
    with open("complete_dummy_data.json", "r") as f:
        data = json.load(f)

    users_df = pd.DataFrame(data.get("user", []))
    if not users_df.empty:
        users_df["user_id"] = users_df["user_id"].astype(str)

    enrollment_df = pd.DataFrame(data.get("enrollment", []))
    if not enrollment_df.empty:
        enrollment_df["user_id"] = enrollment_df["user_id"].astype(str)
        enrollment_df["topic_id"] = enrollment_df["topic_id"].astype(str)
        if "enrolled_at" in enrollment_df.columns:
            enrollment_df["enrolled_at"] = pd.to_datetime(
                enrollment_df["enrolled_at"], errors="coerce", utc=True
            ).dt.tz_convert(None)

    daily_df = pd.DataFrame(data.get("daily_activity_log", []))
    if not daily_df.empty:
        daily_df["user_id"] = daily_df["user_id"].astype(str)
        if "login_timestamp" in daily_df.columns:
            daily_df["login_timestamp"] = pd.to_datetime(
                daily_df["login_timestamp"], errors="coerce", utc=True
            ).dt.tz_convert(None)

    return users_df, enrollment_df, daily_df

# ─── 2) Load & cache the model ───────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_id = "MBZUAI/LaMini-Flan-T5-248M"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, use_safetensors=True)
    return tokenizer, model

# ─── 3) Date heuristics & extraction ───────────────────────────────────────
def looks_like_date_token(tok: str) -> bool:
    if not tok:
        return False
    if re.match(r"^20\d{2}([\/\-](0[1-9]|1[0-2])([\/\-](0[1-9]|[12]\d|3[01]))?)?$", tok):
        return True
    if re.match(r"^\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}$", tok):
        return True
    if re.search(r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\b", tok, re.IGNORECASE):
        return True
    return False

def choose_date_column(question: str) -> str:
    if re.search(r"\b(enroll|enrolled|enrollments?|enrolled_at)\b", question, re.IGNORECASE):
        return "e.enrolled_at"
    return "d.login_timestamp"

def normalize_date(dt: pd.Timestamp) -> pd.Timestamp:
    return dt.normalize()

def extract_date_range(question: str):
    date_col = choose_date_column(question)
    start = None
    end = None

    # 1) "between X and Y" if either looks like a date
    m_between = re.search(r"\bbetween\s+([^\s,]+)\s+(?:and|-)\s+([^\s,]+)", question, re.IGNORECASE)
    if m_between:
        a, b = m_between.group(1), m_between.group(2)
        if looks_like_date_token(a) or looks_like_date_token(b):
            try:
                d1 = dateutil_parser.parse(a, fuzzy=True)
                d2 = dateutil_parser.parse(b, fuzzy=True)
                if d1 <= d2:
                    start = normalize_date(pd.Timestamp(d1))
                    end = normalize_date(pd.Timestamp(d2)) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
                else:
                    start = normalize_date(pd.Timestamp(d2))
                    end = normalize_date(pd.Timestamp(d1)) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
            except Exception:
                pass

    # 2) explicit "from X to Y" (only if one side resembles a date)
    if start is None:
        m_from = re.search(r"from\s+([^\s,]+)\s+to\s+([^\s,]+)", question, re.IGNORECASE)
        if m_from:
            t1, t2 = m_from.group(1), m_from.group(2)
            if looks_like_date_token(t1) or looks_like_date_token(t2):
                try:
                    d1 = dateutil_parser.parse(t1, fuzzy=True)
                    d2 = dateutil_parser.parse(t2, fuzzy=True)
                    if d1 <= d2:
                        start = normalize_date(pd.Timestamp(d1))
                        end = normalize_date(pd.Timestamp(d2)) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
                    else:
                        start = normalize_date(pd.Timestamp(d2))
                        end = normalize_date(pd.Timestamp(d1)) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
                except Exception:
                    pass

    # 3) "after"/"since"
    if start is None:
        m_after = re.search(r"\b(?:after|since)\s+([^\s,]+)", question, re.IGNORECASE)
        if m_after and looks_like_date_token(m_after.group(1)):
            try:
                d = dateutil_parser.parse(m_after.group(1), fuzzy=True)
                start = normalize_date(pd.Timestamp(d))
            except Exception:
                pass

    # 4) "before"
    if end is None:
        m_before = re.search(r"\b(?:before)\s+([^\s,]+)", question, re.IGNORECASE)
        if m_before and looks_like_date_token(m_before.group(1)):
            try:
                d = dateutil_parser.parse(m_before.group(1), fuzzy=True)
                end = normalize_date(pd.Timestamp(d)) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
            except Exception:
                pass

    # 5) month name + year
    if start is None and end is None:
        m_month = re.search(r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+([0-9]{4})\b", question, re.IGNORECASE)
        if m_month:
            month_str = m_month.group(1)
            year = int(m_month.group(2))
            try:
                dt_start = pd.Timestamp(f"{month_str} 1 {year}")
                last_day = monthrange(year, dt_start.month)[1]
                start = normalize_date(dt_start)
                end = pd.Timestamp(year=year, month=dt_start.month, day=last_day) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
            except Exception:
                pass

    # 6) year-month like "2024-07"
    if start is None and end is None:
        ym = re.search(r"\b(20\d{2})[\/\-](0[1-9]|1[0-2])\b", question)
        if ym:
            year = int(ym.group(1))
            month = int(ym.group(2))
            start = pd.Timestamp(year=year, month=month, day=1)
            last_day = monthrange(year, month)[1]
            end = pd.Timestamp(year=year, month=month, day=last_day) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)

    # 7) year only
    if start is None and end is None:
        y = re.search(r"\b(20\d{2})\b", question)
        if y:
            year = int(y.group(1))
            start = pd.Timestamp(year=year, month=1, day=1)
            end = pd.Timestamp(year=year, month=12, day=31) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)

    # 8) exact date fallback
    if start is None and end is None:
        exact = re.search(r"\b(20\d{2}[\/\-](0[1-9]|1[0-2])[\/\-](0[1-9]|[12]\d|3[01]))\b", question)
        if exact:
            try:
                d = dateutil_parser.parse(exact.group(1), fuzzy=True)
                start = normalize_date(pd.Timestamp(d))
                end = normalize_date(pd.Timestamp(d)) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
            except Exception:
                pass

    # sort direction hints
    sort_dir = "DESC"
    if re.search(r"\b(oldest|ascending|earliest first)\b", question, re.IGNORECASE):
        sort_dir = "ASC"
    elif re.search(r"\b(newest|descending|latest first)\b", question, re.IGNORECASE):
        sort_dir = "DESC"

    return date_col, start, end, sort_dir

# ─── 4) UI setup ───────────────────────────────────────────────────────────
st.set_page_config(layout="wide")
st.title("🧠 AI-Driven SQL Explorer (with fixed date-range inference)")

st.markdown("**Example questions:**")
st.markdown("- `Show me enrollments on 2025-05-23`")
st.markdown("- `Users with time_spent from 10 to 35`")
st.markdown("- `Find users with activities_completed between 1 and 30`")
st.markdown("- `List KS1 students with Advanced reading level`")
st.markdown("- `Show users enrolled between 2024-02-01 and 2025-06-01`")
st.markdown("- `Show latest enrollments`")
st.markdown("- `Show oldest activity logs`")

question = st.text_input(
    "Enter your question (e.g. filter by class_level, is_completed, points_earned, activities_completed, time_spent, date/enrollment range):"
)
run = st.button("Run")

if run and question:
    users_df, enrollment_df, daily_df = load_data()
    tokenizer, model = load_model()

    # ─── LLM prompt (minimal to reduce hallucination) ─────────────
    prompt = f"""
Extract filtering instructions from this question. Respond with JSON only.
Example: {{"filters": {{"class_level":"KS1", "reading_level":"Advanced", "topic_id":"3", "is_completed": true}}}}
Question: {question}
"""
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=150, do_sample=False)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    st.subheader("🔍 Raw Model Output")
    st.code(decoded)

    # ─── Parse + deterministic fallbacks ────────────────────────────
    instr = {}
    m = re.search(r"\{[\s\S]*\}", decoded)
    if m:
        try:
            instr = json.loads(m.group())
        except:
            instr = {}
    instr.setdefault("filters", {})

    def mentioned(field: str, q: str) -> bool:
        alt = field.replace("_", "[ _]?")
        return bool(re.search(rf"\b{alt}\b", q, re.IGNORECASE))

    for numeric in ("time_spent", "points_earned", "activities_completed", "chapter_id", "lesson_id", "topic_id"):
        if numeric in instr["filters"] and not mentioned(numeric, question):
            instr["filters"].pop(numeric, None)

    if "class_level" not in instr["filters"]:
        cl = re.search(r"\b(KS1|KS2)\b", question, re.IGNORECASE)
        if cl:
            instr["filters"]["class_level"] = cl.group(1).upper()
    if "reading_level" not in instr["filters"]:
        rl = re.search(r"\b(Beginner|Intermediate|Advanced)\b", question, re.IGNORECASE)
        if rl:
            instr["filters"]["reading_level"] = rl.group(1).capitalize()

    if "is_completed" not in instr["filters"] and re.search(r"\bcompleted\b", question, re.IGNORECASE):
        instr["filters"]["is_completed"] = True
    if "have_certificate" not in instr["filters"] and re.search(r"\bcertificate\b", question, re.IGNORECASE):
        instr["filters"]["have_certificate"] = True

    # numeric range fallback
    for field in ("activities_completed", "time_spent", "points_earned", "chapter_id", "lesson_id"):
        if field not in instr["filters"]:
            base = field.replace("_", "[ _]?")
            patterns = [
                rf"\b{base}\b.*?\b(?:range\s*(?:like\s*)?from)\s*(\d+)\s*(?:to|-)\s*(\d+)\b",
                rf"\b{base}\b\s*(?:from)\s*(\d+)\s*(?:to|-)\s*(\d+)\b",
                rf"\b{base}\b\s*(\d+)\s*[-]\s*(\d+)\b",
                rf"\b{base}\b\s*between\s*(\d+)\s*(?:and|-)\s*(\d+)\b",
                rf"\b{base}\s*(\d+)\s+to\s+(\d+)\b"
            ]
            for pat in patterns:
                m_range = re.search(pat, question, re.IGNORECASE)
                if m_range:
                    low = int(m_range.group(1))
                    high = int(m_range.group(2))
                    if low > high:
                        low, high = high, low
                    instr["filters"][field] = ("range", low, high)
                    break

    # explicit operator parsing
    if "points_earned" not in instr["filters"]:
        pe = re.search(r"\bpoints[_\s]?earned\s*(>=|<=|>|<|=)\s*(\d+)\b", question, re.IGNORECASE)
        if pe:
            instr["filters"]["points_earned"] = (pe.group(1), int(pe.group(2)))
    if "activities_completed" not in instr["filters"]:
        ac = re.search(r"\bactivities[_\s]?completed\s*(>=|<=|>|<|=)\s*(\d+)\b", question, re.IGNORECASE)
        if ac:
            instr["filters"]["activities_completed"] = (ac.group(1), int(ac.group(2)))
    if "time_spent" not in instr["filters"]:
        ts = re.search(r"\btime[_\s]?spent\s*(>=|<=|>|<|=)\s*(\d+)\b", question, re.IGNORECASE)
        if ts:
            instr["filters"]["time_spent"] = (ts.group(1), int(ts.group(2)))
        else:
            bare = re.search(r"\btime[_\s]?spent\s+(\d+)\b", question, re.IGNORECASE)
            if bare:
                instr["filters"]["time_spent"] = (">=", int(bare.group(1)))

    if "topic_id" not in instr["filters"]:
        tid = re.search(r"\btopic[_\s]?id\s*[:=]?\s*(\d+)\b", question, re.IGNORECASE)
        if tid:
            instr["filters"]["topic_id"] = tid.group(1)

    # ─── Date range inference (added into intent for visibility) ───────
    date_col, start_dt, end_dt, sort_dir = extract_date_range(question)
    instr["date_range"] = {
        "column": date_col,
        "start": start_dt.isoformat() if start_dt is not None else None,
        "end": end_dt.isoformat() if end_dt is not None else None,
        "sort_direction": sort_dir
    }

    st.subheader("✅ Parsed Filters & Date Range")
    st.json(instr)

    # ─── 6) Build SQLite & register sanitized tables ─────────────────────
    conn = sqlite3.connect(":memory:")
    users_sql = sanitize_df_for_sql(users_df.copy())
    enrollment_sql = sanitize_df_for_sql(enrollment_df.copy())
    daily_sql = sanitize_df_for_sql(daily_df.copy())

    users_sql.to_sql("users", conn, index=False)
    enrollment_sql.to_sql("enrollment", conn, index=False)
    daily_sql.to_sql("daily_activity_log", conn, index=False)

    # ─── 7) Build WHERE clause ─────────────────────────────────────────
    wh = []
    # date range
    if start_dt is not None:
        wh.append(f"{date_col} >= '{start_dt.isoformat()}'")
    if end_dt is not None:
        wh.append(f"{date_col} <= '{end_dt.isoformat()}'")

    for col, val in instr.get("filters", {}).items():
        if isinstance(val, tuple):
            if val[0] == "range" and len(val) == 3:
                _, low, high = val
                if col in ("chapter_id", "lesson_id", "topic_id"):
                    wh.append(f"e.{col} BETWEEN {low} AND {high}")
                elif col in ("points_earned", "activities_completed", "time_spent"):
                    wh.append(f"d.{col} BETWEEN {low} AND {high}")
                else:
                    wh.append(f"{col} BETWEEN {low} AND {high}")
            elif col in ("points_earned", "activities_completed", "time_spent") and len(val) == 2:
                op, num = val
                wh.append(f"d.{col} {op} {num}")
            elif col in ("chapter_id", "lesson_id") and len(val) == 2:
                op, num = val
                wh.append(f"e.{col} {op} {num}")
            else:
                wh.append(f"{col} = '{val}'")
        elif isinstance(val, bool):
            if col in ("is_completed", "have_certificate"):
                wh.append(f"e.{col} = {1 if val else 0}")
            else:
                wh.append(f"{col} = {1 if val else 0}")
        elif isinstance(val, (int, float)):
            if col in ("chapter_id", "lesson_id", "topic_id"):
                wh.append(f"e.{col} = {val}")
            else:
                wh.append(f"{col} = {val}")
        else:
            if col in ("class_level", "reading_level", "school_name"):
                wh.append(f"u.{col} LIKE '%{val}%'")
            elif col in ("topic_id", "chapter_id", "lesson_id"):
                wh.append(f"e.{col} LIKE '%{val}%'")
            else:
                wh.append(f"{col} LIKE '%{val}%'")

    where_sql = " AND ".join(wh) if wh else "1=1"

    # ─── 8) Construct SQL ───────────────────────────────────────────────
    order_by = f"{date_col} {sort_dir}"
    sql = f"""
SELECT
  u.*,
  e.enrollment_id,
  e.topic_id, e.chapter_id, e.lesson_id,
  e.is_completed, e.have_certificate,
  e.enrolled_at,
  d.login_timestamp, d.time_spent,
  d.points_earned, d.activities_completed
FROM users AS u
LEFT JOIN enrollment        AS e ON u.user_id = e.user_id
LEFT JOIN daily_activity_log AS d ON u.user_id = d.user_id
WHERE {where_sql}
ORDER BY {order_by}
LIMIT 100;
"""
    st.subheader("💾 Generated SQL")
    st.code(sql)

    # ─── 9) Execute & display ───────────────────────────────────────────────
    try:
        df = pd.read_sql(sql, conn, parse_dates=["login_timestamp", "enrolled_at"])
        st.subheader("📋 Results")
        st.dataframe(df, hide_index=True)
    except Exception as e:
        st.error(f"SQL error: {e}")
