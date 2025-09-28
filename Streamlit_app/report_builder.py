# report_builder.py
from __future__ import annotations
from typing import Dict, Any, List


def _arrow(delta: int) -> str:
    return "↑" if delta >= 0 else "↓"


def _fmt_pct(x) -> str:
    try:
        return f"{float(x):.1f}%"
    except Exception:
        return "—"


def _fmt_float(x, digits=1) -> str:
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return "—"


def _fmt_int(x) -> str:
    try:
        return str(int(x))
    except Exception:
        return "—"


def _lines_for_mastery(acc_block: Dict[str, Any]) -> List[str]:
    if not acc_block or not acc_block.get("subjects_meta"):
        return ["No subject accuracy data in the selected period."]
    out = []
    rows = sorted(acc_block["subjects_meta"], key=lambda r: r.get("avg", 0.0), reverse=True)
    for r in rows:
        subj = r.get("subject", "—")
        avg = r.get("avg", 0.0)
        band = r.get("band", "—")
        attempts = r.get("attempts", 0)
        out.append(f"- {subj}: {avg:.1f}% ({band}, {attempts} attempt{'s' if attempts!=1 else ''})")
    return out


def _lines_for_speed(speed_block: Dict[str, Any]) -> List[str]:
    if not speed_block:
        return ["No processing speed data in the selected period."]
    unit = speed_block.get("unit", "mins")
    unit_label = f" {unit}".rstrip()

    lines = []
    attempts = speed_block.get("attempts", 0)
    mean_v = speed_block.get("mean", None)
    med_v = speed_block.get("median", None)
    p90_v = speed_block.get("p90", None)

    if attempts and (mean_v is not None):
        lines.append(
            f"Overall mean: {_fmt_float(mean_v)}{unit_label} • median: {_fmt_float(med_v)}{unit_label} • p90: {_fmt_float(p90_v)}{unit_label} (n={attempts})"
        )

    rows = speed_block.get("per_subject_meta") or []
    if not rows:
        per_dict = speed_block.get("per_subject") or {}
        if per_dict:
            for subj, avg_time in sorted(per_dict.items(), key=lambda kv: kv[1], reverse=True):
                lines.append(f"- {subj}: {_fmt_float(avg_time)}{unit_label}")
        else:
            lines.append("No per-subject processing data.")
        return lines

    rows = sorted(rows, key=lambda r: r.get("avg_time", 0.0), reverse=True)
    for r in rows:
        subj = r.get("subject", "—")
        avg_t = r.get("avg_time", 0.0)
        tot_t = r.get("total_time", 0.0)
        att = r.get("attempts", 0)
        lines.append(
            f"- {subj}: {avg_t:.1f}{unit_label} (total {tot_t:.1f}{unit_label}, {att} attempt{'s' if att!=1 else ''})"
        )
    return lines


def build_report(d: Dict[str, Any]) -> str:
    student = d.get("student", {})
    period = d.get("period", {})
    usage = d.get("usage", {})
    focus = d.get("focus", {})
    learning = d.get("learning", {}) or {}

    rep = []
    rep.append("Student Learning Report (SEN)\n")
    rep.append(f"Student: {student.get('name','')}")
    rep.append(f"Student ID: {student.get('id','')}")
    rep.append(f"Class / Year: {student.get('class','—')} / {student.get('year','—')}")
    rep.append(f"Reporting Period: {period.get('start','—')} – {period.get('end','—')}")
    rep.append(f"Prepared for: {d.get('prepared_for','—')}")
    rep.append(f"Generated on: {period.get('generated_on','—')}")
    rep.append("Data Sources: activity_performance, chapter_session, topic_session, lesson_session, daily_activity_log, topics, enrollment\n")

    focus_delta = int(focus.get("focus_score_delta", 0) or 0)
    comp_pct = float(usage.get("completion_pct", 0) or 0)
    focus_score = int(focus.get("focus_score", 0) or 0)
    class_median = int(focus.get("class_median", 62) or 62)

    if focus_delta >= 5 and comp_pct >= 70:
        exec_summary = "Engagement and comprehension are improving; routine and supports appear to help."
    elif comp_pct < 40 or focus_score < class_median:
        exec_summary = "Engagement, focus, and skill growth have declined compared to last period. Immediate teacher support is advised."
    else:
        exec_summary = "Overall progress is steady with moderate gains; continue current routine and supports."

    rep.append("1) Executive Summary")
    rep.append(f"Summary: {exec_summary}")
    rep.append(f"Focus score: {focus_score} ({_arrow(focus_delta)} {abs(focus_delta)} from last period)")
    rep.append(f"Completion rate: {usage.get('lessons_done',0)}/{usage.get('lessons_total',0)} ({_fmt_pct(usage.get('completion_pct',0))})")
    rep.append(f"Time-on-task: {_fmt_float(usage.get('total_time_mins',0))} mins total ({_fmt_int(usage.get('trend_vs_prev_pct',0))}% vs last period)\n")

    rep.append("2) SEN Profile & Accommodations")
    rep.append("Summary: Derived metrics only (no pre-set accommodations in source).")
    rep.append("Primary Needs: —")
    rep.append("Accommodations: —")
    rep.append("Effectiveness: TTS ON → 0% vs OFF → 0% (0pp)")
    rep.append("Stability: Font size changed 0× this period\n")

    eng_summary = (
        "Strong participation and lesson completion." if comp_pct >= 70 else
        "Very low engagement, with limited active days and short sessions." if comp_pct < 40 else
        "Moderate engagement; room for higher completion."
    )
    rep.append("3) Engagement & Usage")
    rep.append(f"Summary: {eng_summary}")
    rep.append(f"Active Days: {_fmt_int(usage.get('active_days','—'))}")
    rep.append(f"Sessions: {_fmt_int(usage.get('sessions','—'))} (avg. {_fmt_float(usage.get('avg_session_mins','—'))} mins)")
    rep.append(f"Completion: {_fmt_int(usage.get('lessons_done',0))} of {_fmt_int(usage.get('lessons_total',0))} lessons ({_fmt_pct(usage.get('completion_pct',0))})")
    rep.append(f"Trend: {_fmt_int(usage.get('trend_vs_prev_pct',0))}% vs last period\n")

    rep.append("4) Focus & Concentration")
    rep.append(f"Summary: {'Improved attention relative to class median.' if focus_score >= class_median else 'Below class median; consider shorter, more frequent sessions.'}")
    rep.append(f"Focus score: {focus_score} (class median: {class_median})")
    rep.append(f"Avg. attention block: {_fmt_float(focus.get('avg_sustained_block_mins','—'))} mins\n")

    # ---- 5) Accuracy & Mastery ----
    rep.append("5) Learning Progress & Mastery")
    rep.append("Summary: Subject-level growth based on activity performance.")
    for s in learning.get("skills", []):
        try:
            rep.append(f"- {s['name']}: {float(s['value'])*100:.0f}% ({float(s.get('delta',0))*100:+.0f}pp)")
        except Exception:
            pass
    acc_block = d.get("accuracy_mastery", {})
    if acc_block:
        rep.extend(_lines_for_mastery(acc_block))
    pi = learning.get("perseverance_index", None)
    if pi is not None:
        try:
            rep.append(f"Perseverance index: {float(pi):.2f} (fraction of attempts using hints)\n")
        except Exception:
            rep.append(f"Perseverance index: {pi}\n")
    else:
        rep.append("Perseverance index: —\n")

    # ---- 6) Processing Speed ----
    rep.append("6) Processing Speed")
    rep.append("Summary: Average response/processing time per subject based on total time spent on attempts.")
    speed_block = d.get("processing_speed", {})
    rep.extend(_lines_for_speed(speed_block))
    rep.append("")

    rep.append("7) Reading, Language & Expression")
    rep.append("Summary: Not available in this dataset.")
    rep.append("Readability: —")
    rep.append("TTR: —\n")

    rep.append("8) AI Interaction Quality & Support Usage")
    rep.append("Summary: Derived hints usage (no built-in AI support fields in source).")
    rep.append(f"Hints used per attempt: {learning.get('perseverance_index','—')}\n")

    rep.append("9) Motivation & Routine")
    dropoff_risk = d.get("routine", {}).get("dropoff_risk", "—")
    rep.append(f"Summary: {'Low drop-off risk.' if dropoff_risk=='low' else ('Potential drop-off risk.' if dropoff_risk in ('medium','high') else '—')}\n")

    rep.append("10) Technology & Accessibility Diagnostics")
    rep.append("Summary: Device info is partial in this dataset.\n")

    rep.append("11) Goals & Recommendations")
    recs = d.get("recommendations", [])
    if recs:
        rep.append("Recommendations:")
        for r in recs:
            rep.append(f"- {r}")
    else:
        rep.append("Recommendations:")
        rep.append("- Encourage regular short practice sessions (5–7 mins) on weaker subjects")
        rep.append("- Review missed questions in recent attempts")
        rep.append("- Use shorter sessions if average session length is below 10 mins")
    rep.append("")

    rep.append("12) Unanswered & Out-of-Scope Questions")
    rep.append("Summary: Not tracked in this dataset.")
    rep.append("Total questions: —")
    rep.append("Unanswered: — | Out-of-scope: —")
    return "\n".join(rep).strip()
