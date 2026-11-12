"""Invariant checker and summary printer for offline/live /classify run JSON files (Step 4 & Step 5)."""
import glob
import json
import os
import re
import sys
from collections import Counter

APP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if APP_ROOT not in sys.path:
    sys.path.insert(0, APP_ROOT)

from app.reasons import REASON_REGISTRY
from app.conditions import is_related_compound_collision
from app.text_utils import clean_markup_and_unescape, canonicalize


def load_all_snapshots(data_dir: str) -> dict:
    snap_map = {}
    snap_glob = os.path.join(data_dir, "snapshots", "*.json")
    for sp in glob.glob(snap_glob):
        if sp.endswith("manifest.json") or sp.endswith("pubchem_cache.json"):
            continue
        try:
            with open(sp, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        pid = str(item.get("pmid") or item.get("id") or item.get("paper_id") or "")
                        if pid:
                            snap_map.setdefault(pid, []).append(item)
                elif isinstance(data, dict):
                    if "papers" in data and isinstance(data["papers"], list):
                        for item in data["papers"]:
                            pid = str(item.get("pmid") or item.get("id") or item.get("paper_id") or "")
                            if pid:
                                snap_map.setdefault(pid, []).append(item)
                    for k, item in data.items():
                        if isinstance(item, dict):
                            pid = str(item.get("pmid") or item.get("id") or k)
                            snap_map.setdefault(pid, []).append(item)
        except Exception:
            pass
    return snap_map


def _find_latest_run(eval_runs_dir: str) -> str:
    candidates = [
        p for p in glob.glob(os.path.join(eval_runs_dir, "run_*.json"))
        if not p.endswith("_meta.json")
    ]
    if not candidates:
        raise FileNotFoundError(f"No run_*.json found in {eval_runs_dir}")
    return max(candidates, key=os.path.getmtime)


def _looks_cut_off_mid_word(ev: dict) -> list[str]:
    flags = []
    txt = (ev.get("display_text") or ev.get("text") or "").strip()
    if not txt:
        return ["empty_evidence_text"]
    # Lowercase fragment at very start (e.g. "o overcome") OR mid-clause leading conjunction fragment without uppercase start
    if re.match(r"^[a-z]{1,2}\s+", txt):
        flags.append(f"clipped_short_lowercase_prefix:{txt[:20]!r}")
    elif txt[0].islower():
        flags.append(f"starts_lowercase:{txt[:25]!r}")
    # Mid-word cut at end (ends mid-word or on orphan short alpha fragment lacking sentence-ending punctuation; allow clause splitters only? check strictly)
    if re.search(r"[A-Za-z]-$", txt):
        flags.append(f"trailing_hyphen_cut:{txt[-20:]!r}")
    elif not re.search(r"[.!?\"'\)\]\d%]$", txt) and re.search(r"\b[A-Za-z]{1,3}$", txt):
        flags.append(f"possible_clipped_ending:{txt[-25:]!r}")
    return flags


def check_run(run_path: str) -> dict:
    with open(run_path, "r", encoding="utf-8") as f:
        run_data = json.load(f)

    snapshots_by_pid = load_all_snapshots(os.path.join(APP_ROOT, "data"))

    meta_path = run_path[:-5] + "_meta.json"
    health_cal_ver = None
    meta_data = None
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta_data = json.load(f)
        health_cal_ver = (meta_data.get("health") or {}).get("calibration_version")

    all_violations = Counter()
    violation_details = []

    print("=" * 100)
    print(f"RUN FILE: {run_path}")
    if meta_data:
        print(
            f"META: pipeline={meta_data.get('pipeline')} | model_revision={meta_data.get('model_revision')} "
            f"| health_cal={health_cal_ver} | git_head={meta_data.get('git_commit_hash')}"
        )
    print("=" * 100)

    for entry in run_data:
        qname = entry.get("query_name")
        payload = entry.get("payload", {})
        timing_s = entry.get("timing_s")
        resp = entry.get("response", {})

        print(f"\n### {qname} (timing={timing_s}s)")
        if "error" in resp:
            print(f"  ERROR RESPONSE: status={resp['error']} detail={resp.get('detail')}")
            all_violations["http_or_request_errors"] += 1
            continue

        entity = payload.get("entity", "")
        aliases = payload.get("aliases") or []
        top_cal_ver = resp.get("calibration_version")
        cspec = resp.get("criterion_spec") or {}

        # (g) Criterion/role sanity
        print(
            "  [g] criterion_spec: "
            f"entity_role={cspec.get('entity_role')} | scope={cspec.get('scope')} "
            f"| activity={cspec.get('activity')!r} | system={cspec.get('system')} "
            f"| unparsed_clauses={cspec.get('unparsed_clauses')} "
            f"| constraints={cspec.get('constraints')}"
        )

        if health_cal_ver and top_cal_ver != health_cal_ver:
            all_violations["d_cal_version_mismatch_top_vs_health"] += 1
            violation_details.append(f"{qname}: top calibration_version={top_cal_ver!r} != /health={health_cal_ver!r}")

        results = resp.get("results", [])
        q_violations = Counter()
        counts = Counter()

        # Table header (Step 5)
        print("\n  rank | paper_id              | classification        | reason_code               | conf   | ev_status    | title")
        print("  " + "-" * 135)

        qualifies_rows = []
        reason_registry_summary = []

        for r in results:
            rank = r.get("rank")
            pid = str(r.get("paper_id"))
            cls = r.get("classification")
            rcode = r.get("reason_code")
            conf = r.get("confidence")
            ev_status = r.get("evidence_status")
            ev = r.get("evidence")
            title = (r.get("title") or "").replace("\n", " ").strip()
            title_trunc = (title[:77] + "...") if len(title) > 80 else title
            dedup_of = r.get("dedup_of")

            counts[cls] += 1

            conf_val = None
            conf_str = "null"
            if isinstance(conf, dict):
                conf_val = conf.get("value") if conf.get("value") is not None else conf.get("score")
                conf_str = f"{conf_val:.4f}" if isinstance(conf_val, (int, float)) else "dict"
            elif isinstance(conf, (int, float)):
                conf_val = float(conf)
                conf_str = f"{conf_val:.4f}"

            print(
                f"  {str(rank):>4} | {pid:<21} | {str(cls):<21} | {str(rcode):<25} | "
                f"{conf_str:>6} | {str(ev_status):<12} | {title_trunc}"
            )

            # (a) Slice invariant & cut-off check
            if ev is not None:
                d_txt = ev.get("display_text") if ev.get("display_text") is not None else ev.get("text", "")
                r_txt = ev.get("raw_text") or ""
                cs = ev.get("char_start")
                ce = ev.get("char_end")
                rs = ev.get("raw_start")
                re_ = ev.get("raw_end")
                if cs is None or ce is None or cs < 0 or ce <= cs or (ce - cs) != len(d_txt):
                    q_violations["a_slice_offset_len_mismatch"] += 1
                    violation_details.append(f"{qname} [{pid}]: char_end-char_start ({ce}-{cs}) != len(display_text) ({len(d_txt)})")
                if r_txt and rs is not None and re_ is not None and (re_ - rs) != len(r_txt):
                    q_violations["a_raw_offset_len_mismatch"] += 1
                    violation_details.append(f"{qname} [{pid}]: raw_end-raw_start ({re_}-{rs}) != len(raw_text) ({len(r_txt)})")
                if r_txt and clean_markup_and_unescape(r_txt) != d_txt:
                    q_violations["a_raw_vs_display_mismatch"] += 1
                    violation_details.append(f"{qname} [{pid}]: clean(raw_text) != display_text")
                cut_flags = _looks_cut_off_mid_word(ev)
                if cut_flags:
                    q_violations["a_mid_word_or_lowercase_cutoff"] += 1
                    violation_details.append(f"{qname} [{pid}]: cutoff flags={cut_flags}")

            # (b) Title-as-evidence check
            if ev is not None and title:
                d_txt = (ev.get("display_text") or ev.get("text") or "").strip()
                t_norm = re.sub(r"\s+", " ", title).strip().lower()
                d_norm = re.sub(r"\s+", " ", d_txt).strip().lower()
                cs = ev.get("char_start")
                if d_norm == t_norm or (len(d_norm) >= 12 and d_norm in t_norm) or (cs == 0 and t_norm.startswith(d_norm[:30]) and len(d_norm) > 10):
                    q_violations["b_title_as_evidence"] += 1
                    violation_details.append(f"{qname} [{pid}]: evidence matches/contained in title: {d_txt!r}")

            # (c) evidence_status consistency
            if ev_status == "insufficient" and ev is not None:
                q_violations["c_insufficient_with_non_null_evidence"] += 1
                violation_details.append(f"{qname} [{pid}]: evidence_status='insufficient' has non-null evidence")
            if ev_status == "supported":
                if ev is None or cls != "qualifies":
                    q_violations["c_supported_inconsistent"] += 1
                    violation_details.append(f"{qname} [{pid}]: evidence_status='supported' with cls={cls}, ev={ev is not None}")

            # (d) Confidence rules
            if cls == "insufficient_evidence" and conf is not None:
                q_violations["d_confidence_non_null_on_insufficient"] += 1
                violation_details.append(f"{qname} [{pid}]: confidence non-null on insufficient_evidence")
            if rcode and rcode in REASON_REGISTRY and REASON_REGISTRY[rcode].kind == "veto" and conf is not None:
                q_violations["d_confidence_non_null_on_veto"] += 1
                violation_details.append(f"{qname} [{pid}]: confidence non-null on veto ({rcode})")
            if isinstance(conf, dict):
                if conf.get("method") == "heuristic_negative":
                    q_violations["d_heuristic_negative_method"] += 1
                    violation_details.append(f"{qname} [{pid}]: confidence.method == 'heuristic_negative'")
                c_cal = conf.get("calibration_version")
                if top_cal_ver and c_cal != top_cal_ver:
                    q_violations["d_cal_version_mismatch_conf_vs_top"] += 1
                    violation_details.append(f"{qname} [{pid}]: confidence.calibration_version={c_cal!r} != top={top_cal_ver!r}")
                if health_cal_ver and c_cal != health_cal_ver:
                    q_violations["d_cal_version_mismatch_conf_vs_health"] += 1
                    violation_details.append(f"{qname} [{pid}]: confidence.calibration_version={c_cal!r} != /health={health_cal_ver!r}")

            # (e) reason_code
            if cls == "qualifies":
                if rcode is not None:
                    q_violations["e_reason_code_non_null_on_qualifies"] += 1
                    violation_details.append(f"{qname} [{pid}]: reason_code={rcode!r} on qualifies")
            else:
                if rcode is None:
                    q_violations["e_reason_code_null_on_non_qualifies"] += 1
                    violation_details.append(f"{qname} [{pid}]: reason_code=None on {cls}")
                elif rcode not in REASON_REGISTRY:
                    q_violations["e_reason_code_unknown"] += 1
                    violation_details.append(f"{qname} [{pid}]: unknown reason_code={rcode!r}")
                else:
                    reg = REASON_REGISTRY[rcode]
                    reason_registry_summary.append((pid, rcode, reg.label, reg.kind, cls))
                    if reg.label != cls:
                        q_violations["e_reason_label_mismatch"] += 1
                        violation_details.append(
                            f"{qname} [{pid}]: reason_code={rcode} has registry label={reg.label} (kind={reg.kind}) != cls={cls}"
                        )

            # (f) Related-compound collisions on qualifiers
            if cls == "qualifies" and ev is not None:
                ev_txt = ev.get("display_text") or ev.get("text") or ""
                terms = [entity] + list(aliases)
                has_standalone = any(
                    re.search(rf"(?<!\w){re.escape(t)}(?!\w)", ev_txt, re.IGNORECASE)
                    for t in terms if t
                )
                collides = any(is_related_compound_collision(ev_txt, t) for t in terms if t)
                # Also check substring inside longer chemical word e.g. dihydroartemisinic acid
                in_longer_only = False
                if not has_standalone:
                    in_longer_only = any(t.lower() in ev_txt.lower() for t in terms if t)
                elif collides:
                    in_longer_only = True
                if in_longer_only:
                    q_violations["f_related_compound_collision_on_qualifies"] += 1
                    violation_details.append(f"{qname} [{pid}]: related-compound collision in qualifier evidence: {ev_txt!r}")

            # (g) Entity-anchor gate (anaphora-safe)
            if cls == "qualifies":
                if ev is None:
                    q_violations["g_qualifies_without_evidence"] += 1
                    violation_details.append(f"{qname} [{pid}]: qualifies without evidence")
                else:
                    ev_txt = ev.get("display_text") or ev.get("text") or ""
                    terms = [entity] + list(aliases)
                    ev_has_entity = any(
                        re.search(rf"\b{re.escape(t)}\b", ev_txt, re.IGNORECASE)
                        for t in terms if t
                    )
                    ev_collides = any(is_related_compound_collision(ev_txt, t) for t in terms if t)

                    diag_obj = r.get("diagnostics") or {}
                    cond_diag = (r.get("conditions") or {}).get("activity", {}).get("diagnostics") or {}
                    ant_info = diag_obj.get("resolved_antecedent") or cond_diag.get("resolved_antecedent")

                    ant_valid = False
                    if ant_info and isinstance(ant_info, dict):
                        ant_txt = ant_info.get("text", "")
                        ant_has_entity = any(
                            re.search(rf"\b{re.escape(t)}\b", ant_txt, re.IGNORECASE)
                            for t in terms if t
                        )
                        ant_collides = any(is_related_compound_collision(ant_txt, t) for t in terms if t)
                        if ant_has_entity and not ant_collides:
                            ant_valid = True

                    if (not ev_has_entity or ev_collides) and not ant_valid:
                        q_violations["g_entity_anchor_or_anaphora_missing"] += 1
                        violation_details.append(
                            f"{qname} [{pid}]: qualifier evidence lacks entity anchor and has no valid resolved antecedent (ev={ev_txt[:40]!r})"
                        )

            # (h) Dedup
            if dedup_of is not None:
                if not str(dedup_of).strip() or str(dedup_of).strip().lower() == "unknown":
                    q_violations["h_invalid_dedup_of"] += 1
                    violation_details.append(f"{qname} [{pid}]: invalid dedup_of={dedup_of!r}")

            # (i) Complete parent sentence boundary (F4/F6)
            if ev is not None:
                d_txt = (ev.get("display_text") or ev.get("text") or "").strip()
                if not re.match(r'^[A-Z0-9"\'\[\(]', d_txt):
                    q_violations["i_sentence_not_capitalized_start"] += 1
                    violation_details.append(f"{qname} [{pid}]: evidence does not start with capital/opening punctuation: {d_txt[:25]!r}")
                if not re.search(r'[.!?]["\'\)\]]?$', d_txt):
                    q_violations["i_sentence_not_terminal_punct_end"] += 1
                    violation_details.append(f"{qname} [{pid}]: evidence does not end with terminal punctuation: {d_txt[-25:]!r}")
                if len(d_txt) < 30:
                    q_violations["i_sentence_too_short"] += 1
                    violation_details.append(f"{qname} [{pid}]: evidence length < 30 chars ({len(d_txt)}): {d_txt!r}")

            # (j) Degradation or bypass cues in qualifies evidence (F8)
            if cls == "qualifies" and ev is not None:
                ev_txt = ev.get("display_text") or ev.get("text") or ""
                bypass_pattern = r"\bwithout\s+(?:going\s+through|using|passing\s+through)\b|\bbypass\w*|\bnot\s+via\b"
                degrad_pattern = r"\bdegrad\w*|\bdecaffein\w*|\bcatabol\w*"
                if re.search(bypass_pattern, ev_txt, re.IGNORECASE):
                    q_violations["j_bypass_cue_in_qualifies_evidence"] += 1
                    violation_details.append(f"{qname} [{pid}]: bypass cue in qualifies evidence: {ev_txt[:50]!r}")
                if re.search(degrad_pattern, ev_txt, re.IGNORECASE):
                    q_violations["j_degradation_cue_in_qualifies_evidence"] += 1
                    violation_details.append(f"{qname} [{pid}]: degradation cue in qualifies evidence: {ev_txt[:50]!r}")

            # (k) Frozen snapshot join & offset verification
            if ev is not None:
                recs = snapshots_by_pid.get(pid, [])
                if isinstance(recs, dict):
                    recs = [recs]
                rec_sha = r.get("source_record_sha256") or r.get("text_sha256")
                snap_rec = next((s for s in recs if s.get("source_record_sha256") == rec_sha), recs[0] if recs else None)
                if snap_rec is not None:
                    raw_abs = snap_rec.get("abstractText") or snap_rec.get("abstract") or ""
                    canon_body = canonicalize(clean_markup_and_unescape(raw_abs))
                    cs = ev.get("char_start")
                    ce = ev.get("char_end")
                    d_txt = ev.get("display_text") or ev.get("text") or ""
                    if cs is not None and ce is not None:
                        if not (0 <= cs < ce <= len(canon_body)) or canon_body[cs:ce] != d_txt:
                            q_violations["k_snapshot_body_slice_mismatch"] += 1
                            violation_details.append(
                                f"{qname} [{pid}]: snapshot body[{cs}:{ce}] does not match evidence display_text"
                            )
                    snap_sha = snap_rec.get("source_record_sha256")
                    rec_sha = r.get("source_record_sha256") or r.get("text_sha256")
                    if snap_sha and rec_sha and snap_sha != rec_sha:
                        q_violations["k_snapshot_sha256_mismatch"] += 1
                        violation_details.append(
                            f"{qname} [{pid}]: source record sha256 mismatch: {rec_sha} != {snap_sha}"
                        )
                    diag_obj = r.get("diagnostics") or {}
                    cond_diag = (r.get("conditions") or {}).get("activity", {}).get("diagnostics") or {}
                    ant_info = diag_obj.get("resolved_antecedent") or cond_diag.get("resolved_antecedent")
                    if ant_info and isinstance(ant_info, dict):
                        ant_txt = ant_info.get("text", "")
                        if ant_txt and ant_txt not in canon_body:
                            q_violations["k_antecedent_not_in_snapshot"] += 1
                            violation_details.append(
                                f"{qname} [{pid}]: antecedent text not found in snapshot body: {ant_txt[:30]!r}"
                            )

            if cls == "qualifies":
                qualifies_rows.append((rank, pid, conf_str, ev.get("display_text") if ev else None))

        total_n = len(results)
        abstain_n = counts["insufficient_evidence"]
        abstain_rate = (abstain_n / total_n) if total_n else 0.0
        print(
            f"\n  TOTALS (N={total_n}): qualifies={counts['qualifies']} | "
            f"does_not_qualify={counts['does_not_qualify']} | "
            f"insufficient_evidence={abstain_n} | abstention_rate={abstain_rate:.4f}"
        )
        print(f"  INVARIANT VIOLATIONS (query): {dict(q_violations) if q_violations else '0 violations'}")
        if reason_registry_summary:
            uniq_codes = sorted({(rc, rl, rk, c) for _, rc, rl, rk, c in reason_registry_summary})
            print("  [e] reason_code registry check:", uniq_codes)

        if qualifies_rows:
            print("\n  QUALIFIES FULL EVIDENCE SENTENCES:")
            for rk, pid, cstr, etxt in qualifies_rows:
                print(f"    - Rank {rk} [{pid}] (conf={cstr}): {etxt}")
        else:
            print("\n  QUALIFIES FULL EVIDENCE SENTENCES: (none)")

        for k, v in q_violations.items():
            all_violations[k] += v

    print("\n" + "=" * 100)
    print("OVERALL INVARIANT SUMMARY:", dict(all_violations) if all_violations else "0 violations across all queries")
    if violation_details:
        print("\nVIOLATION DETAILS:")
        for d in violation_details:
            print("  *", d)
    print("=" * 100)
    return {"violations": dict(all_violations), "details": violation_details}


if __name__ == "__main__":
    eval_runs_dir = os.path.dirname(os.path.abspath(__file__))
    target = sys.argv[1] if len(sys.argv) > 1 else _find_latest_run(eval_runs_dir)
    check_run(target)

