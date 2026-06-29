"""Generate an HTML report comparing current bench results with baseline.

Inputs:
  --prefill-current  path to the latest prefill_throughput_results.json
  --decode-current   path to the latest decode_tpot_results.json
  --prefill-baseline (optional) path to baseline prefill JSON
  --decode-baseline  (optional) path to baseline decode JSON
  --build-info-json  path to a JSON dict with build context (branch, image, etc.)
  --out-dir          where to write index.html (and copy PNGs into)

When a baseline is missing, the report shows current-only for that section
and prints a banner explaining why. The pipeline does NOT fail in that case.

Gate logic (when --fail-threshold-pct is given):
  Per-row regressions are noisy on small absolute values (e.g. tpot 0.10 -> 0.14
  is +40% but bench-noise sized). Instead we aggregate all rows via the
  geometric mean of per-row src/tgt ratios:
    prefill (higher better): regress iff geomean(src/tgt) < 1 - threshold/100
    decode  (lower better):  regress iff geomean(src/tgt) > 1 + threshold/100
  One bad row no longer fails the build; a systematic shift across many rows
  does. The HTML report still prints per-row Δ% for diagnosis.

Schema:
  prefill JSON:
    {"single_req": {}, "concurrency": {<prefix_ratio>: {"prompt_len_list": [...], "throughput_list": [...]}}}
  decode JSON:
    {"bs_list": [...], "tpot_var_bs_list": [{"bs": <int>, "prompt_len_list": [...], "tpot_var_len_list": [...]}, ...]}
"""

import argparse
import html
import json
import math
import os
import sys
from pathlib import Path


def _load(path):
    with open(path) as f:
        return json.load(f)


def _diff_pct(cur, base):
    """Return signed pct change (cur - base) / base * 100, or None if base falsy."""
    if base in (None, 0):
        return None
    return (cur - base) / base * 100.0


def _geomean(values):
    """Geometric mean of strictly-positive floats. None/<=0 entries dropped.
    Returns None if no usable values."""
    clean = [v for v in values if v is not None and v > 0]
    if not clean:
        return None
    return math.exp(sum(math.log(v) for v in clean) / len(clean))


def _diff_cell(cur, base, higher_is_better):
    """Return inline-styled <td> cells: cur, base, diff%. Pure cosmetics."""
    cur_s = "n/a" if cur is None else f"{cur:.4f}"
    base_s = "n/a" if base is None else f"{base:.4f}"
    pct = _diff_pct(cur, base) if cur is not None else None
    if pct is None:
        return f"<td>{cur_s}</td><td>{base_s}</td><td>n/a</td>"
    sign = "+" if pct >= 0 else ""
    is_improvement = (pct >= 0) if higher_is_better else (pct <= 0)
    color = "#1a7f37" if is_improvement else "#d1242f"
    return f"<td>{cur_s}</td><td>{base_s}</td><td style='color:{color}'>{sign}{pct:.2f}%</td>"


def _aggregate(pairs, higher_is_better, threshold_pct):
    """Compute geomean(cur/base) over (cur, base) pairs and decide regression.

    Returns (geomean, shift_pct, regressed) where:
      geomean      = geometric mean of cur/base ratios (None if no valid rows)
      shift_pct    = (geomean - 1) * 100, signed in cur-vs-base terms
      regressed    = bool, True iff shift exceeds threshold in the bad direction
                     (and threshold_pct is not None)
    """
    ratios = []
    for cur, base in pairs:
        if cur is None or base in (None, 0) or cur <= 0:
            continue
        ratios.append(cur / base)
    gmean = _geomean(ratios)
    if gmean is None:
        return None, None, False
    shift_pct = (gmean - 1.0) * 100.0
    if threshold_pct is None:
        return gmean, shift_pct, False
    if higher_is_better:
        # cur dropped: shift_pct < 0
        regressed = shift_pct < -threshold_pct
    else:
        # cur grew: shift_pct > 0
        regressed = shift_pct > threshold_pct
    return gmean, shift_pct, regressed


def render_prefill(cur, base):
    """prefill: rows = (prefix_ratio, prompt_len), value = throughput.
    higher is better. Returns (html_str, list_of_(cur,base)_pairs)."""
    out = ['<h2>Prefill throughput (tokens/s)</h2>']
    pairs = []
    cur_conc = (cur or {}).get("concurrency", {})
    base_conc = (base or {}).get("concurrency", {}) if base else {}

    if not cur_conc:
        out.append('<p><em>no concurrency results in current run</em></p>')
        return "\n".join(out), pairs

    if base is None:
        out.append('<p style="color:#ef6c00">⚠ No baseline available -- showing current only.</p>')
        out.append('<table><thead><tr><th>prefix_ratio</th><th>prompt_len</th><th>throughput</th></tr></thead><tbody>')
        for ratio, payload in cur_conc.items():
            for plen, tput in zip(payload["prompt_len_list"], payload["throughput_list"]):
                out.append(f"<tr><td>{ratio}</td><td>{plen}</td><td>{tput:.2f}</td></tr>")
        out.append("</tbody></table>")
        return "\n".join(out), pairs

    out.append('<table><thead><tr><th>prefix_ratio</th><th>prompt_len</th>'
               '<th>source</th><th>target</th><th>Δ%</th></tr></thead><tbody>')
    for ratio, payload in cur_conc.items():
        base_payload = base_conc.get(ratio, {})
        base_map = dict(zip(base_payload.get("prompt_len_list", []),
                            base_payload.get("throughput_list", [])))
        for plen, tput in zip(payload["prompt_len_list"], payload["throughput_list"]):
            base_v = base_map.get(plen)
            pairs.append((tput, base_v))
            out.append(f"<tr><td>{ratio}</td><td>{plen}</td>"
                       f"{_diff_cell(tput, base_v, higher_is_better=True)}</tr>")
    out.append("</tbody></table>")
    return "\n".join(out), pairs


def render_decode(cur, base):
    """decode: rows = (bs, prompt_len), value = tpot_var (seconds).
    lower is better. Returns (html_str, list_of_(cur,base)_pairs)."""
    out = ['<h2>Decode TPOT (seconds, lower is better)</h2>']
    pairs = []
    cur_groups = (cur or {}).get("tpot_var_bs_list", [])
    base_groups = (base or {}).get("tpot_var_bs_list", []) if base else []

    if not cur_groups:
        out.append('<p><em>no tpot results in current run</em></p>')
        return "\n".join(out), pairs

    if base is None:
        out.append('<p style="color:#ef6c00">⚠ No baseline available -- showing current only.</p>')
        out.append('<table><thead><tr><th>bs</th><th>prompt_len</th><th>tpot</th></tr></thead><tbody>')
        for grp in cur_groups:
            for plen, tpot in zip(grp["prompt_len_list"], grp["tpot_var_len_list"]):
                out.append(f"<tr><td>{grp['bs']}</td><td>{plen}</td><td>{tpot:.6f}</td></tr>")
        out.append("</tbody></table>")
        return "\n".join(out), pairs

    base_by_bs = {g["bs"]: dict(zip(g["prompt_len_list"], g["tpot_var_len_list"])) for g in base_groups}
    out.append('<table><thead><tr><th>bs</th><th>prompt_len</th>'
               '<th>source</th><th>target</th><th>Δ%</th></tr></thead><tbody>')
    for grp in cur_groups:
        base_map = base_by_bs.get(grp["bs"], {})
        for plen, tpot in zip(grp["prompt_len_list"], grp["tpot_var_len_list"]):
            base_v = base_map.get(plen)
            pairs.append((tpot, base_v))
            out.append(f"<tr><td>{grp['bs']}</td><td>{plen}</td>"
                       f"{_diff_cell(tpot, base_v, higher_is_better=False)}</tr>")
    out.append("</tbody></table>")
    return "\n".join(out), pairs


def render_summary(prefill_pairs, decode_pairs, threshold_pct):
    """Top-of-report aggregate gate verdict + per-section geomean numbers.

    Returns (html_block, regressed_sections_list). regressed_sections is a
    list of dicts with {kind, shift_pct, threshold_pct} for everything that
    breached the gate -- empty means PASS.
    """
    pf_g, pf_shift, pf_reg = _aggregate(prefill_pairs, True, threshold_pct)
    dc_g, dc_shift, dc_reg = _aggregate(decode_pairs, False, threshold_pct)

    breached = []
    if pf_reg:
        breached.append({"kind": "prefill", "metric": "throughput geomean",
                         "shift_pct": pf_shift, "threshold_pct": threshold_pct})
    if dc_reg:
        breached.append({"kind": "decode", "metric": "tpot geomean",
                         "shift_pct": dc_shift, "threshold_pct": threshold_pct})

    def _line(name, gmean, shift, regressed, higher_better):
        if gmean is None:
            return f"<li>{name}: <em>n/a (no comparable rows)</em></li>"
        # Color based on direction (improvement vs regression).
        if higher_better:
            good = shift >= 0
        else:
            good = shift <= 0
        color = "#1a7f37" if good else "#d1242f"
        sign = "+" if shift >= 0 else ""
        tag = " <strong style='color:#82061e'>[REGRESSION]</strong>" if regressed else ""
        return (f"<li>{name}: geomean(src/tgt) = {gmean:.4f} "
                f"(<span style='color:{color}'>{sign}{shift:.2f}%</span>){tag}</li>")

    parts = ["<h2>Aggregate gate</h2><ul>"]
    parts.append(_line("prefill (higher better)", pf_g, pf_shift, pf_reg, True))
    parts.append(_line("decode  (lower better)", dc_g, dc_shift, dc_reg, False))
    parts.append("</ul>")
    if threshold_pct is None:
        parts.append("<p><em>no threshold configured; gate disabled.</em></p>")
    else:
        parts.append(f"<p><em>gate threshold: ±{threshold_pct:.1f}% on the geomean.</em></p>")
    return "\n".join(parts), breached


def render_meta(meta):
    rows = []
    for key in ("build_num", "source_branch", "target_branch",
                "image", "model_path",
                "prefill_args", "decode_args",
                "src_prefill_serve_args", "src_decode_serve_args",
                "tgt_prefill_serve_args", "tgt_decode_serve_args",
                "src_prefill_serve_envs", "src_decode_serve_envs",
                "tgt_prefill_serve_envs", "tgt_decode_serve_envs",
                "regression_threshold_pct",
                "build_url"):
        if key in meta:
            rows.append(f"<tr><th>{html.escape(key)}</th><td>{html.escape(str(meta[key]))}</td></tr>")
    return f"<h2>Build context</h2><table>{''.join(rows)}</table>"


def render_pngs(out_dir):
    """Render plots grouped by metric, with source/target side-by-side.

    Filenames are expected to be `<branch>_<TIMESTAMP>_<metric>.png` where
    <branch> is `source` or `target` (the report stage stages them this way).
    Falls back to a flat list for any unprefixed file so we don't lose plots
    if the naming convention changes.
    """
    pngs = sorted(Path(out_dir).glob("*.png"))
    if not pngs:
        return ""

    # Bucket files by metric (prefill vs decode), tracking which side they are.
    metrics = {"prefill_throughput": {}, "decode_tpot": {}}
    leftovers = []
    for p in pngs:
        name = p.name
        side = None
        if name.startswith("source_"):
            side = "source"
        elif name.startswith("target_"):
            side = "target"
        if side and "prefill_throughput" in name:
            metrics["prefill_throughput"][side] = name
        elif side and "decode_tpot" in name:
            metrics["decode_tpot"][side] = name
        else:
            leftovers.append(name)

    parts = ["<h2>Plots</h2>"]
    titles = [
        ("prefill_throughput", "Prefill throughput"),
        ("decode_tpot", "Decode TPOT"),
    ]
    for key, label in titles:
        sides = metrics[key]
        if not sides:
            continue
        parts.append(f"<h3>{html.escape(label)}</h3>")
        # Side-by-side flex row so source / target are easy to compare.
        parts.append('<div style="display:flex;flex-wrap:wrap;gap:16px">')
        for side in ("source", "target"):
            name = sides.get(side)
            if not name:
                continue
            caption = "source (current)" if side == "source" else "target (baseline)"
            parts.append(
                '<figure style="flex:1 1 480px;margin:0">'
                f'<figcaption style="font-weight:600;color:var(--fg);margin-bottom:6px">'
                f'{html.escape(caption)}</figcaption>'
                f'<img src="{html.escape(name)}" '
                'style="max-width:100%;border:1px solid #ddd;padding:4px"/>'
                '</figure>'
            )
        parts.append("</div>")

    for name in leftovers:
        parts.append(f'<h3>{html.escape(name)}</h3>'
                     f'<img src="{html.escape(name)}" '
                     'style="max-width:100%;border:1px solid #ddd;padding:4px"/>')
    return "\n".join(parts)


CSS = """
:root {
  --fg: #1f2328;
  --fg-muted: #57606a;
  --bg: #ffffff;
  --bg-soft: #f6f8fa;
  --border: #d0d7de;
  --border-soft: #eaeef2;
  --accent: #0969da;
}
* { box-sizing: border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Helvetica Neue",
               "PingFang SC", "Microsoft YaHei", Arial, sans-serif;
  font-size: 14px;
  line-height: 1.55;
  color: var(--fg);
  background: var(--bg);
  margin: 0;
  padding: 32px clamp(16px, 4vw, 48px);
  max-width: 1200px;
}
h1 {
  margin: 0 0 24px;
  font-size: 24px;
  font-weight: 600;
  letter-spacing: -0.01em;
}
h2 {
  margin: 32px 0 12px;
  padding-bottom: 6px;
  border-bottom: 1px solid var(--border-soft);
  font-size: 18px;
  font-weight: 600;
  letter-spacing: -0.005em;
}
h3 {
  margin: 16px 0 8px;
  font-size: 14px;
  font-weight: 500;
  color: var(--fg-muted);
}
p { margin: 8px 0; }
em { color: var(--fg-muted); }
table {
  border-collapse: separate;
  border-spacing: 0;
  margin: 8px 0 16px;
  border: 1px solid var(--border);
  border-radius: 6px;
  overflow: hidden;
  font-size: 13px;
}
th, td {
  padding: 8px 14px;
  text-align: right;
  border-bottom: 1px solid var(--border-soft);
  font-variant-numeric: tabular-nums;
}
th {
  background: var(--bg-soft);
  font-weight: 600;
  color: var(--fg);
  white-space: nowrap;
}
tbody tr:last-child td { border-bottom: none; }
tbody tr:hover { background: var(--bg-soft); }
th:first-child, td:first-child { text-align: left; }
/* Build context table is key/value: keep keys narrow + monospace value-ish */
h2 + table th { width: 180px; }
h2 + table td { font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace; font-size: 12px; word-break: break-all; }
img { display: block; margin: 8px 0; max-width: 100%; border: 1px solid var(--border-soft); border-radius: 6px; }
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefill-current", required=True)
    ap.add_argument("--decode-current", required=True)
    ap.add_argument("--prefill-baseline")
    ap.add_argument("--decode-baseline")
    ap.add_argument("--build-info-json", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--fail-threshold-pct", type=float, default=None,
                    help="Aggregate regression threshold (in percent) on the "
                         "geometric mean of per-row src/tgt ratios. Prefill "
                         "fails when the geomean drops more than this; decode "
                         "fails when it rises more. Omit to disable the gate.")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    prefill_current = _load(args.prefill_current)
    decode_current = _load(args.decode_current)
    prefill_baseline = _load(args.prefill_baseline) if args.prefill_baseline else None
    decode_baseline = _load(args.decode_baseline) if args.decode_baseline else None

    if args.build_info_json == "/dev/stdin":
        meta = json.load(sys.stdin)
    else:
        meta = _load(args.build_info_json)

    threshold = args.fail_threshold_pct

    prefill_html, prefill_pairs = render_prefill(prefill_current, prefill_baseline)
    decode_html, decode_pairs = render_decode(decode_current, decode_baseline)
    summary_html, breached = render_summary(prefill_pairs, decode_pairs, threshold)

    parts = [
        f"<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>auto-mr-perf #{meta.get('build_num', '?')}</title>"
        f"<style>{CSS}</style></head><body>",
        f"<h1>auto-mr-perf report #{html.escape(str(meta.get('build_num', '?')))}</h1>",
    ]
    if breached:
        worst = max(breached, key=lambda b: abs(b["shift_pct"]))
        sign = "+" if worst["shift_pct"] >= 0 else ""
        banner = (f"<p style='padding:12px;border:1px solid #d1242f;"
                  f"background:#ffebe9;border-radius:6px;color:#82061e'>"
                  f"<strong>⛔ Aggregate gate failed.</strong> "
                  f"{worst['kind']} {worst['metric']} shifted "
                  f"{sign}{worst['shift_pct']:.2f}% "
                  f"(threshold ±{worst['threshold_pct']:.1f}%).</p>")
        parts.append(banner)
    parts.extend([
        render_meta(meta),
        summary_html,
        prefill_html,
        decode_html,
        render_pngs(out),
        "</body></html>",
    ])

    (out / "index.html").write_text("\n".join(parts))
    print(f"wrote {out / 'index.html'}")

    # regressions.txt: empty = pass; non-empty lines describe each section
    # that breached the aggregate gate.
    reg_path = out / "regressions.txt"
    with reg_path.open("w") as f:
        for b in breached:
            f.write(f"{b['shift_pct']:+.4f}\t{b['kind']}\t{b['metric']}\n")
    print(f"wrote {reg_path} ({len(breached)} sections regressed)")


if __name__ == "__main__":
    main()
