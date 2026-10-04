"""Turn the profile_env reports into the numbers that go in docs/phase0_env.md.

Keeps the doc and the JSON from drifting apart: every figure quoted in the write-up is
printed here from the report files, not retyped by hand.

    python scripts/summarize_env_profile.py
"""
import json
import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
REPORTS = REPO_ROOT / "reports"

a = json.loads((REPORTS / "env.json").read_text())
b = json.loads((REPORTS / "env_runB.json").read_text())
s = json.loads((REPORTS / "env_shadow_seg.json").read_text())

print("=" * 78)
print("BUCKETS  (p50 ms, and share of pipeline wall time)")
print(f"{'bucket':<10}{'A p50':>9}{'B p50':>9}{'A %':>8}{'B %':>8}{'spread %':>10}")
for k in a["timing"]["buckets"]:
    sa, sb = a["timing"]["buckets"][k], b["timing"]["buckets"][k]
    pa = 100 * sa["total_s"] / a["timing"]["pipeline_wall_s"]
    pb = 100 * sb["total_s"] / b["timing"]["pipeline_wall_s"]
    spread = 100 * abs(sa["p50_ms"] - sb["p50_ms"]) / min(sa["p50_ms"], sb["p50_ms"])
    print(f"{k:<10}{sa['p50_ms']:>9.2f}{sb['p50_ms']:>9.2f}{pa:>7.1f}%{pb:>7.1f}%{spread:>9.1f}%")

print()
print("=" * 78)
for name, r in (("A", a), ("B", b)):
    t, th = r["timing"], r["throughput"]
    print(f"run {name}: {th['recorded_frames_per_wall_second']:.2f} frames/wall-s, "
          f"RTF {th['real_time_factor']:.2f}x, pipeline {t['pipeline_wall_s']:.2f} s, "
          f"timer overhead {t['timer_overhead_ns']:.0f} ns")
fa = a["throughput"]["recorded_frames_per_wall_second"]
fb = b["throughput"]["recorded_frames_per_wall_second"]
print(f"run-to-run throughput spread: {100 * abs(fa - fb) / min(fa, fb):.2f}%")

eps = a["timing"]["per_episode"]
tot = [e["frame_total_p50_ms"] for e in eps]
rst = [e["reset_s"] * 1e3 for e in eps]
print(f"run A per-episode (n={len(eps)}): frame p50 {min(tot):.2f}-{max(tot):.2f} ms, "
      f"reset {min(rst):.1f}-{max(rst):.1f} ms")

print()
print("=" * 78)
print("SHADOW + SEGMENTATION (the BaseAviary defaults) vs this camera's defaults")
ra = a["timing"]["buckets"]["render"]["p50_ms"]
rs = s["timing"]["buckets"]["render"]["p50_ms"]
print(f"render p50 : {ra:.2f} ms  ->  {rs:.2f} ms   ({rs / ra:.2f}x)")
print(f"throughput : {fa:.2f}  ->  "
      f"{s['throughput']['recorded_frames_per_wall_second']:.2f} frames/wall-s")
print(f"png bytes  : {a['bytes_per_frame']['png_on_disk']['mean']:.0f}  ->  "
      f"{s['bytes_per_frame']['png_on_disk']['mean']:.0f}")

print()
print("=" * 78)
print("BYTES PER FRAME")
bp = a["bytes_per_frame"]
raw = bp["raw_rgb_uint8"]
print(f"raw uint8 RGB      : {raw}")
print(f"png (on disk, cl{bp['png_on_disk']['compress_level']}) : "
      f"{bp['png_on_disk']['mean']:.0f} mean, {bp['png_on_disk']['p50']:.0f} p50, "
      f"{bp['png_on_disk']['min']:.0f}-{bp['png_on_disk']['max']:.0f} range "
      f"({raw / bp['png_on_disk']['mean']:.1f}x smaller than raw)")
for q, v in bp["jpeg_in_memory_subsample"].items():
    print(f"jpeg q{q}           : {v['mean']:.0f} mean  "
          f"({v['mean'] / bp['png_on_disk']['mean']:.2f}x the PNG size, "
          f"encode p50 {v['encode_p50_ms']:.2f} ms)")

print()
print("=" * 78)
print("REPLACING THE SECTION 4.4 PLANNING ASSUMPTIONS")
print("Measured: bytes/frame and frames/wall-second. The two right-hand columns are those")
print("numbers times a planned frame count -- projections, not measurements.")
png = bp["png_on_disk"]["mean"]
print(f"{'stage':<18}{'frames':>9}{'roadmap raw GB':>16}{'projected PNG GB':>18}"
      f"{'projected gen h':>17}")
for stage, episodes, raw_gb, est_h in (("v0.1 thin slice", 80, 0.295, "10-40 min"),
                                       ("v0.2 total", 2000, 7.373, "4.2-16.7 h")):
    frames = episodes * 100
    gb = frames * png / 1e9
    # The 1.5x is the roadmap's own overhead factor. It nominally covers reset, planning,
    # rejection and encoding, but reset and encoding are already inside the measured fps,
    # so this double-counts them and is kept only as a conservative margin.
    hours = frames / fa / 3600 * 1.5
    shown = f"{hours * 60:.1f} min" if hours < 1 else f"{hours:.2f} h"
    print(f"{stage:<18}{frames:>9}{raw_gb:>16.3f}{gb:>18.3f}{shown:>17}"
          f"   (was {est_h})")
print("Caveat: measured on a plane plus two untextured primitives; richer scenes compress "
      "worse and render slower, so both projections will grow.")

print()
print("=" * 78)
print("MEMORY / HOST")
m = a["memory_kb"]
print(f"VmRSS after imports {m['vmrss_after_imports'] / 1024:.1f} MB -> after first reset "
      f"{m['vmrss_after_first_reset'] / 1024:.1f} MB -> end {m['vmrss_at_end'] / 1024:.1f} MB")
print(f"VmHWM peak {m['vmhwm_peak'] / 1024:.1f} MB, ru_maxrss {m['ru_maxrss'] / 1024:.1f} MB")
h = a["host"]
print(f"WSL MemTotal {h['meminfo_memtotal_kb'] / 1024 / 1024:.1f} GiB, "
      f"MemAvailable {h['meminfo_memavailable_kb'] / 1024 / 1024:.1f} GiB, "
      f"disk free {h['disk_free_gb']:.0f} GB")
print(f"cpu affinity {h['cpu_count_affinity']}, loadavg {h['loadavg_1_5_15']}")
w = h.get("windows", {})
if w.get("collected") is False:
    print(f"windows: not merged ({w.get('reason')})")
elif "on_ac_power" in w:
    # Reports written before the staleness guard existed carry no age field.
    age = (f"captured {w['age_s_at_run']:.0f} s before the run"
           if "age_s_at_run" in w else "age not recorded (pre-guard report)")
    print(f"windows ({age}): AC={w.get('on_ac_power')}, "
          f"scheme={w.get('power_scheme_name')!r}, "
          f"instant CPU load {w.get('cpu_load_percent_instant')}%, "
          f"host free RAM {w.get('host_ram_free_mb')} MB")
else:
    print(f"windows: nothing usable in the host block ({sorted(w)})")
print(f"torch imported: {a['provenance']['torch_imported']}, "
      f"pybullet numpy enabled: {a['provenance']['pybullet_numpy_enabled']}")
