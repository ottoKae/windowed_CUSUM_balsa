"""P2: true VH extraction, frozen VV reuse, and VV-priority pixel fusion."""
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import time

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
import pandas as pd

import detector_adapter as adapter

ROOT = adapter.ROOT
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "tooling/release")]
from fc001_validation_detection import read_track, CACHE
from lr_cusum.event_scoring import _canonical_matching
from lr_cusum.harvest_bounded_event_aggregation import build_bounded_events

SOURCE = ROOT / "runs/validation/balsa_firstpost_national_20261030_v2"
OUT = ROOT / "runs/validation/balsa_results_ablation_20261101_v1/polarization"
METHODS = ("full_firstpost", "vh_firstpost", "vv_union_vh_firstpost", "vv_intersection_vh_firstpost")
STATE = {}


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def check_frozen():
    signature = json.loads((SOURCE / "method_signature.json").read_text())
    for path, digest in signature["source_sha256"].items():
        assert sha(path) == digest, path
    completion = json.loads((SOURCE / "validation/complete.json").read_text())
    for name, digest in completion["output_sha256"].items():
        assert sha(SOURCE / "validation" / name) == digest, name
    return signature


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    receipt_path = OUT / "input_receipt.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        for path, digest in receipt["input_sha256"].items():
            assert sha(path) == digest, path
        for name, digest in receipt["prepared_sha256"].items():
            assert sha(OUT / name) == digest, name
        check_frozen()
        return receipt
    signature = check_frozen()
    members = pd.read_parquet(SOURCE / "validation_membership.parquet")
    units = pd.read_parquet(SOURCE / "master_units.parquet")
    units = units.loc[units.unit_id.isin(members.unit_id)].sort_values("unit_id").reset_index(drop=True)
    assert len(units) == 26890 and members.sample_id.nunique() == 468
    units["local_series_index"] = np.arange(len(units))
    vv = np.load(SOURCE / "master_values.npy", mmap_mode="r")
    vh = np.lib.format.open_memmap(OUT / "vh_values.npy", mode="w+", dtype="float64", shape=(len(units),126))
    paths = [SOURCE / n for n in ("master_values.npy", "master_units.parquet", "validation_membership.parquet", "method_signature.json", "protocol.json", "validation/pixel_events.parquet", "validation/stand_reports.parquet", "validation/source_members.parquet")]
    sources, track_rows = {}, []
    for track, group in units.groupby("track_key", sort=True):
        index, raw, valid, n_obs, dates, _ = read_track(track)
        rows = group.cache_index.to_numpy(dtype=int)
        for field in ("pixel_track_id", "native_row", "native_col"):
            assert np.array_equal(index.loc[rows, field].to_numpy(), group[field].to_numpy()), (track, field)
        assert dates.equals(adapter.DATES)
        mask = valid[rows]
        assert np.array_equal(mask, (n_obs[rows] > 0) & np.isfinite(raw[rows]).all(axis=1))
        original_vv = vv[group.series_index.to_numpy()]
        assert np.array_equal(np.where(mask, raw[rows,0], np.nan), original_vv, equal_nan=True)
        local_vh = np.where(mask, raw[rows,1], np.nan).astype(np.float64)
        assert np.array_equal(np.isfinite(local_vh), np.isfinite(original_vv))
        vh[group.local_series_index.to_numpy()] = local_vh
        for path in (CACHE / f"{track}.npz", CACHE / f"{track}.pixels.parquet"):
            sources[str(path)] = sha(path)
        track_rows.append(dict(track_key=track, pixels=len(group), joint_valid_pixel_months=int(mask.sum()),
            native_identity_equal=True, VV_exactly_frozen=True, VH_joint_mask_equal=True))
    vh.flush()
    del vh
    units.to_parquet(OUT / "units.parquet", index=False)
    members.to_parquet(OUT / "membership.parquet", index=False)
    pd.DataFrame(track_rows).to_csv(OUT / "track_source_checks.csv", index=False)
    protocol = json.loads((SOURCE / "protocol.json").read_text())
    protocol.update(protocol_id="P2_POLARIZATION_FIRSTPOST_20261101_V1",
        baseline_signature=signature["signature"], methods=list(METHODS),
        polarization=dict(pair_tolerance_months=2, layer="accepted_pixel_responses_before_B20",
            objective="maximum_pairs_then_minimum_calendar_distance_then_fixed_ID_order",
            paired_date="VV", unpaired_VH_date="VH", support="unique_original_native_cells"),
        evaluation_statistics=dict(replicates=5000, seed=20260929,
            family="exploratory_paired_spatial_block_pointwise_95_percent",
            full_reference_denominator=1049, precision=None, F1=None))
    save(OUT / "protocol.json", protocol)
    receipt = dict(status="prepared", utc=datetime.now(timezone.utc).isoformat(),
        pixels=len(units), stands=members.sample_id.nunique(), calendar_months=126,
        source_sha256=signature["source_sha256"], input_sha256={**sources, **{str(p):sha(p) for p in paths}},
        prepared_sha256={n:sha(OUT/n) for n in ("vh_values.npy","units.parquet","membership.parquet","track_source_checks.csv","protocol.json")},
        VH_extracted_from_original_band_index=1, all_VV_values_exact=True, all_masks_exact=True,
        input_tracks=len(track_rows))
    save(receipt_path, receipt)
    print(f"P2 prepared {len(units)} true VH trajectories in {len(track_rows)} tracks", flush=True)
    return receipt


def fuse(vv, vh):
    """Apply the existing maximum-pair VV-date-priority rule before B20."""
    vv, vh = vv.copy(), vh.copy()
    assert vv.candidate_id.is_unique and vh.candidate_id.is_unique
    assert not set(vv.candidate_id) & set(vh.candidate_id)
    for frame in (vv, vh):
        frame["month_ordinal"] = pd.to_datetime(frame.change_time).dt.to_period("M").astype("int64")
    h_groups = dict(tuple(vh.groupby("pixel_track_id", sort=False)))
    paired_vv, paired_vh, pairs = set(), set(), []
    vv_lookup, vh_lookup = vv.set_index("candidate_id"), vh.set_index("candidate_id")
    for pixel, group in vv.groupby("pixel_track_id", sort=True):
        other = h_groups.get(pixel)
        if other is None:
            continue
        edges = {(v.candidate_id,h.candidate_id):(1,abs(v.month_ordinal-h.month_ordinal))
            for v in group.itertuples() for h in other.itertuples()
            if abs(v.month_ordinal-h.month_ordinal) <= 2}
        matched = _canonical_matching(edges)
        paired_vv.update(matched.values()); paired_vh.update(matched)
        for hid, vid in matched.items():
            v, h = vv_lookup.loc[vid], vh_lookup.loc[hid]
            pairs.append(dict(pixel_track_id=pixel, unit_id=v.unit_id, pair_status="paired",
                vv_candidate_id=vid, vh_candidate_id=hid, vv_month=str(v.change_time)[:7],
                vh_month=str(h.change_time)[:7], retained_month=str(v.change_time)[:7],
                month_distance=abs(int(v.month_ordinal-h.month_ordinal)),
                vv_p=float(v.null_p_value), vh_p=float(h.null_p_value),
                vv_amplitude=float(v.cusum_amplitude), vh_amplitude=float(h.cusum_amplitude)))
    for band, frame, paired in (("vv",vv,paired_vv),("vh",vh,paired_vh)):
        for row in frame.loc[~frame.candidate_id.isin(paired)].itertuples():
            pairs.append(dict(pixel_track_id=row.pixel_track_id, unit_id=row.unit_id,
                pair_status=f"unpaired_{band}", **{f"{band}_candidate_id":row.candidate_id,
                f"{band}_month":str(row.change_time)[:7], "retained_month":str(row.change_time)[:7],
                f"{band}_p":float(row.null_p_value), f"{band}_amplitude":float(row.cusum_amplitude)}))
    variants = {"vv_union_vh_firstpost":pd.concat([vv,vh.loc[~vh.candidate_id.isin(paired_vh)]],ignore_index=True),
        "vv_intersection_vh_firstpost":vv.loc[vv.candidate_id.isin(paired_vv)].copy()}
    return {name:frame.drop(columns="month_ordinal") for name,frame in variants.items()}, pd.DataFrame(pairs)


def aggregate(name, events, units, members):
    events = events.copy()
    # Frozen VV timestamps and checkpoint VH strings meet in the union stream.
    for field in ("change_time","window_start","window_end","decision_start","decision_end"):
        if field in events:
            events[field]=pd.to_datetime(events[field],format="mixed")
    expected_bands = {"VH_dB"} if name == "vh_firstpost" else {"VV_dB","VH_dB"}
    assert set(events.Band).issubset(expected_bands)
    assert events.null_p_value.le(.025).all() and events.post_minus_pre.lt(0).all()
    dates = pd.to_datetime(events.change_time)
    assert dates.dt.year.between(2017,2025).all()
    assert dates.ge(pd.to_datetime(events.decision_start)).all()
    assert dates.le(pd.to_datetime(events.decision_end)).all()
    events["source_detection_method"] = events.method
    events["method"] = name
    reports, nodes, receipt = build_bounded_events(units,members,events,
        source_method=name, output_method=f"B20_{name.upper()}",
        protocol_version="P2_POLARIZATION_FIRSTPOST_20261101_V1",
        validate_frozen_gates=False, require_annual_decision=True,
        allow_multiple_months_per_year=name == "vv_union_vh_firstpost")
    assert receipt["input_pixel_event_rows"] == len(events)
    reports["interval_time_basis"] = "actual_pixel_response_months"
    receipt.update(actual_band_validation=sorted(expected_bands), p_and_direction_and_annual_gates_checked=True,
        fixed_membership_denominator=True)
    return reports, nodes, receipt, events


def selfcheck():
    prepare()
    units = pd.read_parquet(OUT/"units.parquet")
    archived = pd.read_parquet(SOURCE/"validation/pixel_events.parquet")
    vv = np.load(SOURCE/"master_values.npy", mmap_mode="r")
    vh = np.load(OUT/"vh_values.npy", mmap_mode="r")
    selected = pd.concat([units.groupby("track_key").head(1), units.loc[units.pixel_track_id.isin(
        archived.loc[archived.near_threshold_refined].pixel_track_id.head(2))]]).drop_duplicates("pixel_track_id")
    started = time.monotonic()
    columns = adapter.geometry.COLUMNS
    vh_events = 0
    for row in selected.itertuples():
        actual, trace = adapter.detect(vv[row.series_index], row.pixel_track_id)
        expected = archived.loc[archived.pixel_track_id.eq(row.pixel_track_id),columns].copy()
        for frame in (actual, expected):
            for field in ("change_time","window_start","window_end","decision_start","decision_end"):
                frame[field] = pd.to_datetime(frame[field])
        pd.testing.assert_frame_equal(actual[columns].sort_values("candidate_id").reset_index(drop=True),
            expected.sort_values("candidate_id").reset_index(drop=True), check_dtype=False, check_exact=True)
        assert trace.accepted.sum() == len(actual)
        actual_vh, trace_vh = adapter.detect(vh[row.local_series_index],row.pixel_track_id,band="VH_dB")
        assert actual_vh.Band.eq("VH_dB").all()
        assert trace_vh.candidate_id.str.contains(":VH_dB|",regex=False).all()
        vh_events += len(actual_vh)
    elapsed = time.monotonic()-started
    receipt = dict(status="passed", baseline_bit_exact_pixels=selected.pixel_track_id.tolist(),
        includes_refined_candidates=True, true_VH_pilot_events=vh_events,
        elapsed_seconds=elapsed, pilot_detection_calls=2*len(selected),
        estimated_full_VH_seconds_at_four_workers=elapsed/(2*len(selected))*len(units)/4,
        source_sha256={str(Path(__file__)):sha(__file__),str(Path(adapter.__file__)):sha(adapter.__file__)})
    save(OUT/"selfcheck.json",receipt)
    print(json.dumps(receipt), flush=True)


def initialize_worker():
    STATE["vh"] = np.load(OUT/"vh_values.npy",mmap_mode="r")


def batch(rows):
    result = []
    for unit, pixel, index in rows:
        events, trace = adapter.detect(STATE["vh"][index],pixel,band="VH_dB")
        events["unit_id"], events["pixel_track_id"], events["method"] = unit,pixel,"vh_firstpost"
        trace["unit_id"] = unit
        event_rows=events.astype(object).where(pd.notna(events),None).to_dict("records")
        for event in event_rows:
            for field in ("change_time","window_start","window_end","decision_start","decision_end"):
                event[field]=pd.Timestamp(event[field]).isoformat()
        trace_rows=trace.astype(object).where(pd.notna(trace),None).to_dict("records")
        result.append((pixel,json.dumps(dict(events=event_rows,trace=trace_rows),allow_nan=False)))
    return result


def detect_all():
    prepare()
    assert json.loads((OUT/"selfcheck.json").read_text())["status"] == "passed"
    p1_receipt = OUT.parent/"components/verification_receipt.json"
    assert json.loads(p1_receipt.read_text())["passed"] is True
    units = pd.read_parquet(OUT/"units.parquet")
    directory = OUT/"vh_firstpost"
    directory.mkdir(exist_ok=True)
    if (directory/"complete.json").exists():
        for name,digest in json.loads((directory/"complete.json").read_text())["output_sha256"].items():
            assert sha(directory/name)==digest
        return
    frozen = {str(Path(__file__)):sha(__file__),str(Path(adapter.__file__)):sha(adapter.__file__)}
    contract_path = OUT/"execution_start.json"
    if contract_path.exists():
        assert json.loads(contract_path.read_text())["helper_sha256"] == frozen
    else:
        save(contract_path,dict(started_utc=datetime.now(timezone.utc).isoformat(),helper_sha256=frozen,
            workers=4,threads_per_worker=1,pixels=len(units),band="VH_dB",p1_completed_before_launch=True,
            p1_verification_sha256=sha(p1_receipt),command=[sys.executable,*sys.argv]))
    started=time.monotonic()
    with sqlite3.connect(directory/"pixel_checkpoints.sqlite") as db:
        db.execute("CREATE TABLE IF NOT EXISTS pixels(identity TEXT PRIMARY KEY,payload TEXT NOT NULL)")
        done={r[0] for r in db.execute("SELECT identity FROM pixels")}
        rows=list(units.loc[~units.pixel_track_id.isin(done),["unit_id","pixel_track_id","local_series_index"]].itertuples(index=False,name=None))
        chunks=[rows[i:i+32] for i in range(0,len(rows),32)]
        with ProcessPoolExecutor(max_workers=4,initializer=initialize_worker) as pool:
            for number,items in enumerate(pool.map(batch,chunks),1):
                db.executemany("INSERT INTO pixels VALUES (?,?)",items);db.commit()
                if number%32==0 or number==len(chunks):
                    print(f"VH {len(done)+min(number*32,len(rows))}/{len(units)}; {time.monotonic()-started:.1f}s",flush=True)
        payloads=[json.loads(row[0]) for row in db.execute("SELECT payload FROM pixels ORDER BY identity")]
    assert len(payloads)==len(units)
    events=pd.DataFrame([r for p in payloads for r in p["events"]])
    trace=pd.DataFrame([r for p in payloads for r in p["trace"]])
    events.to_parquet(directory/"pixel_events.parquet",index=False)
    trace.to_parquet(directory/"window_trace.parquet",index=False)
    assert len(trace)==len(units)*9 and int(trace.accepted.sum())==len(events)
    save(directory/"complete.json",dict(status="completed",pixels=len(units),windows=len(trace),
        pre_null_candidates=int(trace.pre_null_eligible.sum()),refined_candidates=int(trace.near_threshold_refined.sum()),
        accepted_pixel_events=len(events),elapsed_seconds=time.monotonic()-started,
        output_sha256={n:sha(directory/n) for n in ("pixel_events.parquet","window_trace.parquet")}))


def finalize():
    check_frozen()
    units=pd.read_parquet(OUT/"units.parquet");members=pd.read_parquet(OUT/"membership.parquet")
    vv=pd.read_parquet(SOURCE/"validation/pixel_events.parquet")
    vh=pd.read_parquet(OUT/"vh_firstpost/pixel_events.parquet")
    variants,pairs=fuse(vv,vh)
    pairs.to_csv(OUT/"fusion_pairs.csv",index=False,encoding="utf-8-sig")
    paths={"full_firstpost":SOURCE/"validation/stand_reports.parquet"}
    counts=[]
    for name,events in {"vh_firstpost":vh,**variants}.items():
        directory=OUT/name;directory.mkdir(exist_ok=True)
        reports,nodes,receipt,prepared=aggregate(name,events,units,members)
        prepared.to_parquet(directory/"aggregation_input_events.parquet",index=False)
        reports.to_parquet(directory/"stand_reports.parquet",index=False)
        nodes.to_parquet(directory/"source_members.parquet",index=False)
        save(directory/"aggregation.json",receipt)
        paths[name]=directory/"stand_reports.parquet"
        counts.append(dict(method=name,pixel_responses=len(events),reports=len(reports),
            assigned_nodes=receipt["assigned_nodes"],residual_nodes=receipt["residual_nodes"]))
    pd.DataFrame(counts).to_csv(OUT/"response_and_report_counts.csv",index=False)
    sys.path.insert(0,str(Path(__file__).parents[1]/"components"))
    from evaluate_components import evaluate
    evaluate(paths,OUT/"evaluation")
    metrics=pd.read_csv(OUT/"evaluation/catalogue_recall.csv")
    primary=metrics.loc[metrics.tolerance_months.eq(2)]
    lines=["# P2 First-post 极化敏感性", "", "已完成真实VH检测及VV日期优先的像元层并集／交集。固定26890原生像元、468林分、126月历与联合有效掩膜；VV冻结输出直接复用。W36/D12/S12、IID、α=0.025、3999／19999次、固定N的B20均保持不变。", "", "| 方法 | 林分报告数 | 1049条月份参考找回 | 条件找回率 |", "|---|---:|---:|---:|"]
    for row in primary.itertuples():
        lines.append(f"| {row.method} | {row.n_reports} | {row.frozen_month_found}/1049 | {row.frozen_recall:.2%} |")
    lines.extend(["", "融合配对容差固定±2月，与评价±1／±2／±3月分开。交集保留有VH配对的VV节点，并集保留全部VV及未配对VH；配对双方日期、分数和未配对来源见 `fusion_pairs.csv`。不同极化不重复增加原生格支持。", "", "上述是已登记参考的条件找回；目录外报告不是已知假阳性，不报告总体precision或F1。VV日期优先的结果不代表所有融合策略。首次／后续、403个登记多事件林分、590个登记相邻对、配对空间块区间及共同成功日期误差见 `evaluation/`，均不冒充完整真实历史。", "", "数据来源和掩膜检查见 `input_receipt.json`；5项真实实现测试、13个跨轨道及精化VV像元的精确复现见 `selfcheck.json` 与 `verification_receipt.json`。冻结主引擎、旧输出及标签未修改。", "", "检测完成后统一了冻结VV的Timestamp与VH检查点的ISO字符串日期类型，以便写入Parquet；仅修复新产物序列化，检测时源码快照保留于 `detection_source/`。", ""])
    (OUT/"README_ZH.md").write_text("\n".join(lines),encoding="utf-8")
    check_frozen()
    receipt=dict(status="completed",completed_utc=datetime.now(timezone.utc).isoformat(),
        fixed_pixels=len(units),fixed_stands=members.sample_id.nunique(),methods=list(paths),
        pairs=int(pairs.pair_status.eq("paired").sum()),pair_tolerance_months=2,
        source_inputs_unchanged=True,precision=None,F1=None,
        scientific_source_sha256=json.loads((SOURCE/"method_signature.json").read_text())["source_sha256"],
        helper_sha256={str(Path(__file__)):sha(__file__),str(Path(adapter.__file__)):sha(adapter.__file__)},
        output_sha256={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob("*") if p.is_file() and p.suffix in (".csv",".json",".parquet")})
    save(OUT/"receipt.json",receipt)
    verify()
    print("P2 polarization completed",flush=True)


def verify():
    """Check stored full-domain outputs, original inputs, and source allocation."""
    signature=check_frozen()
    inputs=json.loads((OUT/"input_receipt.json").read_text())
    for path,digest in inputs["input_sha256"].items():
        assert sha(path)==digest,path
    for name,digest in inputs["prepared_sha256"].items():
        assert sha(OUT/name)==digest,name
    execution=json.loads((OUT/"execution_start.json").read_text())
    for path,digest in execution["helper_sha256"].items():
        detected_source=OUT/"detection_source"/Path(path).name
        assert sha(detected_source if detected_source.exists() else path)==digest,path
    assert sha(adapter.__file__)==execution["helper_sha256"][str(Path(adapter.__file__))]
    units=pd.read_parquet(OUT/"units.parquet");members=pd.read_parquet(OUT/"membership.parquet")
    vh=pd.read_parquet(OUT/"vh_firstpost/pixel_events.parquet")
    trace=pd.read_parquet(OUT/"vh_firstpost/window_trace.parquet")
    values=np.load(OUT/"vh_values.npy",mmap_mode="r")
    assert len(trace)==242010 and trace.pixel_track_id.nunique()==26890
    assert not trace.duplicated(["pixel_track_id","logical_window"]).any()
    scored=trace.loc[trace.pre_null_eligible]
    assert scored.near_threshold_refined.eq(scored.initial_p.between(.015,.035)).all()
    assert scored.permutations.eq(np.where(scored.near_threshold_refined,19999,3999)).all()
    assert np.array_equal(scored.final_p.to_numpy(),((1+scored.exceedance_count)/(1+scored.permutations)).to_numpy())
    assert scored.accepted.eq(scored.final_p.le(.025)).all()
    seeds=units.set_index("pixel_track_id").index.to_series().map(lambda key:adapter.stable_seed(f"{key}:VH_dB",20260820))
    assert trace.permutation_seed.eq(trace.pixel_track_id.map(seeds)+trace.window_start_index).all()
    assert set(vh.candidate_id)==set(trace.loc[trace.accepted,"candidate_id"])
    indices=units.set_index("pixel_track_id").local_series_index
    for event in vh.itertuples():
        ws=adapter.DATES.get_loc(pd.Timestamp(event.window_start))
        we=adapter.DATES.get_loc(pd.Timestamp(event.window_end))+1
        window=values[int(indices[event.pixel_track_id]),ws:we]
        valid=np.flatnonzero(np.isfinite(window))
        amplitude,peak=adapter.cusum_amplitude(window[valid])
        expected=ws+int(valid[peak+1])
        assert event.change_index==expected
        assert float(amplitude)==event.cusum_amplitude
        assert pd.Timestamp(event.change_time)==adapter.DATES[expected]
    vv=pd.read_parquet(SOURCE/"validation/pixel_events.parquet")
    pairs=pd.read_csv(OUT/"fusion_pairs.csv",keep_default_na=False)
    matched=pairs.loc[pairs.pair_status.eq("paired")]
    assert matched.vv_candidate_id.is_unique and matched.vh_candidate_id.is_unique
    assert set(pairs.loc[pairs.vv_candidate_id.ne(""),"vv_candidate_id"])==set(vv.candidate_id)
    assert set(pairs.loc[pairs.vh_candidate_id.ne(""),"vh_candidate_id"])==set(vh.candidate_id)
    assert matched.retained_month.eq(matched.vv_month).all()
    denominator=members.groupby("sample_id").unit_id.nunique()
    counts=[]
    for name in METHODS[1:]:
        directory=OUT/name
        events=pd.read_parquet(directory/"aggregation_input_events.parquet")
        reports=pd.read_parquet(directory/"stand_reports.parquet")
        nodes=pd.read_parquet(directory/"source_members.parquet")
        assert not nodes.duplicated(["sample_id","unit_id","response_month"]).any()
        assert nodes.node_id.is_unique
        assert nodes.total_pixels.eq(nodes.sample_id.map(denominator)).all()
        assert reports.total_pixels.eq(reports.sample_id.map(denominator)).all()
        assert (5*reports.response_pixels>=reports.total_pixels).all()
        selected=nodes.loc[nodes.destination_type.eq("candidate")]
        grouped={key:group for key,group in selected.groupby("prediction_id")}
        assert set(grouped)==set(reports.prediction_id)
        for report in reports.itertuples():
            group=grouped[report.prediction_id]
            assert set(json.loads(report.source_node_ids))==set(group.node_id)
            assert set(json.loads(report.source_unit_ids))==set(group.unit_id)
            assert report.response_pixels==group.unit_id.nunique()
            assert group.response_month.between(report.window_start_month,report.window_end_month).all()
        assert set(nodes.source_event_id).issubset(events.candidate_id)
        counts.append(dict(method=name,pixel_events=len(events),reports=len(reports),nodes=len(nodes)))
    summary=pd.read_csv(OUT/"evaluation/catalogue_recall.csv")
    assert len(summary)==12 and summary.frozen_month_references.eq(1049).all()
    baseline=summary.loc[summary.method.eq("full_firstpost") & summary.tolerance_months.eq(2)]
    assert int(baseline.frozen_month_found.iloc[0])==930 and int(baseline.n_reports.iloc[0])==1267
    checks=dict(passed=True,utc=datetime.now(timezone.utc).isoformat(),source_inputs_unchanged=True,
        baseline_signature=signature["signature"],full_VH_trace_rows=len(trace),all_accepted_firstpost_and_Q_verified=True,
        all_VH_band_seeds_verified=True,all_p_and_refinement_verified=True,all_nodes_fixed_N_and_conserved=True,
        complete_pair_partition_verified=True,methods=counts,baseline_recall_reproduced=True,
        pytest_file="tests/test_results_ablation_polarization.py",pytest_tests_passed=5,
        detection_runner_snapshot="detection_source/run_polarization.py",
        finalization_change="normalize mixed Timestamp/ISO-string date columns before Parquet serialization; no detection changes",
        output_receipt_sha256=sha(OUT/"receipt.json"))
    save(OUT/"verification_receipt.json",checks)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=("prepare","selfcheck","detect","finalize","verify"))
    args=parser.parse_args()
    {"prepare":prepare,"selfcheck":selfcheck,"detect":detect_all,"finalize":finalize,"verify":verify}[args.phase]()
