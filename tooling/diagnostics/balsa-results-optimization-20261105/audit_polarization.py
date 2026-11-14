"""Verify raw masks and the actual frozen joint-mask VV/VH inputs."""
import sys

import numpy as np
import pandas as pd

from run import ROOT,RUN,SOURCE,digest,write_json

sys.path.insert(0,str(ROOT/"scripts"))
from fc001_validation_detection import read_track,CACHE
from fullchain_ablation import eligibility_matrix


def main():
    source=ROOT/"runs/validation/balsa_results_ablation_20261101_v1/polarization"
    units=pd.read_parquet(source/"units.parquet").sort_values("local_series_index")
    vv=np.load(SOURCE/"master_values.npy",mmap_mode="r")[units.series_index.to_numpy()]
    vh=np.load(source/"vh_values.npy",mmap_mode="r")
    assert vv.shape==vh.shape==(26890,126)
    raw_difference=np.zeros(vv.shape,dtype=bool)
    raw_joint=np.zeros(vv.shape,dtype=bool)
    hashes={str(p):digest(p) for p in [source/"units.parquet",source/"vh_values.npy",SOURCE/"master_values.npy"]}
    for track,group in units.groupby("track_key",sort=True):
        native,raw,valid,nobs,dates,_=read_track(track)
        rows=group.cache_index.to_numpy(dtype=int);pos=group.local_series_index.to_numpy(dtype=int)
        for field in ("pixel_track_id","native_row","native_col"):
            np.testing.assert_array_equal(native.loc[rows,field],group[field])
        joint=(nobs[rows]>0)&np.isfinite(raw[rows]).all(axis=1)
        np.testing.assert_array_equal(valid[rows],joint)
        np.testing.assert_array_equal(np.where(joint,raw[rows,0],np.nan),vv[pos])
        np.testing.assert_array_equal(np.where(joint,raw[rows,1],np.nan),vh[pos])
        raw_difference[pos]=np.isfinite(raw[rows,0])!=np.isfinite(raw[rows,1])
        raw_joint[pos]=joint
        for path in (CACHE/f"{track}.npz",CACHE/f"{track}.pixels.parquet"):
            hashes[str(path)]=digest(path)
    actual_difference=np.isfinite(vv)!=np.isfinite(vh)
    eligibility_difference=eligibility_matrix("full",vv)!=eligibility_matrix("full",vh)
    calendars=pd.period_range("2016-01","2026-06",freq="M").astype(str)
    records={}
    for name,mask in [("raw_band_finite",raw_difference),("actual_joint_mask",actual_difference),
                      ("window_eligibility",eligibility_difference)]:
        bymonth=mask.sum(axis=0)
        records[name]=dict(compared_unit_months=int(mask.size),different_units=int(mask.any(axis=1).sum()),
            different_unit_months=int(mask.sum()),months_with_differences=calendars[bymonth>0].tolist(),
            differences_by_month=dict(zip(calendars,bymonth.astype(int).tolist())))
    assert not actual_difference.any() and not eligibility_difference.any()
    write_json(RUN/"fullchain/polarization_mask_identity.json",dict(status="passed",comparisons=records,
        units=26890,calendar_months=126,input_sha256=hashes,joint_valid_unit_months=int(raw_joint.sum()),
        actual_values_equal_to_frozen_sources=True,reuse_eligibility_permitted=True,
        meaning="low signal-to-noise is not a finite-value mask; no new masking rule"))
    print({name:row["different_unit_months"] for name,row in records.items()},flush=True)


if __name__=="__main__":main()
