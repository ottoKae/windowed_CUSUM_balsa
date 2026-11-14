"""P3: reference-aligned, event-weighted VV/VH descriptions on fixed native cells."""
import warnings
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from run import ROOT,RUN,SOURCE,digest,write_json


def event_signature(values,index):
    """Return paired-pixel changes and centred curves; absent calendar slots stay absent."""
    pre=values[:,:,max(0,index-12):max(0,index-2)]
    post=values[:,:,index:index+3]
    joint=np.isfinite(values).all(axis=1)
    paired=(joint[:,max(0,index-12):max(0,index-2)].sum(axis=1)>=6)&(joint[:,index:index+3].sum(axis=1)>=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore",RuntimeWarning)
        background=np.nanmedian(pre,axis=2)
        delta=np.nanmedian(post,axis=2)-background
        delta[~paired]=np.nan
        curves=np.full((values.shape[0],2,25),np.nan)
        for j,offset in enumerate(range(-12,13)):
            k=index+offset
            if 0<=k<values.shape[2]:curves[:,:,j]=values[:,:,k]-background
        curves[~paired]=np.nan
    return delta,curves,paired


def main():
    out=RUN/"signatures";out.mkdir(exist_ok=True)
    polar=ROOT/"runs/validation/balsa_results_ablation_20261101_v1/polarization"
    paths=[ROOT/"runs/validation/balsa_sts_range_evaluation_20261102_v1/reference_semantics.csv",
        ROOT/"runs/validation/balsa_firstpost_result_analysis_20261030_v1/reference_base_analysis.csv",
        polar/"units.parquet",polar/"vh_values.npy",SOURCE/"master_values.npy",SOURCE/"validation_membership.parquet"]
    refs=pd.read_csv(paths[0],keep_default_na=False)
    base=pd.read_csv(paths[1],keep_default_na=False)
    refs=refs.merge(base[["event_id","block_id"]],on="event_id",validate="one_to_one")
    assert len(refs)==1049 and refs.sample_id.nunique()==459 and refs.block_id.ne("").all()
    assert refs.groupby("sample_id").block_id.nunique().eq(1).all()
    units=pd.read_parquet(paths[2]).sort_values("local_series_index")
    members=pd.read_parquet(paths[5]).merge(units[["unit_id","local_series_index"]],on="unit_id",validate="one_to_one")
    vv=np.load(paths[4],mmap_mode="r")[units.series_index.to_numpy()]
    vh=np.load(paths[3],mmap_mode="r")
    np.testing.assert_array_equal(np.isfinite(vv),np.isfinite(vh))
    values=np.stack([vv,vh],axis=1)
    positions=members.groupby("sample_id").local_series_index.apply(list)
    months=pd.period_range("2016-01","2026-06",freq="M")
    coverage,changes,aligned=[],[],[]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore",RuntimeWarning)
        for r in refs.itertuples():
            ix=positions[r.sample_id];n=len(ix);k=months.get_loc(r.reference_month)
            delta,curves,paired=event_signature(values[ix],k)
            qualified=2*paired.sum()>=n
            coverage.append(dict(event_id=r.event_id,sample_id=r.sample_id,block_id=r.block_id,
                reference_month=r.reference_month,date_basis=r.date_basis,native_N=n,paired_pixels=int(paired.sum()),
                paired_fraction=float(paired.mean()),eligible=qualified,
                exclusion_reason="" if qualified else "paired_pixel_coverage_below_half",
                full_pre_calendar=k>=12,full_post_calendar=k+2<len(months)))
            for b,band in enumerate(("VV","VH")):
                changes.append(dict(event_id=r.event_id,sample_id=r.sample_id,block_id=r.block_id,band=band,
                    eligible=qualified,median_delta_db=float(np.nanmedian(delta[:,b])) if qualified else np.nan,
                    negative_pixel_fraction=float((delta[paired,b]<0).mean()) if qualified else np.nan))
                for j,offset in enumerate(range(-12,13)):
                    finite=np.isfinite(curves[:,b,j]);support=int(finite.sum())
                    aligned.append(dict(event_id=r.event_id,sample_id=r.sample_id,block_id=r.block_id,band=band,
                        offset=offset,native_N=n,contributing_pixels=support,coverage=support/n,
                        centred_db=float(np.nanmedian(curves[:,b,j])) if qualified and 2*support>=n else np.nan))
    coverage=pd.DataFrame(coverage);changes=pd.DataFrame(changes);aligned=pd.DataFrame(aligned)
    coverage.to_csv(out/"event_coverage.csv",index=False)
    changes.to_parquet(out/"event_changes.parquet",index=False)
    aligned.to_parquet(out/"aligned_months.parquet",index=False)
    blocks=sorted(refs.block_id.unique());rng=np.random.default_rng(20261105)
    draws=rng.integers(0,len(blocks),size=(5000,len(blocks)))
    np.save(out/"spatial_block_indices.npy",draws)
    write_json(out/"spatial_block_axes.json",dict(block_ids=blocks,seed=20261105,replicates=5000))
    event_order=refs.event_id.tolist()
    block_events=[np.flatnonzero(refs.block_id.to_numpy()==b) for b in blocks]
    resamples=[np.concatenate([block_events[b] for b in draw]) for draw in draws]
    summaries=[];curve_rows=[]
    for band in ("VV","VH"):
        x=changes.loc[changes.band.eq(band)].set_index("event_id").reindex(event_order).median_delta_db.to_numpy()
        eligible=np.isfinite(x);good=x[eligible]
        boot=np.array([[np.nanmedian(x[ix]),np.mean(x[ix][np.isfinite(x[ix])]<0)] for ix in resamples])
        summaries.append(dict(band=band,n_references=1049,n_eligible_events=len(good),n_blocks=len(blocks),
            median_delta_db=np.median(good),q1=np.quantile(good,.25),q3=np.quantile(good,.75),
            negative_event_fraction=np.mean(good<0),median_ci_low=np.quantile(boot[:,0],.025),
            median_ci_high=np.quantile(boot[:,0],.975),negative_fraction_ci_low=np.quantile(boot[:,1],.025),
            negative_fraction_ci_high=np.quantile(boot[:,1],.975)))
        curves=aligned.loc[aligned.band.eq(band)].pivot(index="event_id",columns="offset",values="centred_db").reindex(event_order).to_numpy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore",RuntimeWarning)
            boot_curve=np.array([np.nanmedian(curves[ix],axis=0) for ix in resamples])
        for j,offset in enumerate(range(-12,13)):
            z=curves[:,j];valid=z[np.isfinite(z)]
            curve_rows.append(dict(band=band,offset=offset,n_events=len(valid),median=np.median(valid),
                q1=np.quantile(valid,.25),q3=np.quantile(valid,.75),ci_low=np.nanquantile(boot_curve[:,j],.025),
                ci_high=np.nanquantile(boot_curve[:,j],.975)))
    summary=pd.DataFrame(summaries);summary.to_csv(out/"summary.csv",index=False)
    curves=pd.DataFrame(curve_rows);curves.to_csv(out/"figure_source_aligned.csv",index=False)
    plt.rcParams.update({"font.family":"serif","font.serif":["Times New Roman"],"font.size":9,"pdf.fonttype":42,"svg.fonttype":"none"})
    fig,axes=plt.subplots(1,2,figsize=(7.2,2.8),layout="constrained")
    for band,color,style in [("VV","#0072B2","-"),("VH","#D55E00","--")]:
        f=curves.loc[curves.band.eq(band)]
        axes[0].plot(f.offset,f["median"],color=color,ls=style,label=band)
        axes[0].fill_between(f.offset,f.ci_low,f.ci_high,color=color,alpha=.15)
        axes[1].plot(f.offset,f.n_events,color=color,ls=style,label=band)
    axes[0].axhline(0,color=".6",lw=.6);axes[0].axvline(0,color=".6",lw=.6)
    axes[0].set(xlabel="Months relative to catalogue date",ylabel="Background-centred backscatter (dB)")
    axes[1].set(xlabel="Months relative to catalogue date",ylabel="Contributing reference events",ylim=(0,1100))
    for ax in axes:ax.legend(frameon=False);ax.spines[["top","right"]].set_visible(False)
    for ext in ("pdf","png","svg"):fig.savefig(out/f"reference_aligned_signatures.{ext}",dpi=300)
    plt.close(fig)
    write_json(out/"receipt.json",dict(status="completed",n_references=1049,n_eligible=int(coverage.eligible.sum()),
        n_blocks=len(blocks),bootstrap_replicates=5000,statistics="event_equal_weight; whole_spatial_block_resampling",
        curve_policy="paired pixels with >=6 pre and >=2 post observations; event and each plotted month >=50% original members",
        input_sha256={str(p):digest(p) for p in paths},code_sha256=digest(__file__),detector_selection=False))
    print(summary.to_string(index=False),flush=True)


if __name__=="__main__":main()
