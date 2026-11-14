"""Render a vector mechanism figure from separately computed schematic evidence."""
from pathlib import Path
import hashlib
import json
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_hex
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "runs/figures/sts_cusum_tda_stp_technical_workflow_20261106_v1"
PDFTOPPM = "pdftoppm"
W, H = 180, 120
INK = "#263D4C"
PURPLE = "#6A51A3"
TEAL = "#24888C"
PALE = "#EFF3F6"
GRID = "#CEDAE3"
QUIET = "#CED9E2"
MUTED = "#738797"
GROUPS = ["#9D87BE", "#78B8B0", "#D7A56D", "#7DA2C8"]
ANCHORS = ["#54C7C9", "#7FD4C8", "#A9DEBF", "#D5E7C5", "#F2E8B3",
           "#F7C98D", "#F79E69", "#F56C4A", "#EF3B2C"]


def load_evidence():
    e=json.loads((OUT/"technical_evidence.json").read_text(encoding="utf-8"))
    assert all(e["verification"].values())
    series=pd.read_csv(OUT/"schematic_inputs.csv")
    window=pd.read_csv(OUT/"cusum_window_evidence.csv").rename(columns={"vv_db":"VV_dB"})
    events=pd.DataFrame(e["events"]).rename(columns={"change_index":"month_index"})
    monthly=pd.DataFrame(e["tda"]["monthly"])
    first=pd.DataFrame(e["stp"]["first_dates"]).rename(columns={
        "first_month":"month_index","shuffled_first_month":"shuffled_month_index"})
    zones=pd.DataFrame(e["stp"]["provisional_zones"]).rename(columns={
        "median_month_index":"centre_month_index"})
    zones["zone_id"]=zones.zone_id.str.rsplit("z",n=1).str[-1].astype(int)
    excerpt=monthly.loc[monthly.month_index.between(60,77)].reset_index(drop=True)
    selected=e["grid"]["selected_pixels"]
    return dict(raw=e,selected=selected,series=series,window=window,events=events,
        peak=e["cusum"]["peak_global_index"],candidate=e["cusum"]["candidate_global_index"],
        candidate_label=e["cusum"]["candidate_month"],excerpt=excerpt,
        excerpt_year_label="Jan 2021–Jun 2022 · excerpt",parent_pixels=first.pixel_id.tolist(),
        first_dates=first,zones=zones,stp_test=e["stp"]["test"],
        outcome=f"{len(e['stp']['final_groups'])} groups · 1 parent")


def axis(fig, x, y, w, h):
    return fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H])


def text(fig, x, y, value, size=7, color=INK, ha="left", weight="normal", **kwargs):
    return fig.text(x / W, 1 - y / H, value, fontsize=size, color=color,
                    ha=ha, va="center", weight=weight, **kwargs)


def arrow(fig, x1, y1, x2, y2, color=INK, style="-|>", lw=.75, dashed=False):
    patch = FancyArrowPatch((x1/W, 1-y1/H), (x2/W, 1-y2/H),
        transform=fig.transFigure, arrowstyle=style, mutation_scale=7,
        color=color, lw=lw, shrinkA=0, shrinkB=0,
        linestyle=(0, (2, 2)) if dashed else "-")
    fig.add_artist(patch)
    return patch


def segment(fig, x1, y1, x2, y2, color=GRID, lw=.55, **kwargs):
    line = Line2D([x1/W, x2/W], [1-y1/H, 1-y2/H], transform=fig.transFigure,
                  color=color, lw=lw, **kwargs)
    fig.add_artist(line)


def style_plot(ax, xticks=(), yticks=()):
    ax.spines[["top", "right"]].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(MUTED)
        ax.spines[side].set_linewidth(.5)
    ax.tick_params(labelsize=6.5, length=1.7, pad=1.5, width=.45, colors=INK)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)


def spatial_grid(fig, box, fills, selected=(), labels=None, borders=None, gid="grid"):
    ax = axis(fig, *box)
    ax.set(xlim=(-.5, 6.5), ylim=(4.5, -.5), aspect="equal")
    ax.set_axis_off()
    for i in range(35):
        row, col = divmod(i, 7)
        selected_cell = i+1 in selected
        cell = Rectangle((col-.5, row-.5), 1, 1, fc=fills[i],
                         ec=PURPLE if selected_cell else "white",
                         lw=1 if selected_cell else .45)
        cell.set_gid(f"{gid}_pixel_{i+1}")
        ax.add_patch(cell)
        if labels is not None and labels[i]:
            ax.text(col, row, str(labels[i]), ha="center", va="center",
                    fontsize=6.5, color=INK, weight="bold" if selected_cell else "normal")
        if borders is not None:
            here = borders[i]
            if here is None:
                continue
            for dr, dc, endpoints in [(-1,0,[(col-.5,row-.5),(col+.5,row-.5)]),
                    (1,0,[(col-.5,row+.5),(col+.5,row+.5)]),
                    (0,-1,[(col-.5,row-.5),(col-.5,row+.5)]),
                    (0,1,[(col+.5,row-.5),(col+.5,row+.5)])]:
                r, c = row+dr, col+dc
                other = borders[r*7+c] if 0 <= r < 5 and 0 <= c < 7 else None
                if other != here:
                    ax.plot(*zip(*endpoints), color=INK, lw=.75)
    ax.add_patch(Rectangle((-.5,-.5),7,5,fc="none",ec=INK,lw=.65))
    return ax


def states(ax, values, y=0, height=.75, gid="state"):
    for i, value in enumerate(values):
        active = value == "active"
        unknown = value == "unknown"
        cell = Rectangle((i-.45,y),.9,height,
                         fc="white" if unknown else PURPLE if active else QUIET,
                         ec=MUTED if unknown else "white", lw=.4,
                         hatch="///" if unknown else None)
        cell.set_gid(f"{gid}_{i}")
        ax.add_patch(cell)


def draw_sts(fig, d):
    text(fig, 18.5, 13, "Fixed locations", ha="center")
    selected = d["selected"]
    labels = [str(i+1) if i+1 in selected else "" for i in range(35)]
    fills = ["#CBBBE0" if i+1 in selected else "#E1EAF0" for i in range(35)]
    spatial_grid(fig, (4.5, 18, 28, 20), fills, selected, labels, gid="sts")
    text(fig,18.5,41,r"$\mathcal{U}_s$ · 35 fixed pixels",ha="center")
    arrow(fig,18.5,43,18.5,47,color=PURPLE)
    for k, pixel in enumerate(selected):
        y = 50 + k*16
        frame = d["series"].loc[d["series"].pixel_id.eq(pixel)]
        ax = axis(fig,7,y,25,11)
        ax.plot(frame.month_index,frame.VV_dB,color=PURPLE,lw=.7, label="VV")[0].set_gid(f"sts_trace_{pixel}_VV")
        ax.plot(frame.month_index,frame.VH_dB,color=TEAL,lw=.65,ls="--", label="VH")[0].set_gid(f"sts_trace_{pixel}_VH")
        ax.set(xlim=(0,125),ylim=(-25,0))
        style_plot(ax,[0,125] if k==2 else [],[])
        if k==2:
            ax.set_xticklabels(["2016","2026"])
        text(fig,7,y-2.2,f"Pixel {pixel}",size=6.5,color=PURPLE)
    text(fig,18.5,99,"Monthly backscatter (dB)",size=6.5,ha="center")
    segment(fig,8,103,12,103,color=PURPLE,lw=.9)
    text(fig,13,103,"VV",size=6.5)
    segment(fig,22,103,26,103,color=TEAL,lw=.8,ls="--")
    text(fig,27,103,"VH",size=6.5)


def draw_cusum(fig, d):
    text(fig,60.5,13,"Renew the local reference",ha="center")
    pixel = d["selected"][0]
    curve = d["series"].loc[d["series"].pixel_id.eq(pixel)]
    top = axis(fig,42,18,37,11)
    top.plot(curve.month_index,curve.VV_dB,color=PURPLE,lw=.65)
    top.set(xlim=(0,125),ylim=(-19,-5))
    style_plot(top,[],[])
    text(fig,42,16,f"Pixel {pixel} · VV",size=6.5,color=PURPLE)
    window = d["window"]
    win_start = int(window.month_index.min())
    peak = d["peak"]
    candidate = d["candidate"]
    decision_start = int(pd.Period(d["candidate_label"][:4]+"-01",freq="M").ordinal-pd.Period("2016-01",freq="M").ordinal)
    windows = axis(fig,42,30.5,37,9)
    windows.set(xlim=(0,125),ylim=(-.5,2.5))
    windows.set_axis_off()
    for i,start in enumerate([win_start-12,win_start,win_start+12]):
        y=2-i
        windows.add_patch(Rectangle((start,y-.29),36,.58,fc="#F0EAF6",ec="#B8A5CE",lw=.5))
        windows.add_patch(Rectangle((start+12,y-.29),12,.58,fc="#B9A4D4",ec="none"))
    text(fig,60.5,41,"W36 · D12 · step 12 months",size=6.5,ha="center")
    arrow(fig,60.5,43,60.5,46,color=PURPLE)

    obs=axis(fig,43,49,35,12)
    x=window.month_index.to_numpy()
    z=window.VV_dB.to_numpy()
    avg=float(window.window_mean.iloc[0])
    obs.axvspan(decision_start,decision_start+11,color="#F2EDF7",zorder=0)
    obs.axhline(avg,color=MUTED,lw=.6,ls="--")
    obs.vlines(x,avg,z,color="#C3B4D7",lw=.5)
    # Keep the missing calendar month visible even though CUSUM uses finite samples.
    complete=np.arange(x.min(),x.max()+1)
    full_values=window.set_index("month_index").VV_dB.reindex(complete)
    obs.plot(complete,full_values,color=PURPLE,lw=.8,marker="o",ms=1.4)
    for missing in complete[full_values.isna()]:
        obs.axvspan(missing-.45,missing+.45,fc="white",ec=GRID,hatch="///",lw=0,zorder=0)
    obs.set(xlim=(x.min(),x.max()),ylim=(z.min()-1,z.max()+1))
    style_plot(obs,[],[])
    obs.text(.02,.06,r"$\bar z$",transform=obs.transAxes,fontsize=7,color=MUTED)
    text(fig,43,46.7,"Centre within the window",size=6.5)

    cum=axis(fig,43,67,35,16)
    sums=window.cumulative.to_numpy()
    cum.axvspan(decision_start,decision_start+11,color="#F2EDF7",zorder=0)
    cum.axhline(0,color=GRID,lw=.5)
    cum.plot(x,sums,color=PURPLE,lw=1,marker="o",ms=1.4)[0].set_gid("cusum_computed_cumulative")
    pi=int(np.flatnonzero(x==peak)[0]); ci=int(np.flatnonzero(x==candidate)[0])
    cum.scatter([peak],[sums[pi]],s=19,fc="white",ec=PURPLE,lw=.9,zorder=5)
    cum.scatter([candidate],[sums[ci]],s=15,marker="D",fc=INK,ec="none",zorder=5)
    cum.axvline(candidate,color=INK,lw=.6,ls=(0,(2,2)))
    cum.set(xlim=(x.min(),x.max()),ylim=(min(0,sums.min())-4,sums.max()+7))
    style_plot(cum,[x[0],x[-1]],[])
    cum.set_xticklabels([window.month.iloc[0],window.month.iloc[-1]])
    cum.annotate("",xy=(.98,sums.max()),xytext=(.98,sums.min()),
                 xycoords=("axes fraction","data"),arrowprops=dict(arrowstyle="|-|",lw=.55,color=INK))
    cum.text(.94,.47,"Q",transform=cum.transAxes,ha="right",fontsize=7)
    text(fig,60.5,64,r"$C_k=C_{k-1}+(z_k-\bar z)$",size=7,ha="center")
    text(fig,60.5,88,"○ peak   ◆ next valid month",size=6.5,ha="center")
    text(fig,60.5,93,r"post < pre    $\hat\tau\in D_j$",size=7,ha="center")
    text(fig,60.5,97,r"$p_{perm}\leq0.025$",size=7,ha="center",color=PURPLE)
    accepted=axis(fig,43,100,35,4)
    accepted.set(xlim=(0,125),ylim=(0,1))
    accepted.set_axis_off()
    accepted.plot([0,125],[.35,.35],color=GRID,lw=.7)
    months=d["events"].loc[d["events"].pixel_id.eq(pixel),"month_index"].to_numpy()
    accepted.scatter(months,np.full(len(months),.35),s=13,marker="D",c=PURPLE)
    text(fig,60.5,106,"Retain all accepted months",size=6.5,ha="center")


def draw_tda(fig,d):
    text(fig,107.5,13,"Connect accepted responses",ha="center")
    monthly=d["excerpt"]
    ids=d["selected"]
    indices=monthly.month_index.to_numpy()
    lookup={int(m):i for i,m in enumerate(indices)}
    mat=axis(fig,92,18,33,31)
    mat.set(xlim=(-.5,len(indices)-.5),ylim=(34.5,-.5))
    for row in range(35):
        mat.axhline(row-.5,color="#E9EEF2",lw=.35,zorder=0)
    for col in range(len(indices)+1):
        mat.axvline(col-.5,color="#E9EEF2",lw=.35,zorder=0)
    for pid in ids:
        mat.axhspan(pid-1.5,pid-.5,color="#F2EDF7",zorder=0)
    for record in d["events"].itertuples():
        if record.month_index in lookup:
            mat.plot(lookup[record.month_index],record.pixel_id-1,"o",ms=1.8,color=PURPLE)[0].set_gid(
                f"tda_response_{record.pixel_id}_{record.month_index}")
    mat.spines[:].set_visible(False)
    mat.set_yticks([i-1 for i in ids],labels=[str(i) for i in ids])
    tick=[i for i,m in enumerate(monthly.month) if m.endswith(("-01","-07"))]
    mat.set_xticks(tick,labels=[pd.Period(monthly.month.iloc[i]).strftime("%b") for i in tick])
    mat.tick_params(length=0,labelsize=6.5,pad=2)
    mat.set_ylabel("Pixel",fontsize=6.5,labelpad=2)
    text(fig,108.5,53,d["excerpt_year_label"],size=6.5,ha="center")
    status=axis(fig,92,55,33,2.7)
    status.set(xlim=(-.5,len(indices)-.5),ylim=(0,1))
    status.set_axis_off()
    states(status,monthly.state.tolist(),height=.95,gid="observed_state")
    for l,r in [(6,8),(8,10)]:
        status.plot([l,l,r,r],[1.05,1.3,1.3,1.05],color=PURPLE,lw=.65,clip_on=False)
    legend=axis(fig,90,59,2,2)
    legend.add_patch(Rectangle((0,0),1,1,fc=QUIET,ec="none"));legend.set_axis_off()
    text(fig,93,59.4,"No detection",size=6.5)
    text(fig,93,62.2,"≥18/35 eligible members",size=6.5)
    text(fig,107.5,65,"Gap rules",size=7,ha="center",weight="bold")
    rules=[(["active","observed_no_candidate","active"],"join"),
           (["active","observed_no_candidate","observed_no_candidate","active"],"separate"),
           (["active","unknown","active"],"unresolved")]
    for i,(values,label) in enumerate(rules):
        y=68+i*5
        a=axis(fig,91,y,17,3)
        a.set(xlim=(-.5,3.5),ylim=(0,1))
        a.set_axis_off()
        states(a,values,height=.9,gid=f"rule_{i}")
        arrow(fig,109,y+1.3,112,y+1.3,lw=.55)
        text(fig,113,y+1.3,label,size=6.5)
    # Rule examples share the state encoding, but are not extra observations.
    text(fig,107.5,84,"Purple: active · hatched: unknown",size=6.5,ha="center",color=MUTED)
    support=d["parent_pixels"]
    fills=["#B9A4D4" if i+1 in support else PALE for i in range(35)]
    spatial_grid(fig,(90,89,14,10),fills,gid="tda_support")
    text(fig,107,91,"Distinct pixels",size=6.5)
    text(fig,107,96,r"$|\mathcal{U}_{report}|/N_s\geq0.20$",size=7)
    text(fig,107.5,103,"Parent P1 · Jul–Nov 2021",size=6.5,weight="bold",ha="center",color=PURPLE)
    segment(fig,94,106,122,106,color=PURPLE,lw=2.5)


def draw_stp(fig,d):
    text(fig,155,13,"Test within-report timing",ha="center")
    first=d["first_dates"]
    palette=LinearSegmentedColormap.from_list("harvest_month",ANCHORS,N=4096)
    dates=sorted(first.month_index.unique())
    def color(m):
        return to_hex(palette((int(m)-min(dates))/(max(dates)-min(dates))))
    first_lookup=first.set_index("pixel_id").month_index.to_dict()
    shuffle_lookup=first.set_index("pixel_id").shuffled_month_index.to_dict()
    obs=[color(first_lookup[i+1]) if i+1 in first_lookup else PALE for i in range(35)]
    shuffled=[color(shuffle_lookup[i+1]) if i+1 in shuffle_lookup else PALE for i in range(35)]
    text(fig,143,17,"P1 first month",size=6.5,ha="center")
    text(fig,167,17,"Shuffled dates",size=6.5,ha="center")
    original=spatial_grid(fig,(134,20,18,12.86),obs,gid="first_dates")
    original.plot([0,1],[0,0],color=INK,lw=.7,marker="o",ms=1.5)
    spatial_grid(fig,(158,20,18,12.86),shuffled,gid="shuffled_dates")
    arrow(fig,153,26.4,157,26.4,lw=.6)
    # Calendar colours span the illustrated parent's dates, and never encode significance.
    for i,m in enumerate(dates):
        x=139+i*(32/max(1,len(dates)-1))
        a=axis(fig,x-2,35,4,1.5)
        a.add_patch(Rectangle((0,0),1,1,fc=color(m),ec="none"));a.set_axis_off()
        label=str(pd.Period("2016-01",freq="M")+int(m))
        text(fig,x,39,pd.Period(label).strftime("%b"),size=6.5,ha="center")
    text(fig,155,43,r"Shared edge: $|\tau_u-\tau_v|\leq\delta$",size=6.5,ha="center")
    text(fig,155,47,r"$\delta=0,1,2$ · 999 shuffles",size=6.5,ha="center")
    test=d["stp_test"]
    p=test.get("p_value")
    p_label="undefined" if p is None or not np.isfinite(p) else f"{p:.3f}"
    text(fig,155,51,"Test cross-zone joins",size=6.5,ha="center")
    text(fig,155,55,rf"$T_{{obs}}<0;\ p_{{sp}}={p_label}\leq0.05$",size=6.5,ha="center",color=PURPLE)
    arrow(fig,155,57,155,59,color=PURPLE)
    text(fig,155,62,r"$\delta^*=0$: merge the small zone",size=6.5,ha="center",weight="bold")
    zone_lookup=first.set_index("pixel_id").merged_zone.to_dict()
    fills=[GROUPS[int(zone_lookup[i+1])%len(GROUPS)] if i+1 in zone_lookup else PALE for i in range(35)]
    borders=[zone_lookup.get(i+1) for i in range(35)]
    spatial_grid(fig,(134,67,18,12.86),fills,borders=borders,gid="merged_zones")
    zones=d["zones"]
    arrow(fig,153,72.5,157.5,72.5,lw=.6)
    text(fig,167,64.8,"Zone centres",size=6.5,ha="center")
    zax=axis(fig,159,67,16,12.86)
    zax.set(xlim=(min(dates)-.8,max(dates)+.8),ylim=(-.5,len(zones)-.3))
    style_plot(zax,dates,[])
    zax.spines["left"].set_visible(False)
    zax.set_xticklabels([pd.Period(str(pd.Period("2016-01",freq="M")+int(m))).strftime("%b")[0] for m in dates])
    for i,z in zones.sort_values("centre_month_index").iterrows():
        y=list(zones.index).index(i)
        zax.scatter([z.centre_month_index],[y],s=90,color=GROUPS[int(z.zone_id)%len(GROUPS)],ec=INK,lw=.45)
        zax.text(z.centre_month_index,y,str(z.response_pixels),ha="center",va="center",fontsize=6.5)
    text(fig,143,83,"3 → 2 zones",size=6.5,ha="center")
    text(fig,167,84,"Counts = pixels",size=6.5,ha="center",color=MUTED)
    text(fig,155,89,"Centre gap >2 mo; ≥2 groups",size=6.5,ha="center")
    text(fig,155,93,"Each ≥20% of fixed members",size=6.5,ha="center")
    arrow(fig,155,95,143,98,color=PURPLE,lw=.6)
    arrow(fig,155,95,167,98,color=MUTED,lw=.6,dashed=True)
    text(fig,142,100,"Retain 2 groups",size=6.5,weight="bold",color=PURPLE,ha="center")
    text(fig,169,100,"Otherwise",size=6.5,color=MUTED,ha="center")
    text(fig,142,104,"17/35 + 18/35",size=6.5,color=PURPLE,ha="center")
    text(fig,169,104,"Unsplit parent",size=6.5,color=MUTED,ha="center")


def main():
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":7,"text.color":INK,
                         "pdf.fonttype":42,"svg.fonttype":"none","mathtext.fontset":"dejavusans"})
    d=load_evidence()
    fig=plt.figure(figsize=(W/25.4,H/25.4),facecolor="white")
    for center,label in [(18.5,"(a) STS"),(60.5,"(b) Windowed CUSUM"),
                         (107.5,"(c) TDA"),(155,"(d) STP")]:
        text(fig,center,6,label,size=9,ha="center",weight="bold")
    for x1,x2 in [(34,39),(83,88),(128,133)]:
        arrow(fig,x1,6,x2,6,lw=.8)
    for x in [36.5,85.5,130.5]:
        segment(fig,x,16,x,107,color="#DFE5EA",lw=.5)
    draw_sts(fig,d)
    draw_cusum(fig,d)
    draw_tda(fig,d)
    draw_stp(fig,d)
    segment(fig,4,110,176,110,color=GRID,lw=.65)
    text(fig,4,114,"Stand history",size=7,weight="bold")
    hist=axis(fig,39,113,55,3)
    hist.set(xlim=(62,108),ylim=(0,1));hist.set_axis_off()
    reports=d["raw"]["tda"]["reports"]
    centre=[]
    for p in reports:
        month=lambda key: pd.Period(p[key],freq="M").ordinal-pd.Period("2016-01",freq="M").ordinal
        lo,hi,mid=[month(k) for k in ["first_response_month","last_response_month","representative_month"]]
        centre.append(mid)
        hist.plot([lo,hi],[.62,.62],color="#B9A4D4",lw=3,solid_capstyle="butt")
        hist.scatter([mid],[.62],color=PURPLE,s=10,zorder=3)
        hist.text(mid,-.15,p["representative_month"],ha="center",va="top",fontsize=6.5)
    hist.plot(centre,[.62]*len(centre),color=MUTED,lw=.55,zorder=0)
    text(fig,69,112,"34 months",size=6.5,ha="center")
    text(fig,103,113.5,"2 parents; spatial groups do not add events",size=6.5)
    text(fig,176,118,"Calculated schematic example",size=6.5,color=MUTED,ha="right")
    for ext in ["pdf","svg","png"]:
        fig.savefig(OUT/f"technical_workflow.{ext}",dpi=350)
    plt.close(fig)
    for suffix,width in [("preview",2400),("180mm",1063)]:
        subprocess.run([str(PDFTOPPM),"-scale-to-x",str(width),"-scale-to-y","-1",
                        "-png","-singlefile",str(OUT/"technical_workflow.pdf"),
                        str(OUT/f"technical_workflow_{suffix}")],check=True)
    pdf=PdfReader(OUT/"technical_workflow.pdf")
    assert len(pdf.pages)==1 and len(pdf.pages[0].images)==0
    svg=(OUT/"technical_workflow.svg").read_text(encoding="utf-8")
    assert svg.count('id="sts_pixel_')==35 and svg.count('id="sts_trace_')==6
    assert svg.count('id="tda_response_')==35
    assert svg.count('id="first_dates_pixel_')==35 and svg.count('id="shuffled_dates_pixel_')==35
    assert svg.count('id="merged_zones_pixel_')==35 and '<image' not in svg
    sources=[Path(__file__),Path(__file__).with_name("prepare_technical_evidence.py")]
    sources += [OUT / f for f in ["technical_evidence.json","schematic_inputs.csv",
        "cusum_window_evidence.csv","accepted_pixel_responses.csv","tda_reports.csv",
        "stp_first_month_pixels.csv","stp_provisional_zones.csv"]]
    protected=d["raw"]["protected_sources"]
    preserved={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==digest for p,digest in protected.items()}
    assert all(preserved.values()),preserved
    manifest=dict(figure="technical_workflow",created_date="2026-11-06",
        canvas_mm=[W,H],backend="Python / Matplotlib; PDF and editable SVG",
        scientific_input_label="Calculated schematic example, not Balsa results",
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        protected_sources_unchanged=preserved,algorithm_checks=d["raw"]["verification"],
        plot_checks=dict(pdf_pages=1,pdf_raster_images=0,svg_raster_images=0,
            fixed_cells_per_map=35,selected_vv_vh_traces=6,tda_excerpt_nodes=35,
            total_accepted_nodes=len(d["events"]),font_range_pt=[6.5,9]),
        date_colour_mapping=dict(anchors=ANCHORS,range=["2021-07","2021-11"],
            scope="P1 first detection dates; NOT the national 2017–2025 calendar scale"),
        group_colour_mapping="Discrete group identity; purple and teal, not dates",
        preview_sizes_pixels=[2400,1063],existing_case_figures="Preserved; not replaced")
    (OUT/"figure_manifest.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(OUT/"technical_workflow.pdf")


if __name__=="__main__":
    main()
