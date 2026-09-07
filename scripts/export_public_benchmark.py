#!/usr/bin/env python3
"""Copy only explicit aggregate benchmark tables; never copy event-level data."""
import json
from pathlib import Path
import shutil
import sys
import itertools
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from lr_cusum.capsule import digest, write_json


def main():
    source = ROOT/"runs/cusum_nrt_ccdc_20260905_v1"
    output = ROOT/"results/nrt_ccdc"
    output.mkdir(parents=True,exist_ok=True)
    files = {
        "method_metrics.csv":"method_metrics.csv",
        "paired_contrasts.csv":"paired_contrasts.csv",
        "diagnostics/method_metrics.csv":"direction_control_metrics.csv",
        "diagnostics/paired_contrasts.csv":"direction_control_contrasts.csv",
    }
    receipt = []
    for original,name in files.items():
        src,dest = source/original,output/name
        if dest.exists() and digest(dest)!=digest(src):
            raise ValueError("existing aggregate differs; create a new result version")
        shutil.copyfile(src,dest)
        receipt.append({"file":name,"source_run":"cusum_nrt_ccdc_20260905_v1",
                        "source_relative_path":original,"sha256":digest(src)})
    cfg=json.loads((ROOT/"configs/cusum_nrt_ccdc_comparison_v1.json").read_text())
    statuses=[]
    for chi,obs,lam in itertools.product(cfg["ccdc_chi"],cfg["ccdc_observations"],cfg["ccdc_lambda"]):
        name=f"ccdc_p{round(chi*100)}_n{obs}_l{lam:g}"
        complete=(source/"ground"/(name+".parquet")).is_file()
        failed=(source/"ccdc_failures"/(name+".json")).is_file()
        if complete==failed:
            raise ValueError("CCDC configuration is either unaccounted for or contradictory")
        statuses.append({"parameter_id":name,"status":"complete" if complete else "failed_not_zero_events"})
    pd.DataFrame(statuses).to_csv(output/"ccdc_grid_status.csv",index=False)
    write_json(output/"SOURCES.json",receipt)
    print(json.dumps(receipt,indent=2))


if __name__=="__main__":
    main()
