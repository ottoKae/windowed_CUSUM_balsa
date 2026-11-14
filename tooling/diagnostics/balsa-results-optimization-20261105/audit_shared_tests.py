"""Same scientific parent input must have identical STP tests across variants/batches."""
import json
import pandas as pd
from fullchain_ablation import OUT,SOURCE,BATCHES,signature,stars
from run import write_json


def main():
    units=pd.read_parquet(SOURCE/"master_units.parquet").set_index("unit_id",drop=False)
    base=OUT/"variants/full"
    reports=pd.read_parquet(base/"octa/reports.parquet")
    members=pd.read_parquet(base/"octa/source_members.parquet")
    groups=dict(tuple(members.groupby("component_id",sort=False)))
    mapping=pd.DataFrame([dict(variant_id="full",component_id=r.component_id,
        signature=signature(r,groups[r.component_id],units,stars.Config())) for r in reports.itertuples()])
    maps=[mapping]
    for q in BATCHES:
        maps.append(pd.read_parquet(OUT/f"calibration/20261102/{q}/variant_parent_signatures.parquet")[["variant_id","component_id","signature"]])
    mapping=pd.concat(maps,ignore_index=True)
    all_states=[]
    for v,g in mapping.groupby("variant_id"):
        states=pd.read_parquet(OUT/"variants"/v/"cstr/stands.parquet")
        states=states.merge(g,on="component_id",validate="one_to_one")
        all_states.append(states)
    states=pd.concat(all_states,ignore_index=True)
    fields=["destination_type","reason","admissible","parent_pixels","response_pixels","tolerance_months",
        "observed_join_count","null_join_mean","null_join_sd","max_standardized_join","p_value","zone_count","round_count"]
    differences=states.groupby("signature")[fields].nunique(dropna=False)
    assert not differences.gt(1).any().any(),differences.loc[differences.gt(1).any(axis=1)].to_dict()
    counts=states.groupby("signature").agg(n_occurrences=("component_id","size"),n_variants=("variant_id","nunique"))
    counts.to_csv(OUT/"shared_scientific_test_counts.csv")
    write_json(OUT/"shared_scientific_tests.json",dict(status="passed",compared_fields=fields,
        unique_signatures=len(counts),parent_occurrences=len(states),reused_across_variants=int(counts.n_variants.gt(1).sum()),
        exact_equal=True,q_values_excluded="variant-specific test family; not decision criterion"))
    print(f"Shared STP inputs exact across {len(states)} parents / {len(counts)} signatures",flush=True)


if __name__=="__main__":main()
