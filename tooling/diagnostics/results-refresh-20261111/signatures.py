"""Recompute reference signatures and verify the published Fig. 5 inputs."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
RUN=ROOT/'runs/validation/results_refresh_20261111_v1'
CODE=ROOT/'tooling/diagnostics/balsa-results-optimization-20261105'
sys.path.insert(0,str(CODE))
spec=importlib.util.spec_from_file_location('refresh_native_signatures',CODE/'signatures.py')
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
mod.RUN=RUN
mod.main()
out=RUN/'signatures'
old=ROOT/'runs/validation/balsa_results_optimization_20261105_v1/signatures'
checks=[]
for name in ['event_coverage.csv','summary.csv','figure_source_aligned.csv']:
    pd.testing.assert_frame_equal(pd.read_csv(out/name),pd.read_csv(old/name),check_exact=False,rtol=0,atol=1e-14)
    checks.append(name)
for name in ['event_changes.parquet','aligned_months.parquet']:
    pd.testing.assert_frame_equal(pd.read_parquet(out/name),pd.read_parquet(old/name),check_exact=True)
    checks.append(name)
np.testing.assert_array_equal(np.load(out/'spatial_block_indices.npy'),np.load(old/'spatial_block_indices.npy'))
figure=ROOT/'runs/figures/fig05_native_signatures_20261109_v1'
data=figure/'data'
provenance=json.loads((data/'provenance.json').read_text(encoding='utf-8'))
for path,expected in provenance['input_sha256'].items():
    assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==expected,path
p3=pd.read_csv(out/'summary.csv').set_index('band')
table=pd.read_csv(data/'table_summary.csv').set_index('band')
np.testing.assert_allclose(p3.median_delta_db,table.paired_delta_median_db,rtol=0,atol=1e-14)
np.testing.assert_allclose(p3.median_ci_low,table.paired_delta_block_bootstrap_lower_db,rtol=0,atol=1e-14)
np.testing.assert_allclose(p3.median_ci_high,table.paired_delta_block_bootstrap_upper_db,rtol=0,atol=1e-14)
curve=pd.read_csv(data/'curve.csv').rename(columns={'relative_month':'offset','q25':'q1','q75':'q3','bootstrap_lower':'ci_low','bootstrap_upper':'ci_high'})
columns=['band','offset','n_events','median','q1','q3','ci_low','ci_high']
pd.testing.assert_frame_equal(pd.read_csv(out/'figure_source_aligned.csv')[columns].sort_values(['band','offset']).reset_index(drop=True),curve[columns].sort_values(['band','offset']).reset_index(drop=True),check_exact=False,rtol=0,atol=1e-14)
events=pd.read_csv(data/'events.csv')
absolute=events.groupby('band').agg(n_events=('event_id','nunique'),pre_median_db=('pre_db','median'),post_median_db=('post_db','median')).reset_index()
assert absolute.n_events.eq(978).all()
absolute.to_csv(out/'absolute_backscatter_summary.csv',index=False)
acceptance=json.loads((figure/'acceptance.json').read_text(encoding='utf-8'))
pdf=ROOT/'manuscript_JAG/figures/fig05_backscatter_signatures.pdf'
assert hashlib.sha256(pdf.read_bytes()).hexdigest()==acceptance['published_pdf_sha256']
receipt=json.loads((out/'receipt.json').read_text(encoding='utf-8'))
receipt.update(status='passed',native_event_statistics_recomputed=True,bootstrap_recomputed=True,
    source_products_exact=checks,spatial_draws_exact=True,figure_curve_and_change_statistics_exact=True,
    absolute_summary_source=str((data/'events.csv').relative_to(ROOT)),published_figure_hash_verified=True,
    figure_rerender_required=False,figure_palette_and_layout_unchanged=True)
(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
print(absolute.to_string(index=False))
print('Native event estimates, 5000 spatial draws, and all published Fig.5 statistics verified.')
