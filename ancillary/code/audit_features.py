"""Read-only reconstruction of the six saved features and the caption witness.

This does not recompute the MD barcodes. Outputs are written beside this script.
"""
import csv
import hashlib
import itertools
import json
from pathlib import Path
from fractions import Fraction

import numpy as np

BASE = Path(__file__).resolve().parents[1] / 'data'
OUT = Path(__file__).resolve().parents[1] / 'reproduced'
OUT.mkdir(exist_ok=True)
FEATURES = ['zz_spanning', 'zz_retained_frac', 'zz_max_overlap',
            'zz_mean_overlap', 'zz_mean_lifetime', 'zz_step_survival']
rows = []
outside_after = outside_either = empty = boundary = nruns = 0
hashes = {}
for f in sorted(BASE.glob('global/*/eps6_deb3/*.json')):
    r = json.loads(f.read_text())
    if r['temperature'] == 320 or not r['qualifying']:
        continue
    nruns += 1
    hashes[str(f.relative_to(BASE))] = hashlib.sha256(f.read_bytes()).hexdigest()
    lo = np.asarray(r['lo'], dtype=int)
    hi = np.asarray(r['hi'], dtype=int)
    assert np.array_equal(lo, r['lo']) and np.array_equal(hi, r['hi'])
    empty += int((lo == hi).sum())
    assert np.all(lo < hi), 'Empty frame supports would need explicit filtering.'
    nframes = len(r['beta'])
    boundary += int(((lo == 0) | (hi == nframes)).sum())
    alive = (lo[:, None] <= np.arange(nframes)) & (np.arange(nframes) < hi[:, None])
    np.testing.assert_array_equal(alive.sum(axis=0), r['beta'])
    for w in r['windows']:
        a, b = w['start'], w['stop']
        length = b - a
        subset = (lo < b) & (hi > a)
        overlap = np.maximum(0, np.minimum(hi, b) - np.maximum(lo, a))[subset]
        spanning = int(((lo <= a) & (hi >= b)).sum())
        survival = [(alive[:, t] & alive[:, t + 1]).sum() / max(alive[:, t].sum(), 1)
                    for t in range(a, b - 1)]
        outside_after += int(np.any(subset & (hi > b)))
        outside_either += int(np.any(subset & ((lo < a) | (hi > b))))
        mean_overlap = float(overlap.mean() / length) if len(overlap) else 0.
        clipped_lifetime = float(overlap.mean()) if len(overlap) else 0.
        assert np.isclose(clipped_lifetime, length * mean_overlap)
        rows.append(dict(protein=r['protein'], T=r['temperature'], rep=r['replica'], start=a,
                         zz_spanning=spanning,
                         zz_retained_frac=spanning / max(alive[:, a].sum(), 1),
                         zz_max_overlap=float(overlap.max() / length) if len(overlap) else 0.,
                         zz_mean_overlap=mean_overlap,
                         zz_mean_lifetime=float((hi-lo)[subset].mean()) if subset.any() else 0.,
                         zz_step_survival=float(np.mean(survival)),
                         clipped_lifetime=clipped_lifetime, length=length,
                         overlap_sum=int(overlap.sum()), intersecting_bars=len(overlap)))

saved_path = BASE / 'baseline_test/window_zigzag_features.csv'
with saved_path.open() as stream:
    saved = list(csv.DictReader(stream))
keys = ['protein', 'T', 'rep', 'start']

def key(row):
    return row['protein'], int(row['T']), int(row['rep']), int(row['start'])

reconstructed = {key(row): row for row in rows}
saved_by_key = {key(row): row for row in saved}
assert len(reconstructed) == len(rows) == len(saved_by_key) == len(saved)
assert reconstructed.keys() == saved_by_key.keys()
errors = {feature: 0. for feature in FEATURES}
for row_key, computed in reconstructed.items():
    for feature in FEATURES:
        error = abs(computed[feature]-float(saved_by_key[row_key][feature]))
        errors[feature] = max(errors[feature], error)
        assert error < 1e-12

window_path = BASE / 'baseline_test/window_features.csv'
with window_path.open() as stream:
    windows = list(csv.DictReader(stream))
assert len({key(row) for row in windows}) == len(windows)
data = []
for window in windows:
    if key(window) not in reconstructed:
        continue
    occupied = window['occupied'].lower() == 'true'
    q = float(window['Q'])
    if not occupied or .3 <= q < .6:
        continue
    row = dict(reconstructed[key(window)])
    # Evaluate the exact saved values used by the original AUC calculation.
    row.update({feature: float(saved_by_key[key(window)][feature]) for feature in FEATURES})
    # Exact rational counts prevent tiny arithmetic-order changes from breaking ties.
    row['exact_mean_overlap'] = Fraction(row['overlap_sum'], row['intersecting_bars']*row['length']) if row['intersecting_bars'] else Fraction(0)
    row['exact_clipped_lifetime'] = Fraction(row['overlap_sum'], row['intersecting_bars']) if row['intersecting_bars'] else Fraction(0)
    assert row['exact_clipped_lifetime'] == row['length']*row['exact_mean_overlap']
    row['label'] = int(q >= .6)
    data.append(row)

def roc_auc_score(labels, scores):
    # Direct positive-versus-negative comparison, with half credit for ties.
    labels, scores = np.asarray(labels), np.asarray(scores)
    positives, negatives = scores[labels == 1], scores[labels == 0]
    return float(((positives[:, None] > negatives).sum()
                  + .5*(positives[:, None] == negatives).sum())/(len(positives)*len(negatives)))

aucs = {}
per_domain = []
for feature in FEATURES + ['exact_mean_overlap', 'exact_clipped_lifetime']:
    values = {}
    for protein in sorted({row['protein'] for row in data}):
        group = [row for row in data if row['protein'] == protein]
        labels = [row['label'] for row in group]
        if len(set(labels)) != 2:
            continue
        value = roc_auc_score(labels, [row[feature] for row in group])
        values[protein] = float(value)
        per_domain.append(dict(protein=protein, feature=feature, auc=value))
    aucs[feature] = dict(median=float(np.median(list(values.values()))), minimum=min(values.values()),
                         domains=len(values), by_domain=values)
assert aucs['exact_clipped_lifetime']['by_domain'] == aucs['exact_mean_overlap']['by_domain']

def gf2_rank(columns):
    basis = {}
    for value in columns:
        while value:
            pivot = value.bit_length()-1
            if pivot in basis:
                value ^= basis[pivot]
            else:
                basis[pivot] = value
                break
    return len(basis)

def rips(points, cutoff):
    edges = {edge for edge in itertools.combinations(range(len(points)), 2)
             if np.linalg.norm(points[edge[0]]-points[edge[1]]) <= cutoff}
    triangles = {tri for tri in itertools.combinations(range(len(points)), 3)
                 if all(tuple(edge) in edges for edge in itertools.combinations(tri, 2))}
    return edges, triangles

def beta1(complex_):
    edges, triangles = complex_
    edge_index = {edge: i for i, edge in enumerate(sorted(edges))}
    boundary1 = [(1 << i) ^ (1 << j) for i, j in edges]
    boundary2 = [sum(1 << edge_index[edge] for edge in itertools.combinations(tri, 2))
                 for tri in triangles]
    return len(edges)-gf2_rank(boundary1)-gf2_rank(boundary2)

a, b, epsilon = .8, .2, 1.
square = np.array([[0,0,0], [a,0,0], [a,a,0], [0,a,0]])
cluster = np.array([[10,0,0], [10+b,0,0], [10+b,b,0], [10,b,0]])
points = np.concatenate([square, cluster])
swapped = np.concatenate([cluster, square])
cutoffs = [.1, .22, .5, .9, 1.2]
betti = [beta1(rips(points, cutoff)) for cutoff in cutoffs]
assert betti == [0, 1, 0, 1, 0]
first, second = rips(points, epsilon), rips(swapped, epsilon)
union = (first[0] | second[0], first[1] | second[1])
assert beta1(first) == beta1(second) == 1 and beta1(union) == 0

result = dict(scope='Saved-barcode and six single-feature AUC audit; no MD or barcode recomputation, combined-model retraining, or live Overleaf edits.',
              runs=nruns, window_rows=len(saved), classification_rows=len(data),
              feature_reproduction_max_errors=errors, empty_frame_support_bars=empty,
              bars_touching_a_record_boundary=boundary,
              windows_with_a_bar_extending_after_window=outside_after,
              windows_with_a_bar_extending_outside_window=outside_either,
              clipped_mean_equals_L_times_overlap=True,
              clipped_auc_equality_checked_using_exact_rational_lengths=True,
              overlap_median_auc_rounded_directly_to_two_decimals='0.98',
              raw_feature_auc=aucs,
              caption_counterexample=dict(a=a,b=b,epsilon=epsilon,cutoffs=cutoffs,beta1=betti,
                                          guaranteed_points_below_intercluster_scales=[[b,b*np.sqrt(2)], [a,a*np.sqrt(2)]],
                                          beta1_at_epsilon=1, beta1_relabelled_union=0),
              hashes={str(saved_path.relative_to(BASE)):hashlib.sha256(saved_path.read_bytes()).hexdigest(),
                      str(window_path.relative_to(BASE)):hashlib.sha256(window_path.read_bytes()).hexdigest(), **hashes})
(OUT/'audit_results.json').write_text(json.dumps(result,indent=2)+'\n')
with (OUT/'feature_auc_by_domain.csv').open('w',newline='') as stream:
    writer = csv.DictWriter(stream,fieldnames=['protein','feature','auc'])
    writer.writeheader()
    writer.writerows(per_domain)
print(json.dumps({k:v for k,v in result.items() if k not in ['hashes','raw_feature_auc']},indent=2))
print('Raw feature AUCs:',json.dumps({k:{q:v[q] for q in ['median','minimum','domains']} for k,v in aucs.items()}))
