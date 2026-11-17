"""Reproduce the archived transition-246 critical-simplex endpoints for Figure 3's case."""
from pathlib import Path
import ast
import csv
import json
import numpy as np
from case_barcode import run

ROOT = Path(__file__).resolve().parents[1]
coordinates = np.load(ROOT / 'data/5sic_450_rep0_ca.npy')
with (ROOT / 'data/5sic_attribution/transition245_246.csv').open() as source:
    archived = sorted((row['kind'], row['op'], tuple(ast.literal_eval(row['res'])))
                      for row in csv.DictReader(source))
recomputed = []
for bar in run(0, 244, 248, eps=8.0, X=coordinates):
    for kind, field in [('birth', 'bop'), ('death', 'dop')]:
        operation = bar[field]
        if operation is not None and operation[2] == 246:
            action, simplex, frame = operation
            recomputed.append((kind, action, tuple(index + 7 for index in simplex)))
assert sorted(recomputed) == archived
assert ('birth', 'add', (7, 57)) in archived
assert ('death', 'rem', (16, 43)) in archived
distances = {}
for first, second in [(7, 57), (16, 43)]:
    distances[f'{first},{second}'] = [float(np.linalg.norm(
        coordinates[frame, first-7] - coordinates[frame, second-7])) for frame in [245, 246]]
result = {'matched_endpoints': len(archived), 'frames_recomputed': [244, 248],
          'transition': [245, 246], 'cutoff_A': 8.0, 'residue_numbering': 'PDB 7 through 113',
          'critical_endpoints': archived, 'contact_distances_A_at_frames_245_246': distances}
output = ROOT / 'reproduced/case_attribution_check.json'
output.parent.mkdir(exist_ok=True)
output.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
