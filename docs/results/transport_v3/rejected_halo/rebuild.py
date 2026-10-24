"""Rebuild the three rejected-halo comparisons; run from the repository root."""
import hashlib
import json
from pathlib import Path

from scipy.io import loadmat

from examples.report_validation import area, overlap, write

root = Path(__file__).resolve().parent
manifest = json.loads((root / 'manifest.json').read_text())
fields, summaries = {}, {}
for name in manifest['cases']:
    folder = root / 'solved' / name
    summary = json.loads((folder / 'summary.json').read_text())
    for filename, digest in summary['sources'].items():
        assert hashlib.sha256((folder / filename).read_bytes()).hexdigest() == digest
    for filename, key in [('projection.mat', 'projection_sha256'), ('problem.json', 'input_sha256')]:
        assert hashlib.sha256((folder / filename).read_bytes()).hexdigest() == summary[key]
    data = loadmat(folder / 'projection.mat')
    fields[name] = data['theta_rad'].ravel(), data['velocity_rad_s'].ravel(), data['projected_value']
    summaries[name] = summary
rows = []
limits = manifest['acceptance']
for pair in manifest['comparisons']:
    a, b = [summaries[pair[k]] for k in ['coarse', 'fine']]
    for key in ['input_sha256', 'sources', 'target', 'effective_profile', 'duration_s', 'method']:
        assert a[key] == b[key]
    coarse, fine = [fields[pair[k]] for k in ['coarse', 'fine']]
    error = 100 * abs(area(*coarse, b['direction']) / area(*fine, b['direction']) - 1)
    iou = overlap(coarse, fine, b['direction'])
    temporal = pair['kind'] == 'temporal'
    passed = (error <= limits['temporal_area_relative_percent' if temporal else 'spatial_area_relative_percent']
              and iou >= limits['temporal_projection_iou' if temporal else 'common_grid_projection_iou'])
    rows.append(dict(pair, area_error_percent=error, projection_iou=iou, area_and_iou_passed=passed,
                     assistance_effect_assessed=False))
write(root / 'comparisons.json', dict(comparisons=rows, acceptance=limits))
print(json.dumps(rows, indent=2))
