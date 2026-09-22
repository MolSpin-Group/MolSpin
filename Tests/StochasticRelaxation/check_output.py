#!/usr/bin/env python3
"""Check the output from st_dephasing.msd (standard Python only)."""
import math
from pathlib import Path


def read(name):
    return [[float(x) for x in line.split()]
            for line in Path(name).read_text().splitlines()[1:] if line.strip()]


direct = read('st-direct.dat')
stochastic = read('st-stochastic.dat')
assert len(direct) == len(stochastic) == 81
assert all(len(row) == 6 for row in direct + stochastic)
assert all(math.isfinite(x) for row in direct + stochastic for x in row)
assert all(a[:2] == b[:2] for a, b in zip(direct, stochastic))
error = max(abs(a - b) for ra, rb in zip(direct, stochastic)
            for a, b in zip(ra[2:], rb[2:]))
# Six times the worst-case standard error of a bounded population estimator.
bound = 6 / (2 * math.sqrt(8192))
assert error < bound, (error, bound)
survival = max(abs(sum(row[2:]) - math.exp(-0.2 * row[1])) for row in stochastic)
assert survival < 2e-6, survival
assert 'no density propagation' in Path('st-stochastic.log').read_text()
print(f'PASS: max population difference = {error:.8g} (bound {bound:.8g})')
print(f'PASS: maximum survival error = {survival:.8g}')
