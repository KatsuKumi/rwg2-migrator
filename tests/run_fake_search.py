"""Runs one search job with the fake sampler, slowly, so a test can kill it mid-search: run_fake_search.py DATA JOB WORLD AREA CACHE."""

import sys

from fake_sampler import FakeSampler
from test_search import SMALL

from rwg2migrator import appearance, jobs, scan, search

DELAY = 0.05

data, job_id, world, area_path, cache = sys.argv[1:6]
area = search.load(area_path)
job = jobs.find(data, job_id)
sampler = FakeSampler(DELAY)
inp = search.Inputs(scan.scan(world, cache, workers=1), area, appearance.load_prefs(None), "2.9.0", 1234567890123, lambda: sampler, SMALL)
jobs.run(job, lambda j: search.run_job(j, inp))
