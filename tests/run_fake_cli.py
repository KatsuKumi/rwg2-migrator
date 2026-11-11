"""The CLI with the fake generator and small generate batches, so the power-cut tests can kill a preview inside generate and smooth:
run_fake_cli.py ARGS..."""

import dataclasses
import os
import sys

import fake_gen

from rwg2migrator import cli, cli_search, generator

BATCH = 8

if __name__ == "__main__":
    fake_gen.install()
    generator.GEN_BATCH = BATCH
    if os.environ.get("RWG2_SMALL"):
        from test_search import SMALL

        inputs = cli_search._inputs
        cli_search._inputs = lambda job, sc, gens: dataclasses.replace(inputs(job, sc, gens), plan=SMALL)
    sys.exit(cli.main(sys.argv[1:]))
