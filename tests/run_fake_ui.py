"""The UI with the fake generator, in itself and in every step it starts: run_fake_ui.py DATA CACHE [PORT]."""

import json
import os
import sys
from types import SimpleNamespace

import fake_gen

from rwg2migrator import provision
from rwg2migrator.web import server

HERE = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    os.environ[server.CLI_ENV] = json.dumps([sys.executable, os.path.join(HERE, "run_fake_cli.py")])
    os.environ["RWG2_SMALL"] = "1"
    fake_gen.install()
    provision.claim_cache(sys.argv[2])
    fake_gen.settings(sys.argv[2])
    port = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    sys.exit(server.serve(SimpleNamespace(data=sys.argv[1], cache=sys.argv[2], port=port, no_browser=True)))
