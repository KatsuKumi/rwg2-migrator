"""Stands in for `java ... lwjgl3ify-forgePatches.jar nogui` in tests: the probe protocol without Minecraft.

FAKE_PROBE=ignore-stop answers stop but keeps running; FAKE_PROBE=die-at-boot exits 7 before listening.
"""

import json
import os
import socket
import sys
import threading

ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("-D") and "=" in a)
MODE = os.environ.get("FAKE_PROBE", "")


def reply_for(request, token):
    if request.get("token") != token:
        return {"ok": False, "op": request.get("op"), "error": "java.lang.SecurityException: missing or wrong token"}
    op = request.get("op")
    out = {"ok": True, "op": op, "seconds": 0.0}
    if op == "status":
        out |= {"instance": ARGS.get("rwg2probe.instance"), "kind": ARGS.get("rwg2probe.kind"), "rwg2": "2.0.6", "java": "fake"}
    elif op == "stop":
        out["stopping"] = True
    elif op == "echo":
        out["echo"] = request.get("value")
    else:
        out = {"ok": False, "op": op, "error": f"unknown op {op}"}
    return out


def serve(conn, token):
    reader = conn.makefile("rb")
    for line in reader:
        reply = reply_for(json.loads(line), token)
        conn.sendall((json.dumps(reply) + "\n").encode())
        if reply.get("stopping") and MODE != "ignore-stop":
            os._exit(0)


def main():
    if MODE == "die-at-boot":
        print("fake boot failure", flush=True)
        sys.exit(7)
    with open("rwg2probe.token", encoding="utf-8") as f:
        token = f.read().strip()
    server = socket.create_server(("127.0.0.1", 0))
    with open("rwg2probe.port.tmp", "w") as f:
        f.write(str(server.getsockname()[1]))
    os.replace("rwg2probe.port.tmp", "rwg2probe.port")
    print("rwg2probe listening", flush=True)
    while True:
        conn, _ = server.accept()
        threading.Thread(target=serve, args=(conn, token), daemon=True).start()


if __name__ == "__main__":
    main()
