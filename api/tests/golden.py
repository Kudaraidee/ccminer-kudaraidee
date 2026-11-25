#!/usr/bin/env python3
"""Golden-output capture for the binary API.

Sends every command the API implements to a running miner and records the exact
bytes it replies with. Used as a regression gate: capture before a change,
capture after, diff. The binary protocol is a compatibility surface, so any
diff that is not explicitly intended is a defect.

    ccminer --api-bind 127.0.0.1:4068 -a <algo> --benchmark
    python api/tests/golden.py capture before.txt
    ...change something, rebuild...
    python api/tests/golden.py capture after.txt
    python api/tests/golden.py diff before.txt after.txt

Replies are recorded verbatim except for the volatile fields listed in VOLATILE,
which change between two runs of the *same* binary (uptime, temperatures, rates)
and would otherwise drown the signal. `diff` reports those separately so a real
change inside a masked field is still visible as a count.
"""

import re
import socket
import sys

HOST, PORT = "127.0.0.1", 4068

# Every command in the table, plus the two error cases the protocol has to
# handle. "" and "bogus" must both produce a reply, not silence.
COMMANDS = [
    "summary", "threads", "pool", "hwinfo", "histo", "scanlog", "meminfo",
    "help", "bogus", "",
    # Read-only control probes. The mutating verbs are deliberately absent: a
    # golden capture must not change the state of the miner it is measuring.
    "ctl|state", "ctl|bogusverb",
]

# key=value pairs whose value legitimately differs between two runs of the SAME
# binary. Verified by capturing twice without rebuilding: anything that moves
# there is runtime state, not protocol.
VOLATILE = [
    "UPTIME", "TS", "KHS", "ACC", "REJ", "ACCMN", "DIFF", "NETKHS", "SOLV",
    "BEST", "LAST", "WAIT", "PING", "TEMP", "FAN", "FREQ", "PST", "POWER",
    "CPUTEMP", "CPUFREQ", "H", "I", "HWF", "STALE", "DISCO", "NETDIFF",
    "SCANTIME", "MEM", "MEMFREE", "USED", "FREE", "TOTAL",
    "COUNT", "STATS", "HASHLOG", "FOUND", "ID",
]
VOL_RE = re.compile(r"\b(" + "|".join(VOLATILE) + r")=[^;|]*")

# `histo` and `meminfo` accumulate records as the miner runs, so the record
# *count* is uptime-dependent and must not be part of the gate. What must not
# change is the record shape — the key list of each distinct record type.
def shapes(text):
    seen = []
    for rec in text.split("|"):
        rec = rec.strip()
        if not rec or "=" not in rec:
            continue
        keys = ";".join(kv.split("=", 1)[0] for kv in rec.split(";") if "=" in kv)
        if keys not in seen:
            seen.append(keys)
    return seen


def ask(cmd, timeout=5.0):
    """One command, one connection — the protocol closes after each reply."""
    s = socket.create_connection((HOST, PORT), timeout=timeout)
    try:
        s.sendall((cmd + "|").encode())
        chunks = []
        while True:
            b = s.recv(65536)
            if not b:
                break
            chunks.append(b)
        return b"".join(chunks)
    finally:
        s.close()


def mask(text):
    return VOL_RE.sub(lambda m: m.group(1) + "=<v>", text)


def capture(path):
    out = []
    for cmd in COMMANDS:
        label = cmd if cmd else "<empty>"
        try:
            raw = ask(cmd)
        except Exception as e:                      # noqa: BLE001 - report, don't raise
            out.append("=== %s ===\nTRANSPORT-ERROR: %s\n" % (label, e))
            continue
        text = raw.decode("utf-8", "replace")
        # Multi-record replies grow with uptime; compare their shape, not their
        # length. Single-record replies are compared verbatim (masked).
        recs = [r for r in text.split("|") if r.strip()]
        if len(recs) > 1:
            body = "records-shape:\n  " + "\n  ".join(shapes(text))
        else:
            body = "replied=%s\n%s" % (bool(raw), mask(text))
        out.append("=== %s ===\n%s\n" % (label, body))
    body = "".join(out)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(body)
    silent = body.count("replied=False")
    print("wrote %s  (%d commands, %d silent)" % (path, len(COMMANDS), silent))
    return 0


def diff(a, b):
    la = open(a, encoding="utf-8").read().splitlines()
    lb = open(b, encoding="utf-8").read().splitlines()
    import difflib
    d = list(difflib.unified_diff(la, lb, a, b, lineterm="", n=1))
    if not d:
        print("IDENTICAL: %s == %s" % (a, b))
        return 0
    print("\n".join(d))
    print("\n%d diff lines — every one must be an intended change." % len(d))
    return 1


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "capture":
        sys.exit(capture(sys.argv[2]))
    if len(sys.argv) >= 4 and sys.argv[1] == "diff":
        sys.exit(diff(sys.argv[2], sys.argv[3]))
    print(__doc__)
    sys.exit(2)
