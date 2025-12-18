import sys

for l in sys.stdin:
    toks = l.strip().split("\t")
    if len(toks) != 5:
        continue

    # ASR + STT task
    print("\t".join(toks))

    # ASR task
    t = toks[:]
    t[3] = ""
    t[4] = ""
    print("\t".join(t))

    # STT task
    t = toks[:]
    t[1] = ""
    t[2] = ""
    print("\t".join(t))
