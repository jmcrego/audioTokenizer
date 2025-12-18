import sys

for i,l in enumerate(sys.stdin):
    toks = l.strip().split("\t")
    if len(toks) != 5:
        sys.stderr.write(f"line {i}: bad number of tokens: {l}")
        continue

    # ASR + STT task
    t = toks[:]
    print(f"{t[0]}\t{t[1]}\t{t[2]}\t{t[3]}\t{t[4]}")

    # ASR task
    t = toks[:]
    t[3] = ""
    t[4] = ""
    print(f"{t[0]}\t{t[1]}\t{t[2]}\t{t[3]}\t{t[4]}")

    # STT task
    t = toks[:]
    t[1] = ""
    t[2] = ""
    print(f"{t[0]}\t{t[1]}\t{t[2]}\t{t[3]}\t{t[4]}")
