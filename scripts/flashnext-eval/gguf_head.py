#!/usr/bin/env python3
import struct, sys

SIZE = {0:1, 1:1, 2:2, 3:2, 4:4, 5:4, 6:4, 7:1, 10:8, 11:8, 12:8}
FMT  = {0:"<B", 1:"<b", 2:"<H", 3:"<h", 4:"<I", 5:"<i", 6:"<f",
        7:"<?", 10:"<Q", 11:"<q", 12:"<d"}

def rd_str(f):
    n = struct.unpack("<Q", f.read(8))[0]
    return f.read(n).decode("utf-8", "replace")

def rd_scalar(f, t):
    return struct.unpack(FMT[t], f.read(SIZE[t]))[0]

def rd_val(f, t):
    if t == 8:
        return rd_str(f)
    if t == 9:
        et, n = struct.unpack("<IQ", f.read(8))
        if et == 8:
            return [rd_str(f) for _ in range(n)]
        data = f.read(SIZE[et] * n)
        cnt = struct.unpack("<" + FMT[et][1] * min(n, 8), data[:SIZE[et]*min(n,8)])
        return f"array<{et}>x{n} first={cnt}"
    return rd_scalar(f, t)

def main(path):
    f = open(path, "rb")
    magic, ver, n_tensors, n_kv = struct.unpack("<4sIQQ", f.read(24))
    print("magic", magic, "ver", ver, "tensors", n_tensors, "kv", n_kv)
    kvs = {}
    for _ in range(n_kv):
        k = rd_str(f)
        t = struct.unpack("<I", f.read(4))[0]
        kvs[k] = rd_val(f, t)
    for k, v in kvs.items():
        if "arch" in k or "nextn" in k or "name" in k or "embedding" in k or "feed_forward" in k or "expert" in k or "attention." in k or "block_count" in k or "head" in k or "rope" in k or "hc" in k or "dsv4" in k or "ple" in k or "indexer" in k or "ssm" in k:
            print(f"  {k} = {v}")
    print("--- tensors ---")
    for _ in range(n_tensors):
        nm = rd_str(f)
        nd = struct.unpack("<I", f.read(4))[0]
        dims = struct.unpack("<" + "Q" * nd, f.read(8 * nd))
        ty, off = struct.unpack("<IQ", f.read(12))
        print(f"  type={ty:3d} dims={dims} {nm}")

if __name__ == "__main__":
    main(sys.argv[1])
