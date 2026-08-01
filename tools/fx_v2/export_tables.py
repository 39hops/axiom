"""FX-V2 table exporter: p3_tables.pt -> flat AXP3 binary.

Stdlib ONLY (zipfile + pickle) — never imports torch. The .pt zip
holds data.pkl (an OrderedDict of tensors) plus raw little-endian
int64 storage blobs; we rebuild shapes from the pickle and copy the
bytes verbatim. Input sha256 is verified against the FX-V1-H pin
before anything is written.

AXP3 format (little-endian):
  magic "AXP3", u32 n_tensors, then per tensor:
    u16 name_len, name utf-8, u8 ndim, u64 dim[ndim], i64 data[...]

Pickle safety: input bytes are sha256-pinned to the FX-V1-H artifact
before parsing, and the Unpickler whitelist rejects every global
except OrderedDict / _rebuild_tensor_v2 / LongStorage — no arbitrary
class construction is reachable.

Usage: python export_tables.py <tables.pt> <out.bin> [expected_sha256]
(the FX-V1-H p3_tables pin is the default; FX-V3 passes its own)
"""
import hashlib
import io
import pickle
import struct
import sys
import zipfile

PIN = "ecef909d46eb821429e48b470a07dfdedc492c9247de5cff665a57ac5378e219"


def _rebuild(storage, offset, size, stride, *_):
    assert offset == 0
    return {"key": storage["key"], "size": size, "numel": storage["numel"]}


class U(pickle.Unpickler):
    def find_class(self, mod, name):
        if (mod, name) == ("torch._utils", "_rebuild_tensor_v2"):
            return _rebuild
        if mod == "torch" and name.endswith("Storage"):
            assert name == "LongStorage", f"non-int64 storage {name}"
            return name
        if (mod, name) == ("collections", "OrderedDict"):
            import collections
            return collections.OrderedDict
        raise pickle.UnpicklingError(f"unexpected global {mod}.{name}")

    def persistent_load(self, pid):
        tag, stype, key, _loc, numel = pid
        assert tag == "storage" and stype == "LongStorage"
        return {"key": key, "numel": numel}


def main(src, dst, pin=PIN):
    raw = open(src, "rb").read()
    sha = hashlib.sha256(raw).hexdigest()
    assert sha == pin, f"sha mismatch: {sha}"
    z = zipfile.ZipFile(io.BytesIO(raw))
    root = z.namelist()[0].split("/")[0]
    d = U(io.BytesIO(z.read(f"{root}/data.pkl"))).load()
    out = io.BytesIO()
    out.write(b"AXP3" + struct.pack("<I", len(d)))
    for name, t in d.items():
        blob = z.read(f"{root}/data/{t['key']}")
        assert len(blob) == 8 * t["numel"]
        nb = name.encode()
        out.write(struct.pack("<H", len(nb)) + nb)
        out.write(struct.pack("<B", len(t["size"])))
        for s in t["size"]:
            out.write(struct.pack("<Q", s))
        out.write(blob)
    open(dst, "wb").write(out.getvalue())
    print(f"{len(d)} tensors -> {dst} "
          f"({len(out.getvalue())} bytes) from sha-verified input")


if __name__ == "__main__":
    main(*sys.argv[1:4])
