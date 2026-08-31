#!/usr/bin/env python3
"""List a GGUF's tensor names, shapes and types -- no gguf package, no model load.

    python3 scripts/gguf_shapes.py <file.gguf> [name-substring]

Written to settle a layout question the hard way was costing device runs: what ggml `ne` ordering
does a given weight actually have? Answering it from the file beats instrumenting the consumer.

Worked example, the fish-audio DAC decoder upsample stack in s2-pro-q6_k.gguf:
    c.decoder.model.1.block.1.conv.weight  ne=[16, 768, 1536] f16   k=16 1536->768 stride 8
    c.decoder.model.2.block.1.conv.weight  ne=[16, 384,  768] f16   k=16  768->384 stride 8
    c.decoder.model.3.block.1.conv.weight  ne=[ 8, 192,  384] f16   k= 8  384->192 stride 4
    c.decoder.model.4.block.1.conv.weight  ne=[ 4,  96,  192] f16   k= 4  192-> 96 stride 2
ne[0] is fastest-varying, so memory order is k -> c_out -> c_in, i.e. numpy C-order [c_in, c_out, k].
Total upsample 8*8*4*2 = 512x, matching decoder_rates [8,8,4,2]; 44100/512/4 = 21.53 Hz frame rate.
Note the codec weights are f16, not quantized, even in a q6_k file.
"""
import struct, sys
# minimal GGUF tensor-info reader: header, skip KV metadata, then list tensors
T = {0:'f32',1:'f16',2:'q4_0',3:'q4_1',6:'q5_0',7:'q5_1',8:'q8_0',9:'q8_1',10:'q2_k',11:'q3_k',
     12:'q4_k',13:'q5_k',14:'q6_k',15:'q8_k',24:'i8',25:'i16',26:'i32',27:'i64',28:'f64',30:'bf16'}
f = open(sys.argv[1],'rb')
assert f.read(4)==b'GGUF'
ver, = struct.unpack('<I', f.read(4))
n_tensors, = struct.unpack('<Q', f.read(8))
n_kv, = struct.unpack('<Q', f.read(8))
def rd_str():
    n, = struct.unpack('<Q', f.read(8)); return f.read(n).decode('utf-8','replace')
def skip_val(t):
    sz = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
    if t in sz: f.read(sz[t])
    elif t == 8: rd_str()
    elif t == 9:
        et, = struct.unpack('<I', f.read(4)); n, = struct.unpack('<Q', f.read(8))
        for _ in range(n): skip_val(et)
    else: raise ValueError(f'unknown kv type {t}')
for _ in range(n_kv):
    rd_str(); t, = struct.unpack('<I', f.read(4)); skip_val(t)
pat = sys.argv[2] if len(sys.argv) > 2 else ''
for _ in range(n_tensors):
    name = rd_str()
    nd, = struct.unpack('<I', f.read(4))
    dims = struct.unpack(f'<{nd}Q', f.read(8*nd))
    ty, = struct.unpack('<I', f.read(4))
    struct.unpack('<Q', f.read(8))
    if pat in name:
        print(f"{name:58s} ne={list(dims)} type={T.get(ty,ty)}")
