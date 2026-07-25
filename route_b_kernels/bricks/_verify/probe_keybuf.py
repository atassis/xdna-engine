import numpy as np, importlib.util
from bricklib import GEN, iron, _build_oneshot
spec=importlib.util.spec_from_file_location("g","route_b_kernels/bricks/rope-lut/golden.py")
g=importlib.util.module_from_spec(spec); spec.loader.exec_module(g)
M,ROT=4,128; half=ROT//2
inv_freq=g.build_inv_freq(ROT).astype(np.float32)
pos=np.arange(M,dtype=np.int32)
cbuf=np.concatenate([pos.view(np.int32), inv_freq.view(np.int32)]).astype(np.int32)   # [pos | inv_freq]
# standalone kernel: replicate rope_lut.cc's key computation, output int8 key_buf [M, half]
shim=('#include <aie_api/aie.hpp>\n#include <stdint.h>\n'
 'extern "C" void key_probe(int32_t*restrict cbuf, int8_t*restrict out){\n'
 f'  const int M={M}, HALF={half}; const float PI=3.14159265358979f, SCALE_INV=1.0f;\n'
 '  const int32_t*pos=cbuf; const float*inv_freq=(const float*)(cbuf+M);\n'
 '  for(int m=0;m<M;++m){ float posf=(float)pos[m]*SCALE_INV;\n'
 '    for(int i=0;i<HALF;++i){ float theta=posf*inv_freq[i];\n'
 '      float kw=theta*(1.0f/(2.0f*PI)); int k=(int)(kw+(kw>=0.0f?0.5f:-0.5f));\n'
 '      float wr=theta-(float)k*(2.0f*PI); float q=wr*(128.0f/PI);\n'
 '      int key=(int)(q+(q>=0.0f?0.5f:-0.5f)); if(key>127)key=127; if(key<-128)key=-128;\n'
 '      out[m*HALF+i]=(int8_t)key; } } }\n')
gshim=GEN/"key_probe_shim.cc"; gshim.write_text(shim)
design=_build_oneshot("key_probe", gshim, [cbuf.size], M*half, [np.int32], np.int8, [])
ct=iron.tensor(np.ascontiguousarray(cbuf),dtype=np.int32,device="npu")
ot=iron.zeros((M*half,),dtype=np.int8,device="npu"); design(ct,ot)
dev=ot.numpy().reshape(M,half).astype(int)
# host golden keys
for m in range(M):
    theta=pos[m]*inv_freq; key=g._quantize_key(theta)
    print(f"row{m} pos={pos[m]}: dev_key[:6]={dev[m,:6].tolist()} host_key[:6]={key[:6].tolist()} match={np.array_equal(dev[m],key)}")
print("dev rows all identical?", all(np.array_equal(dev[0],dev[m]) for m in range(M)))
