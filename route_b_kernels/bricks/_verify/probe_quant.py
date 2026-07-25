import verify_cast_quant as vc
r = vc.do_quantize()
print("QUANT RESULT:", r.get("status"), "rel_l2=", r.get("rel_l2"))
