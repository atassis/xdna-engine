import verify_rope_lut as m
r = m.do_rope_lut()
print("ROPE RESULT:", r.get("status"), "rel_l2=", r.get("rel_l2"))
