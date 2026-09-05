import verify_moe_topk_router as m
r = m.do_moe_topk_router()
print("MOE RESULT:", r.get("status"), "rel_l2=", r.get("rel_l2"), "nz=", r.get("nonzero"))
