import verify_bfp16 as m
r = m.do_gemm_bfp16_ebs8()
print("BFP16-EBS8 RESULT:", r.get("status"), "rel_l2=", r.get("rel_l2"))
