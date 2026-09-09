module {
  aie.device(npu2) {
    %logical_core = aie.logical_tile<CoreTile>(?, ?)
    %logical_core_0 = aie.logical_tile<CoreTile>(?, ?)
    %logical_core_1 = aie.logical_tile<CoreTile>(?, ?)
    %logical_core_2 = aie.logical_tile<CoreTile>(?, ?)
    %logical_core_3 = aie.logical_tile<CoreTile>(?, ?)
    %logical_core_4 = aie.logical_tile<CoreTile>(?, ?)
    %logical_core_5 = aie.logical_tile<CoreTile>(?, ?)
    %logical_core_6 = aie.logical_tile<CoreTile>(?, ?)
    %logical_core_7 = aie.logical_tile<CoreTile>(?, ?)
    %logical_core_8 = aie.logical_tile<CoreTile>(?, ?)
    %logical_core_9 = aie.logical_tile<CoreTile>(?, ?)
    %logical_core_10 = aie.logical_tile<CoreTile>(?, ?)
    %logical_core_11 = aie.logical_tile<CoreTile>(?, ?)
    %logical_core_12 = aie.logical_tile<CoreTile>(?, ?)
    %logical_core_13 = aie.logical_tile<CoreTile>(?, ?)
    %logical_core_14 = aie.logical_tile<CoreTile>(?, ?)
    %logical_shim_noc = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_15 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_16 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_17 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_18 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_19 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_20 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_21 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_22 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_23 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_24 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_25 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_26 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_27 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_28 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_29 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_30 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_31 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_32 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_33 = aie.logical_tile<ShimNOCTile>(?, ?)
    aie.objectfifo @ac0(%logical_core_0, {%logical_core_1}, 2 : i32) : !aie.objectfifo<memref<1408xf32>>  
    aie.objectfifo @ac1(%logical_core_4, {%logical_core_5}, 2 : i32) : !aie.objectfifo<memref<1408xf32>>  
    aie.objectfifo @ac2(%logical_core_8, {%logical_core_9}, 2 : i32) : !aie.objectfifo<memref<1408xf32>>  
    aie.objectfifo @ac3(%logical_core_12, {%logical_core_13}, 2 : i32) : !aie.objectfifo<memref<1408xf32>>  
    aie.objectfifo @bd0(%logical_core, {%logical_core_0}, 1 : i32) : !aie.objectfifo<memref<2432xbf16>>  
    aie.objectfifo @bd1(%logical_core_3, {%logical_core_4}, 1 : i32) : !aie.objectfifo<memref<2432xbf16>>  
    aie.objectfifo @bd2(%logical_core_7, {%logical_core_8}, 1 : i32) : !aie.objectfifo<memref<2432xbf16>>  
    aie.objectfifo @bd3(%logical_core_11, {%logical_core_12}, 1 : i32) : !aie.objectfifo<memref<2432xbf16>>  
    aie.objectfifo @ctx0(%logical_core_2, {%logical_shim_noc}, 2 : i32) : !aie.objectfifo<memref<1024xbf16>>  
    aie.objectfifo @ctx1(%logical_core_6, {%logical_shim_noc_15}, 2 : i32) : !aie.objectfifo<memref<1024xbf16>>  
    aie.objectfifo @ctx2(%logical_core_10, {%logical_shim_noc_16}, 2 : i32) : !aie.objectfifo<memref<1024xbf16>>  
    aie.objectfifo @ctx3(%logical_core_14, {%logical_shim_noc_17}, 2 : i32) : !aie.objectfifo<memref<1024xbf16>>  
    aie.objectfifo @k0(%logical_shim_noc_18, {%logical_core_0}, 1 : i32) : !aie.objectfifo<memref<22528xbf16>>  
    aie.objectfifo @k1(%logical_shim_noc_19, {%logical_core_4}, 1 : i32) : !aie.objectfifo<memref<22528xbf16>>  
    aie.objectfifo @k2(%logical_shim_noc_20, {%logical_core_8}, 1 : i32) : !aie.objectfifo<memref<22528xbf16>>  
    aie.objectfifo @k3(%logical_shim_noc_21, {%logical_core_12}, 1 : i32) : !aie.objectfifo<memref<22528xbf16>>  
    aie.objectfifo @p0(%logical_shim_noc_22, {%logical_core}, 2 : i32) : !aie.objectfifo<memref<4992xbf16>>  
    aie.objectfifo @p1(%logical_shim_noc_23, {%logical_core_3}, 2 : i32) : !aie.objectfifo<memref<4992xbf16>>  
    aie.objectfifo @p2(%logical_shim_noc_24, {%logical_core_7}, 2 : i32) : !aie.objectfifo<memref<4992xbf16>>  
    aie.objectfifo @p3(%logical_shim_noc_25, {%logical_core_11}, 2 : i32) : !aie.objectfifo<memref<4992xbf16>>  
    aie.objectfifo @probs0(%logical_core_1, {%logical_core_2}, 2 : i32) : !aie.objectfifo<memref<1408xbf16>>  
    aie.objectfifo @probs1(%logical_core_5, {%logical_core_6}, 2 : i32) : !aie.objectfifo<memref<1408xbf16>>  
    aie.objectfifo @probs2(%logical_core_9, {%logical_core_10}, 2 : i32) : !aie.objectfifo<memref<1408xbf16>>  
    aie.objectfifo @probs3(%logical_core_13, {%logical_core_14}, 2 : i32) : !aie.objectfifo<memref<1408xbf16>>  
    aie.objectfifo @qpv0(%logical_shim_noc_26, {%logical_core}, 2 : i32) : !aie.objectfifo<memref<2048xbf16>>  
    aie.objectfifo @qpv1(%logical_shim_noc_27, {%logical_core_3}, 2 : i32) : !aie.objectfifo<memref<2048xbf16>>  
    aie.objectfifo @qpv2(%logical_shim_noc_28, {%logical_core_7}, 2 : i32) : !aie.objectfifo<memref<2048xbf16>>  
    aie.objectfifo @qpv3(%logical_shim_noc_29, {%logical_core_11}, 2 : i32) : !aie.objectfifo<memref<2048xbf16>>  
    aie.objectfifo @v0(%logical_shim_noc_30, {%logical_core_2}, 1 : i32) : !aie.objectfifo<memref<22528xbf16>>  
    aie.objectfifo @v1(%logical_shim_noc_31, {%logical_core_6}, 1 : i32) : !aie.objectfifo<memref<22528xbf16>>  
    aie.objectfifo @v2(%logical_shim_noc_32, {%logical_core_10}, 1 : i32) : !aie.objectfifo<memref<22528xbf16>>  
    aie.objectfifo @v3(%logical_shim_noc_33, {%logical_core_14}, 1 : i32) : !aie.objectfifo<memref<22528xbf16>>  
    func.func private @bd_block_bake(memref<2048xbf16>, memref<4992xbf16>) attributes {link_with = "kernels.a"}
    func.func private @bd_emit_bake(memref<2048xbf16>, memref<2432xbf16>) attributes {link_with = "kernels.a"}
    func.func private @stage_scores_relpos_bd(memref<2432xbf16>, memref<22528xbf16>, memref<1408xf32>) attributes {link_with = "kernels.a"}
    func.func private @stage_softmax(memref<1408xf32>, memref<1408xbf16>) attributes {link_with = "kernels.a"}
    func.func private @stage_ctx(memref<1408xbf16>, memref<22528xbf16>, memref<1024xbf16>) attributes {link_with = "kernels.a"}
    %0 = aie.core(%logical_core) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_34 = arith.constant 0 : index
        %c22 = arith.constant 22 : index
        %c1_35 = arith.constant 1 : index
        scf.for %arg1 = %c0_34 to %c22 step %c1_35 {
          %16 = aie.objectfifo.acquire @qpv0(Consume, 1) : !aie.objectfifosubview<memref<2048xbf16>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<2048xbf16>> -> memref<2048xbf16>
          %c0_36 = arith.constant 0 : index
          %c9 = arith.constant 9 : index
          %c1_37 = arith.constant 1 : index
          scf.for %arg2 = %c0_36 to %c9 step %c1_37 {
            %20 = aie.objectfifo.acquire @p0(Consume, 1) : !aie.objectfifosubview<memref<4992xbf16>>
            %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<4992xbf16>> -> memref<4992xbf16>
            func.call @bd_block_bake(%17, %21) : (memref<2048xbf16>, memref<4992xbf16>) -> ()
            aie.objectfifo.release @p0(Consume, 1)
          }
          %18 = aie.objectfifo.acquire @bd0(Produce, 1) : !aie.objectfifosubview<memref<2432xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<2432xbf16>> -> memref<2432xbf16>
          func.call @bd_emit_bake(%17, %19) : (memref<2048xbf16>, memref<2432xbf16>) -> ()
          aie.objectfifo.release @qpv0(Consume, 1)
          aie.objectfifo.release @bd0(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 4096 : i32}
    %1 = aie.core(%logical_core_0) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %16 = aie.objectfifo.acquire @k0(Consume, 1) : !aie.objectfifosubview<memref<22528xbf16>>
        %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<22528xbf16>> -> memref<22528xbf16>
        %c0_34 = arith.constant 0 : index
        %c22 = arith.constant 22 : index
        %c1_35 = arith.constant 1 : index
        scf.for %arg1 = %c0_34 to %c22 step %c1_35 {
          %18 = aie.objectfifo.acquire @bd0(Consume, 1) : !aie.objectfifosubview<memref<2432xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<2432xbf16>> -> memref<2432xbf16>
          %20 = aie.objectfifo.acquire @ac0(Produce, 1) : !aie.objectfifosubview<memref<1408xf32>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<1408xf32>> -> memref<1408xf32>
          func.call @stage_scores_relpos_bd(%19, %17, %21) : (memref<2432xbf16>, memref<22528xbf16>, memref<1408xf32>) -> ()
          aie.objectfifo.release @bd0(Consume, 1)
          aie.objectfifo.release @ac0(Produce, 1)
        }
        aie.objectfifo.release @k0(Consume, 1)
      }
      aie.end
    }
    %2 = aie.core(%logical_core_1) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_34 = arith.constant 0 : index
        %c22 = arith.constant 22 : index
        %c1_35 = arith.constant 1 : index
        scf.for %arg1 = %c0_34 to %c22 step %c1_35 {
          %16 = aie.objectfifo.acquire @ac0(Consume, 1) : !aie.objectfifosubview<memref<1408xf32>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<1408xf32>> -> memref<1408xf32>
          %18 = aie.objectfifo.acquire @probs0(Produce, 1) : !aie.objectfifosubview<memref<1408xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1408xbf16>> -> memref<1408xbf16>
          func.call @stage_softmax(%17, %19) : (memref<1408xf32>, memref<1408xbf16>) -> ()
          aie.objectfifo.release @ac0(Consume, 1)
          aie.objectfifo.release @probs0(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 4096 : i32}
    %3 = aie.core(%logical_core_2) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_34 = arith.constant 0 : index
        %c22 = arith.constant 22 : index
        %c1_35 = arith.constant 1 : index
        scf.for %arg1 = %c0_34 to %c22 step %c1_35 {
          %16 = aie.objectfifo.acquire @v0(Consume, 1) : !aie.objectfifosubview<memref<22528xbf16>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<22528xbf16>> -> memref<22528xbf16>
          %18 = aie.objectfifo.acquire @probs0(Consume, 1) : !aie.objectfifosubview<memref<1408xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1408xbf16>> -> memref<1408xbf16>
          %20 = aie.objectfifo.acquire @ctx0(Produce, 1) : !aie.objectfifosubview<memref<1024xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<1024xbf16>> -> memref<1024xbf16>
          func.call @stage_ctx(%19, %17, %21) : (memref<1408xbf16>, memref<22528xbf16>, memref<1024xbf16>) -> ()
          aie.objectfifo.release @probs0(Consume, 1)
          aie.objectfifo.release @ctx0(Produce, 1)
          aie.objectfifo.release @v0(Consume, 1)
        }
      }
      aie.end
    }
    %4 = aie.core(%logical_core_3) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_34 = arith.constant 0 : index
        %c22 = arith.constant 22 : index
        %c1_35 = arith.constant 1 : index
        scf.for %arg1 = %c0_34 to %c22 step %c1_35 {
          %16 = aie.objectfifo.acquire @qpv1(Consume, 1) : !aie.objectfifosubview<memref<2048xbf16>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<2048xbf16>> -> memref<2048xbf16>
          %c0_36 = arith.constant 0 : index
          %c9 = arith.constant 9 : index
          %c1_37 = arith.constant 1 : index
          scf.for %arg2 = %c0_36 to %c9 step %c1_37 {
            %20 = aie.objectfifo.acquire @p1(Consume, 1) : !aie.objectfifosubview<memref<4992xbf16>>
            %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<4992xbf16>> -> memref<4992xbf16>
            func.call @bd_block_bake(%17, %21) : (memref<2048xbf16>, memref<4992xbf16>) -> ()
            aie.objectfifo.release @p1(Consume, 1)
          }
          %18 = aie.objectfifo.acquire @bd1(Produce, 1) : !aie.objectfifosubview<memref<2432xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<2432xbf16>> -> memref<2432xbf16>
          func.call @bd_emit_bake(%17, %19) : (memref<2048xbf16>, memref<2432xbf16>) -> ()
          aie.objectfifo.release @qpv1(Consume, 1)
          aie.objectfifo.release @bd1(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 4096 : i32}
    %5 = aie.core(%logical_core_4) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %16 = aie.objectfifo.acquire @k1(Consume, 1) : !aie.objectfifosubview<memref<22528xbf16>>
        %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<22528xbf16>> -> memref<22528xbf16>
        %c0_34 = arith.constant 0 : index
        %c22 = arith.constant 22 : index
        %c1_35 = arith.constant 1 : index
        scf.for %arg1 = %c0_34 to %c22 step %c1_35 {
          %18 = aie.objectfifo.acquire @bd1(Consume, 1) : !aie.objectfifosubview<memref<2432xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<2432xbf16>> -> memref<2432xbf16>
          %20 = aie.objectfifo.acquire @ac1(Produce, 1) : !aie.objectfifosubview<memref<1408xf32>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<1408xf32>> -> memref<1408xf32>
          func.call @stage_scores_relpos_bd(%19, %17, %21) : (memref<2432xbf16>, memref<22528xbf16>, memref<1408xf32>) -> ()
          aie.objectfifo.release @bd1(Consume, 1)
          aie.objectfifo.release @ac1(Produce, 1)
        }
        aie.objectfifo.release @k1(Consume, 1)
      }
      aie.end
    }
    %6 = aie.core(%logical_core_5) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_34 = arith.constant 0 : index
        %c22 = arith.constant 22 : index
        %c1_35 = arith.constant 1 : index
        scf.for %arg1 = %c0_34 to %c22 step %c1_35 {
          %16 = aie.objectfifo.acquire @ac1(Consume, 1) : !aie.objectfifosubview<memref<1408xf32>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<1408xf32>> -> memref<1408xf32>
          %18 = aie.objectfifo.acquire @probs1(Produce, 1) : !aie.objectfifosubview<memref<1408xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1408xbf16>> -> memref<1408xbf16>
          func.call @stage_softmax(%17, %19) : (memref<1408xf32>, memref<1408xbf16>) -> ()
          aie.objectfifo.release @ac1(Consume, 1)
          aie.objectfifo.release @probs1(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 4096 : i32}
    %7 = aie.core(%logical_core_6) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_34 = arith.constant 0 : index
        %c22 = arith.constant 22 : index
        %c1_35 = arith.constant 1 : index
        scf.for %arg1 = %c0_34 to %c22 step %c1_35 {
          %16 = aie.objectfifo.acquire @v1(Consume, 1) : !aie.objectfifosubview<memref<22528xbf16>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<22528xbf16>> -> memref<22528xbf16>
          %18 = aie.objectfifo.acquire @probs1(Consume, 1) : !aie.objectfifosubview<memref<1408xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1408xbf16>> -> memref<1408xbf16>
          %20 = aie.objectfifo.acquire @ctx1(Produce, 1) : !aie.objectfifosubview<memref<1024xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<1024xbf16>> -> memref<1024xbf16>
          func.call @stage_ctx(%19, %17, %21) : (memref<1408xbf16>, memref<22528xbf16>, memref<1024xbf16>) -> ()
          aie.objectfifo.release @probs1(Consume, 1)
          aie.objectfifo.release @ctx1(Produce, 1)
          aie.objectfifo.release @v1(Consume, 1)
        }
      }
      aie.end
    }
    %8 = aie.core(%logical_core_7) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_34 = arith.constant 0 : index
        %c22 = arith.constant 22 : index
        %c1_35 = arith.constant 1 : index
        scf.for %arg1 = %c0_34 to %c22 step %c1_35 {
          %16 = aie.objectfifo.acquire @qpv2(Consume, 1) : !aie.objectfifosubview<memref<2048xbf16>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<2048xbf16>> -> memref<2048xbf16>
          %c0_36 = arith.constant 0 : index
          %c9 = arith.constant 9 : index
          %c1_37 = arith.constant 1 : index
          scf.for %arg2 = %c0_36 to %c9 step %c1_37 {
            %20 = aie.objectfifo.acquire @p2(Consume, 1) : !aie.objectfifosubview<memref<4992xbf16>>
            %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<4992xbf16>> -> memref<4992xbf16>
            func.call @bd_block_bake(%17, %21) : (memref<2048xbf16>, memref<4992xbf16>) -> ()
            aie.objectfifo.release @p2(Consume, 1)
          }
          %18 = aie.objectfifo.acquire @bd2(Produce, 1) : !aie.objectfifosubview<memref<2432xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<2432xbf16>> -> memref<2432xbf16>
          func.call @bd_emit_bake(%17, %19) : (memref<2048xbf16>, memref<2432xbf16>) -> ()
          aie.objectfifo.release @qpv2(Consume, 1)
          aie.objectfifo.release @bd2(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 4096 : i32}
    %9 = aie.core(%logical_core_8) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %16 = aie.objectfifo.acquire @k2(Consume, 1) : !aie.objectfifosubview<memref<22528xbf16>>
        %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<22528xbf16>> -> memref<22528xbf16>
        %c0_34 = arith.constant 0 : index
        %c22 = arith.constant 22 : index
        %c1_35 = arith.constant 1 : index
        scf.for %arg1 = %c0_34 to %c22 step %c1_35 {
          %18 = aie.objectfifo.acquire @bd2(Consume, 1) : !aie.objectfifosubview<memref<2432xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<2432xbf16>> -> memref<2432xbf16>
          %20 = aie.objectfifo.acquire @ac2(Produce, 1) : !aie.objectfifosubview<memref<1408xf32>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<1408xf32>> -> memref<1408xf32>
          func.call @stage_scores_relpos_bd(%19, %17, %21) : (memref<2432xbf16>, memref<22528xbf16>, memref<1408xf32>) -> ()
          aie.objectfifo.release @bd2(Consume, 1)
          aie.objectfifo.release @ac2(Produce, 1)
        }
        aie.objectfifo.release @k2(Consume, 1)
      }
      aie.end
    }
    %10 = aie.core(%logical_core_9) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_34 = arith.constant 0 : index
        %c22 = arith.constant 22 : index
        %c1_35 = arith.constant 1 : index
        scf.for %arg1 = %c0_34 to %c22 step %c1_35 {
          %16 = aie.objectfifo.acquire @ac2(Consume, 1) : !aie.objectfifosubview<memref<1408xf32>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<1408xf32>> -> memref<1408xf32>
          %18 = aie.objectfifo.acquire @probs2(Produce, 1) : !aie.objectfifosubview<memref<1408xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1408xbf16>> -> memref<1408xbf16>
          func.call @stage_softmax(%17, %19) : (memref<1408xf32>, memref<1408xbf16>) -> ()
          aie.objectfifo.release @ac2(Consume, 1)
          aie.objectfifo.release @probs2(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 4096 : i32}
    %11 = aie.core(%logical_core_10) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_34 = arith.constant 0 : index
        %c22 = arith.constant 22 : index
        %c1_35 = arith.constant 1 : index
        scf.for %arg1 = %c0_34 to %c22 step %c1_35 {
          %16 = aie.objectfifo.acquire @v2(Consume, 1) : !aie.objectfifosubview<memref<22528xbf16>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<22528xbf16>> -> memref<22528xbf16>
          %18 = aie.objectfifo.acquire @probs2(Consume, 1) : !aie.objectfifosubview<memref<1408xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1408xbf16>> -> memref<1408xbf16>
          %20 = aie.objectfifo.acquire @ctx2(Produce, 1) : !aie.objectfifosubview<memref<1024xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<1024xbf16>> -> memref<1024xbf16>
          func.call @stage_ctx(%19, %17, %21) : (memref<1408xbf16>, memref<22528xbf16>, memref<1024xbf16>) -> ()
          aie.objectfifo.release @probs2(Consume, 1)
          aie.objectfifo.release @ctx2(Produce, 1)
          aie.objectfifo.release @v2(Consume, 1)
        }
      }
      aie.end
    }
    %12 = aie.core(%logical_core_11) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_34 = arith.constant 0 : index
        %c22 = arith.constant 22 : index
        %c1_35 = arith.constant 1 : index
        scf.for %arg1 = %c0_34 to %c22 step %c1_35 {
          %16 = aie.objectfifo.acquire @qpv3(Consume, 1) : !aie.objectfifosubview<memref<2048xbf16>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<2048xbf16>> -> memref<2048xbf16>
          %c0_36 = arith.constant 0 : index
          %c9 = arith.constant 9 : index
          %c1_37 = arith.constant 1 : index
          scf.for %arg2 = %c0_36 to %c9 step %c1_37 {
            %20 = aie.objectfifo.acquire @p3(Consume, 1) : !aie.objectfifosubview<memref<4992xbf16>>
            %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<4992xbf16>> -> memref<4992xbf16>
            func.call @bd_block_bake(%17, %21) : (memref<2048xbf16>, memref<4992xbf16>) -> ()
            aie.objectfifo.release @p3(Consume, 1)
          }
          %18 = aie.objectfifo.acquire @bd3(Produce, 1) : !aie.objectfifosubview<memref<2432xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<2432xbf16>> -> memref<2432xbf16>
          func.call @bd_emit_bake(%17, %19) : (memref<2048xbf16>, memref<2432xbf16>) -> ()
          aie.objectfifo.release @qpv3(Consume, 1)
          aie.objectfifo.release @bd3(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 4096 : i32}
    %13 = aie.core(%logical_core_12) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %16 = aie.objectfifo.acquire @k3(Consume, 1) : !aie.objectfifosubview<memref<22528xbf16>>
        %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<22528xbf16>> -> memref<22528xbf16>
        %c0_34 = arith.constant 0 : index
        %c22 = arith.constant 22 : index
        %c1_35 = arith.constant 1 : index
        scf.for %arg1 = %c0_34 to %c22 step %c1_35 {
          %18 = aie.objectfifo.acquire @bd3(Consume, 1) : !aie.objectfifosubview<memref<2432xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<2432xbf16>> -> memref<2432xbf16>
          %20 = aie.objectfifo.acquire @ac3(Produce, 1) : !aie.objectfifosubview<memref<1408xf32>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<1408xf32>> -> memref<1408xf32>
          func.call @stage_scores_relpos_bd(%19, %17, %21) : (memref<2432xbf16>, memref<22528xbf16>, memref<1408xf32>) -> ()
          aie.objectfifo.release @bd3(Consume, 1)
          aie.objectfifo.release @ac3(Produce, 1)
        }
        aie.objectfifo.release @k3(Consume, 1)
      }
      aie.end
    }
    %14 = aie.core(%logical_core_13) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_34 = arith.constant 0 : index
        %c22 = arith.constant 22 : index
        %c1_35 = arith.constant 1 : index
        scf.for %arg1 = %c0_34 to %c22 step %c1_35 {
          %16 = aie.objectfifo.acquire @ac3(Consume, 1) : !aie.objectfifosubview<memref<1408xf32>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<1408xf32>> -> memref<1408xf32>
          %18 = aie.objectfifo.acquire @probs3(Produce, 1) : !aie.objectfifosubview<memref<1408xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1408xbf16>> -> memref<1408xbf16>
          func.call @stage_softmax(%17, %19) : (memref<1408xf32>, memref<1408xbf16>) -> ()
          aie.objectfifo.release @ac3(Consume, 1)
          aie.objectfifo.release @probs3(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 4096 : i32}
    %15 = aie.core(%logical_core_14) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_34 = arith.constant 0 : index
        %c22 = arith.constant 22 : index
        %c1_35 = arith.constant 1 : index
        scf.for %arg1 = %c0_34 to %c22 step %c1_35 {
          %16 = aie.objectfifo.acquire @v3(Consume, 1) : !aie.objectfifosubview<memref<22528xbf16>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<22528xbf16>> -> memref<22528xbf16>
          %18 = aie.objectfifo.acquire @probs3(Consume, 1) : !aie.objectfifosubview<memref<1408xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<1408xbf16>> -> memref<1408xbf16>
          %20 = aie.objectfifo.acquire @ctx3(Produce, 1) : !aie.objectfifosubview<memref<1024xbf16>>
          %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<1024xbf16>> -> memref<1024xbf16>
          func.call @stage_ctx(%19, %17, %21) : (memref<1408xbf16>, memref<22528xbf16>, memref<1024xbf16>) -> ()
          aie.objectfifo.release @probs3(Consume, 1)
          aie.objectfifo.release @ctx3(Produce, 1)
          aie.objectfifo.release @v3(Consume, 1)
        }
      }
      aie.end
    }
    aie.runtime_sequence(%arg0: memref<180224xbf16>, %arg1: memref<179712xbf16>, %arg2: memref<90112xbf16>, %arg3: memref<90112xbf16>, %arg4: memref<90112xbf16>) {
      %d_a0 = arith.constant 3268096 : i32
      %d_v0 = arith.constant 2 : i32
      aiex.npu.write32(%d_a0, %d_v0) : i32, i32
      %d_a1 = arith.constant 3268096 : i32
      %d_v1 = arith.constant 0 : i32
      aiex.npu.write32(%d_a1, %d_v1) : i32, i32
      %d_a2 = arith.constant 3272736 : i32
      %d_v2 = arith.constant 0 : i32
      aiex.npu.write32(%d_a2, %d_v2) : i32, i32
      %d_a3 = arith.constant 3272752 : i32
      %d_v3 = arith.constant 1 : i32
      aiex.npu.write32(%d_a3, %d_v3) : i32, i32
      %d_a4 = arith.constant 36822528 : i32
      %d_v4 = arith.constant 2 : i32
      aiex.npu.write32(%d_a4, %d_v4) : i32, i32
      %d_a5 = arith.constant 36822528 : i32
      %d_v5 = arith.constant 0 : i32
      aiex.npu.write32(%d_a5, %d_v5) : i32, i32
      %d_a6 = arith.constant 36827168 : i32
      %d_v6 = arith.constant 0 : i32
      aiex.npu.write32(%d_a6, %d_v6) : i32, i32
      %d_a7 = arith.constant 36827184 : i32
      %d_v7 = arith.constant 1 : i32
      aiex.npu.write32(%d_a7, %d_v7) : i32, i32
      %d_a8 = arith.constant 70376960 : i32
      %d_v8 = arith.constant 2 : i32
      aiex.npu.write32(%d_a8, %d_v8) : i32, i32
      %d_a9 = arith.constant 70376960 : i32
      %d_v9 = arith.constant 0 : i32
      aiex.npu.write32(%d_a9, %d_v9) : i32, i32
      %d_a10 = arith.constant 70381600 : i32
      %d_v10 = arith.constant 0 : i32
      aiex.npu.write32(%d_a10, %d_v10) : i32, i32
      %d_a11 = arith.constant 70381616 : i32
      %d_v11 = arith.constant 1 : i32
      aiex.npu.write32(%d_a11, %d_v11) : i32, i32
      %d_a12 = arith.constant 103931392 : i32
      %d_v12 = arith.constant 2 : i32
      aiex.npu.write32(%d_a12, %d_v12) : i32, i32
      %d_a13 = arith.constant 103931392 : i32
      %d_v13 = arith.constant 0 : i32
      aiex.npu.write32(%d_a13, %d_v13) : i32, i32
      %d_a14 = arith.constant 103936032 : i32
      %d_v14 = arith.constant 0 : i32
      aiex.npu.write32(%d_a14, %d_v14) : i32, i32
      %d_a15 = arith.constant 103936048 : i32
      %d_v15 = arith.constant 1 : i32
      aiex.npu.write32(%d_a15, %d_v15) : i32, i32
      %16 = aiex.dma_configure_task_for @qpv0 {
        aie.dma_bd(%arg0 : memref<180224xbf16> offset = 0 len = 2048 sizes = [22, 1, 1, 2048] strides = [2048, 0, 0, 1])
        aie.end
      } {repeat_count = 21 : i32}
      aiex.dma_start_task(%16)
      %17 = aiex.dma_configure_task_for @p0 {
        aie.dma_bd(%arg1 : memref<179712xbf16> offset = 0 len = 44928 sizes = [22, 9, 39, 128] strides = [0, 4992, 128, 1])
        aie.end
      } {repeat_count = 21 : i32}
      aiex.dma_start_task(%17)
      %19 = aiex.dma_configure_task_for @v0 {
        aie.dma_bd(%arg3 : memref<90112xbf16> offset = 0 len = 22528 sizes = [22, 1, 176, 128] strides = [0, 0, 128, 1])
        aie.end
      } {repeat_count = 21 : i32}
      aiex.dma_start_task(%19)
      %20 = aiex.dma_configure_task_for @ctx0 {
        aie.dma_bd(%arg4 : memref<90112xbf16> offset = 0 len = 1024 sizes = [22, 1, 8, 128] strides = [1024, 0, 128, 1])
        aie.end
      } {issue_token = true, repeat_count = 21 : i32}
      aiex.dma_start_task(%20)
      %21 = aiex.dma_configure_task_for @qpv1 {
        aie.dma_bd(%arg0 : memref<180224xbf16> offset = 45056 len = 2048 sizes = [22, 1, 1, 2048] strides = [2048, 0, 0, 1])
        aie.end
      } {repeat_count = 21 : i32}
      aiex.dma_start_task(%21)
      %22 = aiex.dma_configure_task_for @p1 {
        aie.dma_bd(%arg1 : memref<179712xbf16> offset = 44928 len = 44928 sizes = [22, 9, 39, 128] strides = [0, 4992, 128, 1])
        aie.end
      } {repeat_count = 21 : i32}
      aiex.dma_start_task(%22)
      %24 = aiex.dma_configure_task_for @v1 {
        aie.dma_bd(%arg3 : memref<90112xbf16> offset = 22528 len = 22528 sizes = [22, 1, 176, 128] strides = [0, 0, 128, 1])
        aie.end
      } {repeat_count = 21 : i32}
      aiex.dma_start_task(%24)
      %25 = aiex.dma_configure_task_for @ctx1 {
        aie.dma_bd(%arg4 : memref<90112xbf16> offset = 22528 len = 1024 sizes = [22, 1, 8, 128] strides = [1024, 0, 128, 1])
        aie.end
      } {issue_token = true, repeat_count = 21 : i32}
      aiex.dma_start_task(%25)
      %26 = aiex.dma_configure_task_for @qpv2 {
        aie.dma_bd(%arg0 : memref<180224xbf16> offset = 90112 len = 2048 sizes = [22, 1, 1, 2048] strides = [2048, 0, 0, 1])
        aie.end
      } {repeat_count = 21 : i32}
      aiex.dma_start_task(%26)
      %27 = aiex.dma_configure_task_for @p2 {
        aie.dma_bd(%arg1 : memref<179712xbf16> offset = 89856 len = 44928 sizes = [22, 9, 39, 128] strides = [0, 4992, 128, 1])
        aie.end
      } {repeat_count = 21 : i32}
      aiex.dma_start_task(%27)
      %29 = aiex.dma_configure_task_for @v2 {
        aie.dma_bd(%arg3 : memref<90112xbf16> offset = 45056 len = 22528 sizes = [22, 1, 176, 128] strides = [0, 0, 128, 1])
        aie.end
      } {repeat_count = 21 : i32}
      aiex.dma_start_task(%29)
      %30 = aiex.dma_configure_task_for @ctx2 {
        aie.dma_bd(%arg4 : memref<90112xbf16> offset = 45056 len = 1024 sizes = [22, 1, 8, 128] strides = [1024, 0, 128, 1])
        aie.end
      } {issue_token = true, repeat_count = 21 : i32}
      aiex.dma_start_task(%30)
      %31 = aiex.dma_configure_task_for @qpv3 {
        aie.dma_bd(%arg0 : memref<180224xbf16> offset = 135168 len = 2048 sizes = [22, 1, 1, 2048] strides = [2048, 0, 0, 1])
        aie.end
      } {repeat_count = 21 : i32}
      aiex.dma_start_task(%31)
      %32 = aiex.dma_configure_task_for @p3 {
        aie.dma_bd(%arg1 : memref<179712xbf16> offset = 134784 len = 44928 sizes = [22, 9, 39, 128] strides = [0, 4992, 128, 1])
        aie.end
      } {repeat_count = 21 : i32}
      aiex.dma_start_task(%32)
      %34 = aiex.dma_configure_task_for @v3 {
        aie.dma_bd(%arg3 : memref<90112xbf16> offset = 67584 len = 22528 sizes = [22, 1, 176, 128] strides = [0, 0, 128, 1])
        aie.end
      } {repeat_count = 21 : i32}
      aiex.dma_start_task(%34)
      %35 = aiex.dma_configure_task_for @ctx3 {
        aie.dma_bd(%arg4 : memref<90112xbf16> offset = 67584 len = 1024 sizes = [22, 1, 8, 128] strides = [1024, 0, 128, 1])
        aie.end
      } {issue_token = true, repeat_count = 21 : i32}
      aiex.dma_start_task(%35)
      aiex.dma_await_task(%20)
      aiex.dma_await_task(%25)
      aiex.dma_await_task(%30)
      aiex.dma_await_task(%35)
      aiex.dma_free_task(%16)
      aiex.dma_free_task(%17)
      aiex.dma_free_task(%19)
      aiex.dma_free_task(%21)
      aiex.dma_free_task(%22)
      aiex.dma_free_task(%24)
      aiex.dma_free_task(%26)
      aiex.dma_free_task(%27)
      aiex.dma_free_task(%29)
      aiex.dma_free_task(%31)
      aiex.dma_free_task(%32)
      aiex.dma_free_task(%34)
    }
  }
}

