;===- tanh_subreg_clobber.ll ---------------------------------------------===;
;
; MINIMAL reproducer for the aie2p (Peano/llvm-aie) SUBREGISTER-LIVENESS wrong-code
; defect that made tanh_ab's second aie::tanh return 0 -- gelu_hw degenerating to
; exactly 0.5*x on 510/512 lanes.
;
; The defect needs neither a second tanh nor a noinline call, which is why the
; original "second-call miscompile" framing never reduced: one llvm.aie2p.tanh, no
; call, no loop. What it DOES need is the subregister insert -- a 256-bit
; <16 x bfloat> widened to a 512-bit <32 x bfloat> by `shufflevector` against
; zeroinitializer and fed to bf.mul.conf -- twice, so two widenings compete for one
; x register. The coalescer then joins intervals across a lane it has lost track of.
;
; SIGNATURE, in llc -O2 output: the vtanh destination `wh<N>` is redefined by a
; `vmov wh<N>, ...` before its only consumer reads it. check_vtanh_clobber.py scores
; this; it models both x<N> = {wl<N>, wh<N>} and VLIW bundle order (every slot reads
; before any slot writes), and a checker missing either reports false positives.
;
;   llc -O2 -mtriple=aie2p-none-unknown-elf tanh_subreg_clobber.ll -o - \
;     | python3 check_vtanh_clobber.py /dev/stdin
;
; SELF-VALIDATING via flags -- either of these makes the same file correct, and
; neither -verify-machineinstrs nor -verify-coalescing reports anything:
;   -enable-subreg-liveness=false   correct
;   -join-liveintervals=false       correct
;
; Reduced by llvm-reduce from tanh_ab.cc -DTANHAB_NOSW=0 (1814 lines -> 60).
; Reproduces on all four local Peano builds, incl. spillfix-2026-08-20.
;
; SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
;===----------------------------------------------------------------------===;
target datalayout = "e-m:e-p:20:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-f32:32:32-i64:32-f64:32-a:0:32-n32"
target triple = "aie2p-none-unknown-elf"

define void @tanh_ab() {
  %1 = tail call <16 x bfloat> @llvm.aie2p.tanh(<16 x float> zeroinitializer)
  %2 = bitcast <16 x bfloat> %1 to <8 x i32>
  %3 = shufflevector <8 x i32> %2, <8 x i32> zeroinitializer, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %4 = bitcast <16 x i32> %3 to <32 x bfloat>
  %5 = tail call <32 x float> @llvm.aie2p.I512.I512.ACC1024.bf.mul.conf(<32 x bfloat> zeroinitializer, <32 x bfloat> %4, i32 0)
  %6 = shufflevector <32 x float> %5, <32 x float> zeroinitializer, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %7 = tail call <16 x bfloat> @llvm.aie2p.v16accfloat.to.v16bf16(<16 x float> %6)
  %8 = bitcast <16 x bfloat> %7 to <8 x i32>
  %9 = shufflevector <8 x i32> %8, <8 x i32> zeroinitializer, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %10 = bitcast <16 x i32> %9 to <32 x bfloat>
  %11 = tail call <32 x float> @llvm.aie2p.I512.I512.ACC1024.bf.mul.conf(<32 x bfloat> zeroinitializer, <32 x bfloat> %10, i32 0)
  %12 = shufflevector <32 x float> %11, <32 x float> zeroinitializer, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %13 = tail call <16 x bfloat> @llvm.aie2p.v16accfloat.to.v16bf16(<16 x float> %12)
  %14 = tail call <16 x float> @llvm.aie2p.v16bf16.to.v16accfloat(<16 x bfloat> %13)
  store <16 x float> %14, ptr null, align 64
  %15 = tail call <16 x bfloat> @llvm.aie2p.v16accfloat.to.v16bf16(<16 x float> zeroinitializer)
  %16 = bitcast <16 x bfloat> %15 to <8 x i32>
  %17 = shufflevector <8 x i32> %16, <8 x i32> zeroinitializer, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %18 = bitcast <16 x i32> %17 to <32 x bfloat>
  %19 = tail call <16 x float> @llvm.aie2p.I512.I512.ACC512.bf.mul.conf(<32 x bfloat> zeroinitializer, <32 x bfloat> %18, i32 0)
  %20 = bitcast <16 x float> %19 to <8 x i64>
  %21 = shufflevector <8 x i64> %20, <8 x i64> zeroinitializer, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %22 = bitcast <32 x i64> %21 to <64 x float>
  %23 = tail call <64 x float> @llvm.aie2p.ACC2048.accfloat.add.conf(<64 x float> zeroinitializer, <64 x float> %22, i32 0)
  %24 = tail call <64 x float> @llvm.aie2p.ACC2048.accfloat.add.conf(<64 x float> %23, <64 x float> zeroinitializer, i32 0)
  %25 = tail call <64 x float> @llvm.aie2p.ACC2048.accfloat.add.conf(<64 x float> %24, <64 x float> zeroinitializer, i32 0)
  %26 = tail call <64 x float> @llvm.aie2p.ACC2048.accfloat.add.conf(<64 x float> %25, <64 x float> zeroinitializer, i32 0)
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: read)
declare <16 x bfloat> @llvm.aie2p.v16accfloat.to.v16bf16(<16 x float>) #0

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: read)
declare <64 x float> @llvm.aie2p.ACC2048.accfloat.add.conf(<64 x float>, <64 x float>, i32) #0

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: read)
declare <32 x float> @llvm.aie2p.I512.I512.ACC1024.bf.mul.conf(<32 x bfloat>, <32 x bfloat>, i32) #0

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: read)
declare <16 x float> @llvm.aie2p.I512.I512.ACC512.bf.mul.conf(<32 x bfloat>, <32 x bfloat>, i32) #0

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(none)
declare <16 x bfloat> @llvm.aie2p.tanh(<16 x float>) #1

; Function Attrs: nounwind memory(none)
declare <16 x float> @llvm.aie2p.v16bf16.to.v16accfloat(<16 x bfloat>) #2

; uselistorder directives

attributes #0 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: read) }
attributes #1 = { nocallback nofree nosync nounwind willreturn memory(none) }
attributes #2 = { nounwind memory(none) }
