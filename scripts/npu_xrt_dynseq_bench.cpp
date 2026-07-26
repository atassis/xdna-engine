// Copyright (C) 2026. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Throughput/latency benchmark host for the dyn-seq dynamic whole-array matmul
// (mlir-aie PR #3368). Companion to npu_xrt_dynseq_harness.sh; compiled the same
// way as the PR's test.cpp but times the two costs that decide whether the
// runtime-built path is "free" vs a shape-specialized static xclbin:
//
//   t_build    host cost of generate_txn_main_sequence(M,K,N) -- the dynamic
//              path's ONLY extra cost (the static path bakes this at compile
//              time). Measured as the median over many rebuilds.
//   t_dispatch on-device kernel(...) + wait() with a PREBUILT stream. The PR
//              proves the runtime-built stream is register-equivalent to a
//              static build, so dispatching a prebuilt copy of it IS the
//              static-baked path's latency. Median over many warm dispatches.
//
// Reported per (M,K,N): insts, t_build (us), t_dispatch (us), and the effective
// GFLOP/s of the dispatch (2*M*N*K / t_dispatch). If t_build << t_dispatch the
// dynamic path is ~free vs static; if comparable it is a real tax on tiny shapes.
//
// Build (see the `bench` subcommand in npu_xrt_dynseq_harness.sh):
//   host CXX bench.cpp test_utils.cpp -DGEN_HDR=... -DXCLBIN=... <includes> <xrt>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <optional>
#include <vector>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#include "common.h"

#include GEN_HDR

#ifndef XCLBIN
#define XCLBIN std::string("final.xclbin")
#endif
#ifndef KERNEL_NAME
#define KERNEL_NAME "MLIR_AIE"
#endif

using A_DTYPE = std::int16_t;
using B_DTYPE = std::int16_t;
using C_DTYPE = std::int16_t;
using ACC_DTYPE = std::int16_t;

using clk = std::chrono::steady_clock;
static double us_since(clk::time_point t0) {
  return std::chrono::duration<double, std::micro>(clk::now() - t0).count();
}
static double median(std::vector<double> &v) {
  std::sort(v.begin(), v.end());
  return v.empty() ? 0.0 : v[v.size() / 2];
}

int main(int argc, const char *argv[]) {
  int M = (argc > 1) ? std::atoi(argv[1]) : 512;
  int K = (argc > 2) ? std::atoi(argv[2]) : 512;
  int N = (argc > 3) ? std::atoi(argv[3]) : 512;
  int iters = (argc > 4) ? std::atoi(argv[4]) : 200;
  int warmup = (argc > 5) ? std::atoi(argv[5]) : 20;

  // --- host txn-build cost (the dynamic path's extra tax) ---
  std::vector<double> build_us;
  build_us.reserve(iters);
  std::optional<std::vector<uint32_t>> instr_opt;
  for (int i = 0; i < iters; ++i) {
    auto t0 = clk::now();
    instr_opt = generate_txn_main_sequence(M, K, N);
    build_us.push_back(us_since(t0));
    if (!instr_opt) {
      std::cout << "builder returned nullopt for M=" << M << " K=" << K
                << " N=" << N << "\n";
      return 1;
    }
  }
  std::vector<uint32_t> instr_v = std::move(*instr_opt);

  unsigned int device_index = 0;
  xrt::device device = xrt::device(device_index);
  xrt::xclbin xclbin = xrt::xclbin(XCLBIN);
  std::vector<xrt::xclbin::kernel> xkernels = xclbin.get_kernels();
  auto xkernel = std::find_if(
      xkernels.begin(), xkernels.end(), [](xrt::xclbin::kernel &k) {
        return k.get_name().rfind(KERNEL_NAME, 0) == 0;
      });
  if (xkernel == xkernels.end()) {
    std::cout << "no kernel matching '" << KERNEL_NAME << "'\n";
    return 1;
  }
  std::string kernel_name = xkernel->get_name();
  device.register_xclbin(xclbin);
  xrt::hw_context context(device, xclbin.get_uuid());
  auto kernel = xrt::kernel(context, kernel_name);

  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(instr_v[0]),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_a = xrt::bo(device, M * K * sizeof(A_DTYPE), XRT_BO_FLAGS_HOST_ONLY,
                      kernel.group_id(3));
  auto bo_b = xrt::bo(device, K * N * sizeof(B_DTYPE), XRT_BO_FLAGS_HOST_ONLY,
                      kernel.group_id(4));
  auto bo_c = xrt::bo(device, M * N * sizeof(C_DTYPE), XRT_BO_FLAGS_HOST_ONLY,
                      kernel.group_id(5));

  std::vector<A_DTYPE> A_vec(M * K), B_vec(K * N);
  for (auto &v : A_vec)
    v = matmul_common::get_random<A_DTYPE>();
  for (auto &v : B_vec)
    v = matmul_common::get_random<B_DTYPE>();
  std::memcpy(bo_a.map<A_DTYPE *>(), A_vec.data(), M * K * sizeof(A_DTYPE));
  std::memcpy(bo_b.map<B_DTYPE *>(), B_vec.data(), K * N * sizeof(B_DTYPE));
  std::memset(bo_c.map<C_DTYPE *>(), 0, M * N * sizeof(C_DTYPE));
  std::memcpy(bo_instr.map<void *>(), instr_v.data(),
              instr_v.size() * sizeof(instr_v[0]));
  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_c.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  unsigned int opcode = 3;
  // --- on-device dispatch latency (prebuilt stream == static-baked path) ---
  for (int i = 0; i < warmup; ++i) {
    auto run = kernel(opcode, bo_instr, instr_v.size(), bo_a, bo_b, bo_c);
    if (run.wait() != ERT_CMD_STATE_COMPLETED) {
      std::cout << "warmup dispatch did not complete\n";
      return 1;
    }
  }
  std::vector<double> disp_us;
  disp_us.reserve(iters);
  for (int i = 0; i < iters; ++i) {
    auto t0 = clk::now();
    auto run = kernel(opcode, bo_instr, instr_v.size(), bo_a, bo_b, bo_c);
    ert_cmd_state r = run.wait();
    disp_us.push_back(us_since(t0));
    if (r != ERT_CMD_STATE_COMPLETED) {
      std::cout << "dispatch did not complete\n";
      return 1;
    }
  }

  // correctness guard (one verified result)
  bo_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  std::vector<C_DTYPE> C_vec(M * N);
  std::memcpy(C_vec.data(), bo_c.map<C_DTYPE *>(), M * N * sizeof(C_DTYPE));
  int n_errors = matmul_common::verify<A_DTYPE, C_DTYPE, ACC_DTYPE>(
      M, N, K, A_vec, B_vec, C_vec, 0);

  double t_build = median(build_us);
  double t_disp = median(disp_us);
  double gflops = (2.0 * M * N * K) / (t_disp * 1e3); // 2MNK flops / (us*1e3 ns)... = flops/ns = GFLOP/s
  std::cout << "BENCH M=" << M << " K=" << K << " N=" << N
            << " insts=" << instr_v.size() << " t_build_us=" << t_build
            << " t_dispatch_us=" << t_disp << " gflops=" << gflops
            << " build_frac=" << (t_build / (t_build + t_disp))
            << " errors=" << n_errors << (n_errors ? " FAIL" : " OK") << "\n";
  return n_errors == 0 ? 0 : 1;
}
