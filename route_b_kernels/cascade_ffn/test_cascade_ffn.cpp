//===- test_cascade_ffn.cpp -------------------------------------*- C++ -*-===//
//
// SPDX-License-Identifier: MIT
//
// XRT correctness + dispatch-timing harness for the single-launch bf16
// Whisper-FFN cascade (Phase 0, Task 5).
//
// Kernel signature (5-BO ABI, all bf16). The generator folds the two static
// bias buffers into ONE contiguous BO (bfc1 first, bfc2 second):
//   ffn_cascade(x[768], Wfc1[3072x768], biases[3840], Wfc2[768x3072], out[768])
//   biases = bfc1.bin[3072] CONCAT bfc2.bin[768]   (offsets 0.. and 3072..)
//
// Golden buffers (IRON-free, device-faithful) live under --buffers as raw
// bf16 little-endian .bin files:
//   x.bin (768), Wfc1.bin (3072x768), bfc1.bin (3072), Wfc2.bin (768x3072),
//   bfc2.bin (768), out.bin (768 = GOLDEN output).
//
// Modes:
//   --check-only        run once, copy out back, compute rel_l2 vs out.bin,
//                       PASS if rel_l2 <= 0.08, print first-8 dev-vs-golden.
//   (default / timing)  --warmup N --iters M : time kernel(...)+run.wait()
//                       with std::chrono, skip warmup, report avg/min/max us.
//
// Mirrors the XRT pattern of mlir-air attention_decode/test_xclbin_decode.cpp:
//   device(0) -> xclbin -> register_xclbin -> hw_context -> kernel("MLIR_AIE")
//   instr BO from air.insts.bin (group_id 1); opcode 3; instr len in uint32.
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdfloat>
#include <string>
#include <vector>

#include "test_utils.h"

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

using BF16 = std::bfloat16_t;

// FFN dims (Whisper decode FFN).
static constexpr size_t D = 768;
static constexpr size_t FF = 3072;
static constexpr size_t WFC1_VOL = FF * D;  // 3072 x 768
static constexpr size_t WFC2_VOL = D * FF;  // 768 x 3072
static constexpr size_t BIAS_VOL = FF + D;  // 3840 = bfc1[3072] ++ bfc2[768]

static std::vector<BF16> load_bf16(const std::string &path, size_t n_elem) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) {
    std::cerr << "Error: cannot open " << path << "\n";
    std::exit(1);
  }
  size_t bytes = (size_t)f.tellg();
  size_t want = n_elem * sizeof(BF16);
  if (bytes != want) {
    std::cerr << "Error: " << path << " is " << bytes << " bytes, expected "
              << want << " (" << n_elem << " bf16)\n";
    std::exit(1);
  }
  f.seekg(0);
  std::vector<BF16> v(n_elem);
  f.read(reinterpret_cast<char *>(v.data()), want);
  return v;
}

static std::string opt(int argc, const char **argv, const std::string &flag,
                       const std::string &def) {
  for (int i = 1; i < argc - 1; i++)
    if (flag == argv[i])
      return argv[i + 1];
  return def;
}
static bool has(int argc, const char **argv, const std::string &flag) {
  for (int i = 1; i < argc; i++)
    if (flag == argv[i])
      return true;
  return false;
}

int main(int argc, const char **argv) {
  std::string xclbin_path =
      opt(argc, argv, "-x", opt(argc, argv, "--xclbin", ""));
  std::string instr_path =
      opt(argc, argv, "-i", opt(argc, argv, "--instr", ""));
  std::string buffers =
      opt(argc, argv, "--buffers", opt(argc, argv, "-b", ""));
  std::string kernel_name = opt(argc, argv, "--kernel", "MLIR_AIE");
  bool check_only = has(argc, argv, "--check-only");
  int n_warmup = std::stoi(opt(argc, argv, "--warmup", "20"));
  int n_iters = std::stoi(opt(argc, argv, "--iters", "200"));

  if (xclbin_path.empty() || instr_path.empty() || buffers.empty()) {
    std::cerr << "Usage: " << argv[0]
              << " [--check-only | --warmup N --iters M]"
                 " -x <xclbin> -i <insts.bin> --buffers <dir>\n";
    return 1;
  }

  // Load the golden buffers (raw bf16 LE). The two static biases (bfc1, bfc2)
  // are concatenated into ONE 3840-bf16 buffer to match the 5-BO kernel ABI:
  // biases = bfc1[3072] (offset 0) ++ bfc2[768] (offset 3072).
  std::vector<BF16> h_x = load_bf16(buffers + "/x.bin", D);
  std::vector<BF16> h_Wfc1 = load_bf16(buffers + "/Wfc1.bin", WFC1_VOL);
  std::vector<BF16> h_bfc1 = load_bf16(buffers + "/bfc1.bin", FF);
  std::vector<BF16> h_bfc2 = load_bf16(buffers + "/bfc2.bin", D);
  std::vector<BF16> h_Wfc2 = load_bf16(buffers + "/Wfc2.bin", WFC2_VOL);
  std::vector<BF16> h_golden = load_bf16(buffers + "/out.bin", D);

  std::vector<BF16> h_biases;
  h_biases.reserve(BIAS_VOL);
  h_biases.insert(h_biases.end(), h_bfc1.begin(), h_bfc1.end()); // [0:3072]
  h_biases.insert(h_biases.end(), h_bfc2.begin(), h_bfc2.end()); // [3072:3840]

  // Instruction stream.
  std::vector<uint32_t> instr_v = test_utils::load_instr_binary(instr_path);

  // XRT setup (mirror attention_decode/test_xclbin_decode.cpp).
  unsigned int device_index = 0;
  auto device = xrt::device(device_index);
  auto xclbin = xrt::xclbin(xclbin_path);

  auto xkernels = xclbin.get_kernels();
  auto it = std::find_if(xkernels.begin(), xkernels.end(),
                         [&](xrt::xclbin::kernel &k) {
                           return k.get_name().rfind(kernel_name, 0) == 0;
                         });
  if (it == xkernels.end()) {
    std::cerr << "Error: kernel '" << kernel_name << "' not found. Available:";
    for (auto &k : xkernels)
      std::cerr << "\n  - " << k.get_name();
    std::cerr << std::endl;
    return 1;
  }
  auto kernelName = it->get_name();

  device.register_xclbin(xclbin);
  xrt::hw_context context(device, xclbin.get_uuid());
  auto kernel = xrt::kernel(context, kernelName);

  // BO order (5-BO ABI): (opcode, instr[gid1], ninstr, x[gid3], Wfc1[gid4],
  //            biases[gid5], Wfc2[gid6], out[gid7]).
  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(uint32_t),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_x = xrt::bo(device, D * sizeof(BF16), XRT_BO_FLAGS_HOST_ONLY,
                      kernel.group_id(3));
  auto bo_Wfc1 = xrt::bo(device, WFC1_VOL * sizeof(BF16),
                         XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
  auto bo_biases = xrt::bo(device, BIAS_VOL * sizeof(BF16),
                           XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));
  auto bo_Wfc2 = xrt::bo(device, WFC2_VOL * sizeof(BF16),
                         XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(6));
  auto bo_out = xrt::bo(device, D * sizeof(BF16), XRT_BO_FLAGS_HOST_ONLY,
                        kernel.group_id(7));

  std::memcpy(bo_instr.map<void *>(), instr_v.data(),
              instr_v.size() * sizeof(uint32_t));
  std::memcpy(bo_x.map<void *>(), h_x.data(), D * sizeof(BF16));
  std::memcpy(bo_Wfc1.map<void *>(), h_Wfc1.data(), WFC1_VOL * sizeof(BF16));
  std::memcpy(bo_biases.map<void *>(), h_biases.data(),
              BIAS_VOL * sizeof(BF16));
  std::memcpy(bo_Wfc2.map<void *>(), h_Wfc2.data(), WFC2_VOL * sizeof(BF16));
  std::memset(bo_out.map<void *>(), 0, D * sizeof(BF16)); // zero-init output

  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_x.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_Wfc1.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_biases.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_Wfc2.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  const unsigned int opcode = 3;

  if (check_only) {
    auto run = kernel(opcode, bo_instr, instr_v.size(), bo_x, bo_Wfc1,
                      bo_biases, bo_Wfc2, bo_out);
    run.wait();
    bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    BF16 *dev = bo_out.map<BF16 *>();
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < D; i++) {
      double d = (double)(float)dev[i] - (double)(float)h_golden[i];
      num += d * d;
      double g = (double)(float)h_golden[i];
      den += g * g;
    }
    double rel_l2 = std::sqrt(num) / std::sqrt(den);

    std::cout << "first 8 (dev vs golden):\n";
    for (size_t i = 0; i < 8; i++)
      std::cout << "  [" << i << "] dev=" << (float)dev[i]
                << "  golden=" << (float)h_golden[i] << "\n";
    std::cout << "rel_l2 = " << rel_l2 << "\n";
    std::cout << (rel_l2 <= 0.08 ? "PASS" : "FAIL") << " (gate rel_l2 <= 0.08)"
              << std::endl;
    return rel_l2 <= 0.08 ? 0 : 2;
  }

  // Timing mode.
  unsigned int num_iter = n_warmup + n_iters;
  float t_total = 0, t_min = std::numeric_limits<float>::max(), t_max = 0;
  for (unsigned iter = 0; iter < num_iter; iter++) {
    auto start = std::chrono::high_resolution_clock::now();
    auto run = kernel(opcode, bo_instr, instr_v.size(), bo_x, bo_Wfc1,
                      bo_biases, bo_Wfc2, bo_out);
    run.wait();
    auto stop = std::chrono::high_resolution_clock::now();
    bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    if ((int)iter < n_warmup)
      continue;
    float us =
        std::chrono::duration_cast<std::chrono::microseconds>(stop - start)
            .count();
    t_total += us;
    t_min = std::min(t_min, us);
    t_max = std::max(t_max, us);
  }
  std::cout << "Cascade FFN dispatch (warmup=" << n_warmup
            << ", iters=" << n_iters << "):\n";
  std::cout << "  avg = " << t_total / n_iters << " us\n";
  std::cout << "  min = " << t_min << " us\n";
  std::cout << "  max = " << t_max << " us" << std::endl;
  return 0;
}
