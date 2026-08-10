/** @file r2_rig.mm
 *  FP32LIMB R2/R3 Metal rig (PRE-REG FP32LIMB-METAL; relay 2026-08-10-13 GO).
 *
 *  Modes:
 *    compile   (default) CPU-legal: create device handle, compile all
 *              kernels with fast-math OFF, run compile probes (integer
 *              simdgroup MMA, 64-bit int arithmetic). NO command buffer is
 *              ever committed in this mode.
 *    ftz       GPU: measure denormal behavior.        Requires --gpu-ok.
 *    biteq     GPU: P-KERNEL-BITEQ vs the R1 oracle.  Requires --gpu-ok.
 *    wall      GPU: R3 wall timing, both instantiations vs CPU fp64
 *              baseline, every iteration evaluated.   Requires --gpu-ok.
 *
 *  --gpu-ok is the house-ping interlock: the crown battery owns the Mac
 *  GPU until the house pings; do not pass the flag before that.
 *
 *  Build (not in CMake, per tools/ convention):
 *    clang++ -std=c++2b -ObjC++ -fobjc-arc -Iinclude \
 *        tools/fp32limb/metal/r2_rig.mm build-rel/libaxiom.a \
 *        -framework Metal -framework Foundation -o /tmp/r2_rig
 *  Run:
 *    /tmp/r2_rig compile
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <ax/la/fp32limb.hpp>
#include <ax/st/rng.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace ax::la::fp32limb;
using ax::bigint;

namespace {

// ---- compile options: the pinned-flags receipt item ------------------------
MTLCompileOptions* pinned_options() {
  MTLCompileOptions* opts = [MTLCompileOptions new];
  // fast-math OFF. On macOS 15+ the API is mathMode; fastMathEnabled is the
  // deprecated spelling — set both so the pin survives either runtime.
  if (@available(macOS 15.0, *)) {
    opts.mathMode = MTLMathModeSafe;
  }
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
  opts.fastMathEnabled = NO;
#pragma clang diagnostic pop
  return opts;
}

id<MTLLibrary> compile_source(id<MTLDevice> dev, NSString* src,
                              NSString** err_out) {
  NSError* err = nil;
  id<MTLLibrary> lib = [dev newLibraryWithSource:src
                                         options:pinned_options()
                                           error:&err];
  if (err_out) *err_out = err ? [err localizedDescription] : nil;
  return lib;
}

std::string nss(NSString* s) { return s ? std::string([s UTF8String]) : ""; }

// ---- probes (compile-only; no dispatch) ------------------------------------
void run_probes(id<MTLDevice> dev) {
  // 64-bit integer arithmetic in MSL?
  NSString* err = nil;
  compile_source(dev,
                 @"#include <metal_stdlib>\nusing namespace metal;\n"
                 @"kernel void p(device long* o [[buffer(0)]]) {"
                 @" long a = 123456789012345; o[0] = a * 3 + o[0]; }",
                 &err);
  std::printf("probe.int64_long: %s%s\n", err ? "UNSUPPORTED: " : "SUPPORTED",
              err ? nss(err).c_str() : "");

  // integer simdgroup MMA? (the banked question — settles whether the
  // int8-MMA port is superseded or merely deferred)
  compile_source(dev,
                 @"#include <metal_stdlib>\nusing namespace metal;\n"
                 @"kernel void p() { simdgroup_matrix<int, 8, 8> m; }",
                 &err);
  std::printf("probe.simdgroup_matrix_int: %s\n",
              err ? ("UNSUPPORTED: " + nss(err)).c_str() : "SUPPORTED");

  // control: float simdgroup_matrix (should compile — validates the probe)
  compile_source(dev,
                 @"#include <metal_stdlib>\nusing namespace metal;\n"
                 @"kernel void p() { simdgroup_float8x8 m; }",
                 &err);
  std::printf("probe.simdgroup_matrix_float_control: %s\n",
              err ? ("UNSUPPORTED: " + nss(err)).c_str() : "SUPPORTED");
}

// ---- shared host-side helpers ----------------------------------------------
struct padded_slices {
  std::vector<float> data;  // [count][MAX_SLICES][BLOCK], zero-padded
  std::vector<int> e_align;
};

// slice a full row/col of length K into per-block padded slices
padded_slices slice_axis(const matf& m, bool rows) {
  const int outer = rows ? m.rows : m.cols;
  const int K = rows ? m.cols : m.rows;
  const int nblk = (K + BLOCK - 1) / BLOCK;
  padded_slices ps;
  ps.data.assign(static_cast<std::size_t>(outer) * nblk * MAX_SLICES * BLOCK,
                 0.0f);
  ps.e_align.assign(static_cast<std::size_t>(outer) * nblk, 0);
  std::vector<float> seg(BLOCK);
  for (int i = 0; i < outer; ++i)
    for (int b0 = 0, blk = 0; b0 < K; b0 += BLOCK, ++blk) {
      const int n = std::min(BLOCK, K - b0);
      for (int k = 0; k < n; ++k)
        seg[static_cast<std::size_t>(k)] =
            rows ? m.at(i, b0 + k) : m.at(b0 + k, i);
      for (int k = n; k < BLOCK; ++k) seg[static_cast<std::size_t>(k)] = 0.0f;
      const sliced s = slice_row(seg.data(), BLOCK);
      const std::size_t base =
          (static_cast<std::size_t>(i) * nblk + blk) * MAX_SLICES * BLOCK;
      for (std::size_t p = 0; p < s.sl.size(); ++p)
        std::memcpy(&ps.data[base + p * BLOCK], s.sl[p].data(),
                    BLOCK * sizeof(float));
      ps.e_align[static_cast<std::size_t>(i) * nblk + blk] = s.e_align;
    }
  return ps;
}

matf random_matf_f24(int r, int c, ax::st::rng& g, double scale, bool normal) {
  matf m(r, c);
  const double flush = scale * std::ldexp(1.0, -24);
  for (auto& v : m.v) {
    const double d = (normal ? g.normal() : g.uniform(-1.0, 1.0)) * scale;
    v = std::fabs(d) < flush ? 0.0f : static_cast<float>(d);
  }
  return m;
}

id<MTLComputePipelineState> pipeline(id<MTLDevice> dev, id<MTLLibrary> lib,
                                     NSString* name) {
  NSError* err = nil;
  id<MTLComputePipelineState> ps = [dev
      newComputePipelineStateWithFunction:[lib newFunctionWithName:name]
                                    error:&err];
  if (!ps) {
    std::fprintf(stderr, "pipeline %s: %s\n", nss(name).c_str(),
                 nss([err localizedDescription]).c_str());
    std::exit(1);
  }
  return ps;
}

id<MTLBuffer> make_buf(id<MTLDevice> dev, const void* p, std::size_t bytes) {
  return p ? [dev newBufferWithBytes:p
                              length:bytes
                             options:MTLResourceStorageModeShared]
           : [dev newBufferWithLength:bytes
                              options:MTLResourceStorageModeShared];
}

double median_of(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

// ---- ftz mode (GPU) --------------------------------------------------------
int mode_ftz(id<MTLDevice> dev, id<MTLLibrary> lib) {
  const float den = std::ldexp(1.0f, -140);
  const float in[3] = {den, std::ldexp(1.0f, -126),
                       std::ldexp(1.017f, -126)};
  id<MTLBuffer> bin = make_buf(dev, in, sizeof in);
  id<MTLBuffer> bout = make_buf(dev, nullptr, 4 * sizeof(float));
  id<MTLCommandQueue> q = [dev newCommandQueue];
  id<MTLCommandBuffer> cb = [q commandBuffer];
  id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
  [enc setComputePipelineState:pipeline(dev, lib, @"ftz_probe")];
  [enc setBuffer:bin offset:0 atIndex:0];
  [enc setBuffer:bout offset:0 atIndex:1];
  [enc dispatchThreads:MTLSizeMake(1, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
  [enc endEncoding];
  [cb commit];
  [cb waitUntilCompleted];
  const float* o = static_cast<const float*>(bout.contents);
  const float exp0 = std::ldexp(1.0f, -141), exp1 = std::ldexp(1.0f, -139);
  const float exp3 = in[1] - in[2];  // CPU (no FTZ) reference
  std::printf("ftz.mul_half:  got %a expect %a  %s\n", o[0], exp0,
              o[0] == exp0 ? "PRESERVED" : "FLUSHED");
  std::printf("ftz.add_self:  got %a expect %a  %s\n", o[1], exp1,
              o[1] == exp1 ? "PRESERVED" : "FLUSHED");
  std::printf("ftz.mul_ident: got %a expect %a  %s\n", o[2], den,
              o[2] == den ? "PRESERVED" : "FLUSHED");
  std::printf("ftz.sub_makes: got %a expect %a  %s\n", o[3], exp3,
              o[3] == exp3 ? "PRESERVED" : "FLUSHED");
  const bool clean = o[0] == exp0 && o[1] == exp1 && o[2] == den &&
                     o[3] == exp3;
  std::printf("ftz.verdict: %s\n",
              clean ? "NO-FTZ (no range restriction needed)"
                    : "FTZ-PRESENT (range restriction REQUIRED)");
  return clean ? 0 : 2;
}

// ---- biteq mode (GPU): P-KERNEL-BITEQ --------------------------------------
int mode_biteq(id<MTLDevice> dev, id<MTLLibrary> lib) {
  int fails = 0;
  for (unsigned long long seed : {1ULL, 2ULL, 3ULL}) {
    const int N = 64;
    ax::st::rng g(seed);
    const matf a = random_matf_f24(N, N, g, 1.0, false);
    const matf b = random_matf_f24(N, N, g, 1.0, false);
    const padded_slices pa = slice_axis(a, true);
    const padded_slices pb = slice_axis(b, false);
    const int nblk = (N + BLOCK - 1) / BLOCK;
    // build per-(i,j,blk) threadgroup inputs by gathering row/col slices
    const std::size_t ntg =
        static_cast<std::size_t>(N) * N * nblk;
    const std::size_t seg = static_cast<std::size_t>(MAX_SLICES) * BLOCK;
    std::vector<float> ga(ntg * seg), gb(ntg * seg);
    std::size_t t = 0;
    for (int i = 0; i < N; ++i)
      for (int j = 0; j < N; ++j)
        for (int blk = 0; blk < nblk; ++blk, ++t) {
          std::memcpy(&ga[t * seg],
                      &pa.data[(static_cast<std::size_t>(i) * nblk + blk) * seg],
                      seg * sizeof(float));
          std::memcpy(&gb[t * seg],
                      &pb.data[(static_cast<std::size_t>(j) * nblk + blk) * seg],
                      seg * sizeof(float));
        }
    id<MTLBuffer> ba = make_buf(dev, ga.data(), ga.size() * sizeof(float));
    id<MTLBuffer> bb = make_buf(dev, gb.data(), gb.size() * sizeof(float));
    id<MTLBuffer> bp = make_buf(
        dev, nullptr, ntg * MAX_SLICES * MAX_SLICES * sizeof(float));
    unsigned int zero = 0;
    id<MTLBuffer> bo = make_buf(dev, &zero, sizeof zero);
    id<MTLCommandQueue> q = [dev newCommandQueue];
    id<MTLCommandBuffer> cb = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pipeline(dev, lib, @"r2_slicepair_dots")];
    [enc setBuffer:ba offset:0 atIndex:0];
    [enc setBuffer:bb offset:0 atIndex:1];
    [enc setBuffer:bp offset:0 atIndex:2];
    [enc setBuffer:bo offset:0 atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(ntg, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(BLOCK, 1, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if (*static_cast<const unsigned int*>(bo.contents) != 0) {
      std::printf("biteq seed=%llu OVERFLOW FLAG\n", seed);
      ++fails;
      continue;
    }
    // recombine kernel partials through the exact bigint path and compare
    // to gemm_ref — the immovable integer oracle. Also bit-compare each
    // partial float against the CPU-recomputed partial.
    const float* part = static_cast<const float*>(bp.contents);
    const auto ref = gemm_ref(a, b);
    std::size_t bad = 0;
    t = 0;
    for (int i = 0; i < N && bad == 0; ++i)
      for (int j = 0; j < N && bad == 0; ++j) {
        dyadic out{bigint(0), 0};
        for (int blk = 0; blk < nblk; ++blk, ++t) {
          const int ea = pa.e_align[static_cast<std::size_t>(i) * nblk + blk];
          const int eb = pb.e_align[static_cast<std::size_t>(j) * nblk + blk];
          const float* asl2 =
              &pa.data[(static_cast<std::size_t>(i) * nblk + blk) * seg];
          const float* bsl2 =
              &pb.data[(static_cast<std::size_t>(j) * nblk + blk) * seg];
          for (int p = 0; p < MAX_SLICES; ++p)
            for (int qq = 0; qq < MAX_SLICES; ++qq) {
              // CPU recompute of the same fp32 partial, bit-compare
              float cacc = 0.0f;
              for (int k = 0; k < BLOCK; ++k)
                cacc += asl2[p * BLOCK + k] * bsl2[qq * BLOCK + k];
              const float gacc =
                  part[t * MAX_SLICES * MAX_SLICES + p * MAX_SLICES + qq];
              if (std::memcmp(&cacc, &gacc, sizeof(float)) != 0) ++bad;
              acc(out, bigint(std::llround(static_cast<double>(gacc))),
                  -SLICE_W * (p + 1) - SLICE_W * (qq + 1) + ea + eb);
            }
        }
        if (!dyadic_eq(out, ref[static_cast<std::size_t>(i) * N + j])) ++bad;
      }
    std::printf("biteq seed=%llu n=%d %s (bad=%zu)\n", seed, N,
                bad == 0 ? "BIT-IDENTICAL" : "MISMATCH", bad);
    if (bad) ++fails;
  }
  return fails == 0 ? 0 : 1;
}

// ---- wall mode (GPU): R3 timing, both instantiations -----------------------
int mode_wall(id<MTLDevice> dev, id<MTLLibrary> lib) {
  const int N = 512, reps = 7;
  const int nblk = N / BLOCK;
  ax::st::rng g(20260810);
  const matf a = random_matf_f24(N, N, g, 1.0, false);
  const matf b = random_matf_f24(N, N, g, 1.0, false);
  // CPU fp64 baseline, matched N, every iteration evaluated
  std::vector<double> ad(a.v.begin(), a.v.end()), bd(b.v.begin(), b.v.end());
  std::vector<double> cd(static_cast<std::size_t>(N) * N);
  std::vector<double> t_cpu;
  volatile double sink = 0.0;
  for (int r = 0; r < reps; ++r) {
    const auto t0 = [NSDate date];
    for (int i = 0; i < N; ++i)
      for (int j = 0; j < N; ++j) {
        double s = 0.0;
        for (int k = 0; k < N; ++k)
          s += ad[static_cast<std::size_t>(i) * N + k] *
               bd[static_cast<std::size_t>(k) * N + j];
        cd[static_cast<std::size_t>(i) * N + j] = s;
      }
    sink = sink + cd[0];
    t_cpu.push_back(-[t0 timeIntervalSinceNow]);
  }
  // fp32-limb tiled instantiation
  const padded_slices pa = slice_axis(a, true);
  const padded_slices pb = slice_axis(b, false);
  id<MTLBuffer> ba = make_buf(dev, pa.data.data(),
                              pa.data.size() * sizeof(float));
  id<MTLBuffer> bb = make_buf(dev, pb.data.data(),
                              pb.data.size() * sizeof(float));
  id<MTLBuffer> bp = make_buf(dev, nullptr,
                              static_cast<std::size_t>(N) * N * nblk *
                                  MAX_SLICES * MAX_SLICES * sizeof(float));
  unsigned int zero = 0;
  id<MTLBuffer> bo = make_buf(dev, &zero, sizeof zero);
  const unsigned int rows = N, cols = N, nb = nblk;
  id<MTLCommandQueue> q = [dev newCommandQueue];
  auto run_kernel = [&](NSString* name, id<MTLBuffer> in0, id<MTLBuffer> in1,
                        id<MTLBuffer> out) {
    std::vector<double> ts;
    id<MTLComputePipelineState> ps = pipeline(dev, lib, name);
    for (int r = 0; r < reps; ++r) {
      id<MTLCommandBuffer> cb = [q commandBuffer];
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      [enc setComputePipelineState:ps];
      [enc setBuffer:in0 offset:0 atIndex:0];
      [enc setBuffer:in1 offset:0 atIndex:1];
      [enc setBuffer:out offset:0 atIndex:2];
      [enc setBuffer:bo offset:0 atIndex:3];
      [enc setBytes:&rows length:4 atIndex:4];
      [enc setBytes:&cols length:4 atIndex:5];
      [enc setBytes:&nb length:4 atIndex:6];
      [enc dispatchThreadgroups:MTLSizeMake((N + 7) / 8, (N + 7) / 8, 1)
          threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
      [enc endEncoding];
      const auto t0 = [NSDate date];
      [cb commit];
      [cb waitUntilCompleted];  // every iteration evaluated — no lazy graphs
      ts.push_back(-[t0 timeIntervalSinceNow]);
    }
    return ts;
  };
  const auto t_limb = run_kernel(@"r3_fp32limb_tiled", ba, bb, bp);
  // integer instantiation (aligned per-matrix mantissas; global alignment
  // keeps the demo exact for _f24 unit-scale inputs)
  std::vector<int> am(static_cast<std::size_t>(N) * N),
      bmv(static_cast<std::size_t>(N) * N);
  for (std::size_t t = 0; t < am.size(); ++t)
    am[t] = static_cast<int>(std::llround(std::ldexp(double(a.v[t]), 24)));
  for (std::size_t t = 0; t < bmv.size(); ++t)
    bmv[t] = static_cast<int>(std::llround(std::ldexp(double(b.v[t]), 24)));
  std::vector<int> bmt(bmv.size());  // transpose so kernel reads [cols][K]
  for (int k = 0; k < N; ++k)
    for (int j = 0; j < N; ++j)
      bmt[static_cast<std::size_t>(j) * N + k] =
          bmv[static_cast<std::size_t>(k) * N + j];
  id<MTLBuffer> bam = make_buf(dev, am.data(), am.size() * sizeof(int));
  id<MTLBuffer> bbm = make_buf(dev, bmt.data(), bmt.size() * sizeof(int));
  id<MTLBuffer> bacc = make_buf(
      dev, nullptr, static_cast<std::size_t>(N) * N * nblk * sizeof(long long));
  const auto t_int = run_kernel(@"r3_intacc_tiled", bam, bbm, bacc);
  const double mc = median_of(t_cpu), ml = median_of(t_limb),
               mi = median_of(t_int);
  auto spread = [](std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v.back() - v.front();
  };
  std::printf("wall.cpu_fp64:  median %.4fs spread %.4fs\n", mc,
              spread(t_cpu));
  std::printf("wall.fp32limb:  median %.4fs spread %.4fs ratio %.3fx %s\n",
              ml, spread(t_limb), ml / mc,
              ml / mc <= 1.07 ? "PASS<=1.07" : "MISS");
  std::printf("wall.intacc:    median %.4fs spread %.4fs ratio %.3fx %s\n",
              mi, spread(t_int), mi / mc,
              mi / mc <= 1.07 ? "PASS<=1.07" : "MISS");
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  @autoreleasepool {
    const std::string mode = argc > 1 ? argv[1] : "compile";
    bool gpu_ok = false;
    for (int i = 2; i < argc; ++i)
      if (std::string(argv[i]) == "--gpu-ok") gpu_ok = true;
    if (mode != "compile" && !gpu_ok) {
      std::fprintf(stderr,
                   "mode '%s' dispatches on the GPU; the crown battery owns "
                   "it until the house ping. Pass --gpu-ok only after the "
                   "ping.\n",
                   mode.c_str());
      return 3;
    }
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (!dev) {
      std::fprintf(stderr, "no Metal device\n");
      return 1;
    }
    std::printf("device: %s\n", nss(dev.name).c_str());
    NSError* ferr = nil;
    NSString* src = [NSString
        stringWithContentsOfFile:@"tools/fp32limb/metal/fp32limb_kernels.metal"
                        encoding:NSUTF8StringEncoding
                           error:&ferr];
    if (!src) {
      std::fprintf(stderr, "kernel source: %s (run from repo root)\n",
                   nss([ferr localizedDescription]).c_str());
      return 1;
    }
    NSString* cerr = nil;
    id<MTLLibrary> lib = compile_source(dev, src, &cerr);
    std::printf("compile.fast_math: OFF (MTLMathModeSafe + "
                "fastMathEnabled=NO)\n");
    std::printf("compile.kernels: %s\n",
                lib ? "OK" : ("FAIL: " + nss(cerr)).c_str());
    if (!lib) return 1;
    if (mode == "compile") {
      run_probes(dev);
      std::printf("compile mode done; no command buffer committed.\n");
      return 0;
    }
    if (mode == "ftz") return mode_ftz(dev, lib);
    if (mode == "biteq") return mode_biteq(dev, lib);
    if (mode == "wall") return mode_wall(dev, lib);
    std::fprintf(stderr, "unknown mode %s\n", mode.c_str());
    return 1;
  }
}
