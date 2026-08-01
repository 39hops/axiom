/** pybind11 module `intbirth` — the integer-birth engine exposed to
    house Python (relay 2026-08-01 engine ask + primitives follow-up).

    Two layers, same integers:
      FullBirth                 — the composed, certified R2b loop
      Block / AdamW / int_gemm* — the primitives, for house-side
                                  composition (multi-block etc.)
    FullBirth is implemented ON the primitives in C++, so both layers
    are certified by the same r2b_ref.json digests. Protocol
    unchanged: tables/init are opaque bytes, the contract dict
    carries the r2b_ref.json constants, digests engine-side.

    Array convention: int64 numpy arrays, row-major. Weight dicts use
    the KEYS names (wq wk wv wo wg wu wd wh g1 g2 g3), Q scale for
    Block, Q_w scale for AdamW params. AdamW.step mutates the param
    arrays in place (the optimizer owns m/v state, keyed by index).
    rdiv placement is the caller's, as in the reference. */
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <ax/nn/intbirth.hpp>

#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace py = pybind11;
using ax::nn::ib::adamw;
using ax::nn::ib::block;
using ax::nn::ib::block_cache;
using ax::nn::ib::contract;
using ax::nn::ib::full_birth;
using ax::nn::ib::i64;
using ax::nn::ib::Mat;

namespace {

using Arr = py::array_t<i64, py::array::c_style | py::array::forcecast>;

contract contract_from_dict(const py::dict& d) {
  contract c;
  const auto geti = [&](const char* k, int dflt) {
    return d.contains(k) ? d[k].cast<int>() : dflt;
  };
  const auto getl = [&](const char* k, long long dflt) {
    return d.contains(k) ? d[k].cast<long long>() : dflt;
  };
  c.T = geti("T", c.T);
  c.D = geti("D", c.D);
  c.DH = geti("DH", c.DH);
  c.F = geti("F", c.F);
  c.V = geti("V", c.V);
  c.shift = geti("SHIFT", c.shift);
  c.gboost = getl("GBOOST", c.gboost);
  c.pq = getl("PQ", c.pq);
  c.act_clamp = getl("ACT_CLAMP", c.act_clamp);
  c.eps32 = getl("EPS32", c.eps32);
  c.lrn = getl("LRN", c.lrn);
  c.lrd = getl("LRD", c.lrd);
  return c;
}

Mat to_mat(const Arr& a) {
  Mat m(std::size_t(a.size()));
  std::memcpy(m.data(), a.data(), m.size() * 8);
  return m;
}
Arr to_arr(const Mat& m, const std::vector<py::ssize_t>& shape) {
  Arr a(shape);
  std::memcpy(a.mutable_data(), m.data(), m.size() * 8);
  return a;
}

std::map<std::string, std::vector<py::ssize_t>> key_shapes(
    const contract& c) {
  return {{"wq", {c.DH, c.D}}, {"wk", {c.DH, c.D}},
          {"wv", {c.DH, c.D}}, {"wo", {c.D, c.DH}},
          {"wg", {c.F, c.D}},  {"wu", {c.F, c.D}},
          {"wd", {c.D, c.F}},  {"wh", {c.V, c.D}},
          {"g1", {c.D}},       {"g2", {c.D}},
          {"g3", {c.D}}};
}

std::map<std::string, Mat> weights_from_dict(const py::dict& d) {
  std::map<std::string, Mat> w;
  for (const char* k : block::KEYS) {
    if (!d.contains(k))
      throw std::runtime_error(std::string("missing weight ") + k);
    w[k] = to_mat(d[k].cast<Arr>());
  }
  return w;
}

py::dict grads_to_dict(const std::map<std::string, Mat>& g,
                       const contract& c) {
  const auto sh = key_shapes(c);
  py::dict out;
  for (const auto& [k, m] : g) out[k.c_str()] = to_arr(m, sh.at(k));
  return out;
}

}  // namespace

PYBIND11_MODULE(intbirth, m) {
  m.doc() = "axiom integer-birth engine (composed FullBirth + the "
            "Block/AdamW/int_gemm primitives; digests engine-side, "
            "comparison house-side)";

  // ---- primitives: int_gemm forms (exact int64; caller rounds)
  m.def("int_gemm",
        [](const Arr& a, const Arr& w) {
          if (a.ndim() != 2 || w.ndim() != 2 ||
              a.shape(1) != w.shape(1))
            throw std::runtime_error("int_gemm: want a[r,K], w[N,K]");
          return to_arr(ax::nn::ib::int_gemm(
                            to_mat(a), int(a.shape(0)),
                            int(a.shape(1)), to_mat(w),
                            int(w.shape(0))),
                        {a.shape(0), w.shape(0)});
        },
        "a[r,K] @ w[N,K]^T -> [r,N] (the torch int_mm convention)");
  m.def("int_gemm_nt",
        [](const Arr& a, const Arr& w) {
          if (a.ndim() != 2 || w.ndim() != 2 ||
              a.shape(1) != w.shape(0))
            throw std::runtime_error(
                "int_gemm_nt: want a[r,K], w[K,N]");
          return to_arr(ax::nn::ib::int_gemm_nt(
                            to_mat(a), int(a.shape(0)),
                            int(a.shape(1)), to_mat(w),
                            int(w.shape(1))),
                        {a.shape(0), w.shape(1)});
        },
        "a[r,K] @ w[K,N] -> [r,N]");
  m.def("int_gemm_xty",
        [](const Arr& x, const Arr& y) {
          if (x.ndim() != 2 || y.ndim() != 2 ||
              x.shape(0) != y.shape(0))
            throw std::runtime_error(
                "int_gemm_xty: want x[r,K], y[r,N]");
          return to_arr(ax::nn::ib::int_gemm_xty(
                            to_mat(x), int(x.shape(0)),
                            int(x.shape(1)), to_mat(y),
                            int(y.shape(1))),
                        {x.shape(1), y.shape(1)});
        },
        "x[r,K]^T @ y[r,N] -> [K,N] (the dW outer form)");
  m.def("rdiv",
        [](const Arr& a, long long d) {
          Mat v = to_mat(a);
          ax::nn::ib::rdiv_inplace(v, d);
          std::vector<py::ssize_t> shape(a.shape(),
                                         a.shape() + a.ndim());
          return to_arr(v, shape);
        },
        "round-half-away division, elementwise (the program rdiv)");

  // ---- primitives: the block + optimizer
  py::class_<block_cache>(m, "BlockCache",
                          "opaque forward cache for Block.bwd");
  py::class_<block>(m, "Block")
      .def(py::init([](const py::bytes& tables, const py::dict& c) {
             return new block(std::string(tables),
                              contract_from_dict(c));
           }),
           py::arg("tables_bytes"), py::arg("contract"))
      .def("fwd",
           [](const block& b, const py::dict& w, const Arr& x) {
             auto wm = weights_from_dict(w);
             auto cache = new block_cache();
             const Mat logits = b.fwd(wm, to_mat(x), *cache);
             return py::make_tuple(
                 to_arr(logits, {b.cfg().T, b.cfg().V}),
                 py::cast(cache,
                          py::return_value_policy::take_ownership));
           },
           py::arg("weights"), py::arg("x"),
           "(logits [T,V], cache); weights at Q scale, KEYS names")
      .def("bwd",
           [](const block& b, const py::dict& w, const Arr& dlogits,
              const block_cache& cache) {
             auto wm = weights_from_dict(w);
             Mat dx0;
             auto G = b.bwd(wm, to_mat(dlogits), cache, &dx0);
             return py::make_tuple(
                 grads_to_dict(G, b.cfg()),
                 to_arr(dx0, {b.cfg().T, b.cfg().D}));
           },
           py::arg("weights"), py::arg("dlogits"), py::arg("cache"),
           "(grads dict at boosted scale, dx0 [T,D] — the "
           "multi-block chain point)")
      .def("softmax_rows",
           [](const block& b, const Arr& s, long long scale) {
             if (s.ndim() != 2)
               throw std::runtime_error("softmax_rows: want 2-D");
             return to_arr(
                 b.softmax_rows(to_mat(s), int(s.shape(0)),
                                int(s.shape(1)), scale),
                 {s.shape(0), s.shape(1)});
           },
           py::arg("s"), py::arg("scale"),
           "integer row softmax at `scale` units (shipped exp table)");

  py::class_<adamw>(m, "AdamW")
      .def(py::init<int, long long, long long>(), py::arg("shift"),
           py::arg("lrn") = 1, py::arg("lrd") = 1000)
      .def("step",
           [](adamw& o, const py::list& params, const py::list& grads) {
             std::vector<Mat> pv, gv;
             std::vector<Arr> arrs;
             for (const auto& p : params) {
               arrs.push_back(p.cast<Arr>());
               pv.push_back(to_mat(arrs.back()));
             }
             for (const auto& g : grads) gv.push_back(to_mat(g.cast<Arr>()));
             std::vector<Mat*> pp;
             std::vector<const Mat*> gp;
             for (auto& v : pv) pp.push_back(&v);
             for (auto& v : gv) gp.push_back(&v);
             o.step(pp, gp);
             for (std::size_t i = 0; i < pv.size(); i++)  // write back
               std::memcpy(arrs[i].mutable_data(), pv[i].data(),
                           pv[i].size() * 8);
           },
           py::arg("params"), py::arg("grads"),
           "one IntAdamWQw step; params (Q_w scale) mutated in "
           "place, grads at the unboosted Q scale")
      .def_property_readonly("nz", &adamw::nz_last)
      .def_property_readonly("step_count", &adamw::step_count);

  // ---- the composed loop
  py::class_<full_birth>(m, "FullBirth")
      .def(py::init([](const py::bytes& tables, const py::bytes& init,
                       const py::dict& c) {
             return new full_birth(std::string(tables),
                                   std::string(init),
                                   contract_from_dict(c));
           }),
           py::arg("tables_bytes"), py::arg("init_bytes"),
           py::arg("contract"))
      .def("run", &full_birth::run, py::arg("steps"),
           py::call_guard<py::gil_scoped_release>())
      .def("mark", &full_birth::mark)
      .def("traj_sha", &full_birth::traj_sha)
      .def("milestone_sha", &full_birth::traj_sha,
           "alias of traj_sha (the relay-proposed name)")
      .def("weights_bytes",
           [](const full_birth& fb) {
             return py::bytes(fb.weights_bytes());
           })
      .def_property_readonly("step_count", &full_birth::step_count)
      .def_property_readonly("loss", &full_birth::last_loss)
      .def_property_readonly("nz", &full_birth::nz_last);
}
