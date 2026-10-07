// python_api.cpp — mxmegakernel._C 扩展入口（pybind11）。
//
// 真实绑定：布局计算、平台显存/IPC、SymmBuffer 组装、fp8_mega_moe 发射、
// mini_mega 裸 kernel 探针（mega_launch_empty / fill_weights_lane）。
#include <torch/python.h>

#include <c10/cuda/CUDAStream.h>

#include "apis/layout.hpp"
#include "apis/mega.hpp"
#include "memory.hpp"

// flags_bridge.cu 导出（C ABI）
extern "C" {
int64_t mxk_local_flags_bytes();
int64_t mxk_ranks_flags_bytes();
int64_t mxk_alloc_local_flags(int device);
int64_t mxk_alloc_ranks_flags_fg(int device);
}

namespace mxk {
void launch_empty(int grid, int threads, int64_t smem_bytes, void* stream);
void fill_weights_lane(uint8_t* w, int n_rows, int k_cols, uint32_t seed,
                       void* stream);
std::vector<std::string> bridge_symbols();
}  // namespace mxk

namespace {

[[noreturn]] void not_implemented(const std::string& name) {
  throw std::runtime_error(
      std::string("mxmegakernel not_implemented: ") + name +
      "（mini_mega 仅实现 fp8xfp8 生产路径；详见 torch/README_torch.md）");
}

void* current_stream() {
  return reinterpret_cast<void*>(c10::cuda::getCurrentCUDAStream().stream());
}

torch::Tensor wrap_device_ptr(int64_t ptr, int64_t nbytes, int64_t device) {
  if (ptr == 0 || nbytes <= 0) {
    throw std::runtime_error("wrap_device_ptr: 指针/长度非法");
  }
  auto opts = torch::TensorOptions()
                  .dtype(torch::kUInt8)
                  .device(torch::kCUDA, device);
  return torch::from_blob(reinterpret_cast<void*>(ptr), {nbytes}, opts);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "mini_mega torch 层 _C：布局 + 平台显存/IPC + fp8_mega_moe 发射";

  // ---------------- host 布局 ----------------
  m.def("get_symm_layout_for_mega_moe",
        [](int64_t num_ranks, int64_t num_experts,
           int64_t num_max_tokens_per_rank, int64_t num_topk, int64_t hidden,
           int64_t intermediate_hidden, int64_t num_shared_experts,
           const std::string& mma_type, int64_t expert_window) {
          return mxk::layout_to_vector(mxk::get_symm_layout_for_mega_moe(
              num_ranks, num_experts, num_max_tokens_per_rank, num_topk, hidden,
              intermediate_hidden, num_shared_experts, mma_type,
              expert_window));
        },
        py::arg("num_ranks"), py::arg("num_experts"),
        py::arg("num_max_tokens_per_rank"), py::arg("num_topk"),
        py::arg("hidden"), py::arg("intermediate_hidden"),
        py::arg("num_shared_experts") = 0, py::arg("mma_type") = "fp8xfp8",
        py::arg("expert_window") = 128);

  m.def("get_symm_buffer_size_for_mega_moe",
        &mxk::get_symm_buffer_size_for_mega_moe, py::arg("num_ranks"),
        py::arg("num_experts"), py::arg("num_max_tokens_per_rank"),
        py::arg("num_topk"), py::arg("hidden"), py::arg("intermediate_hidden"),
        py::arg("num_shared_experts") = 0, py::arg("mma_type") = "fp8xfp8",
        py::arg("num_tokens") = 0, py::arg("expert_window") = 128);

  m.def("get_token_alignment_for_mega_moe",
        &mxk::get_token_alignment_for_mega_moe);

  m.def("get_block_m_for_mega_moe", &mxk::get_block_m_for_mega_moe,
        py::arg("num_ranks"), py::arg("num_experts"),
        py::arg("num_max_tokens_per_rank"), py::arg("num_tokens"),
        py::arg("num_topk"), py::arg("mma_type") = "fp8xfp8");

  // ---------------- 平台显存 / IPC ----------------
  m.def("allocate_finegrained",
        [](int64_t nbytes, int64_t device) {
          return mxk::allocate_finegrained(nbytes, device);
        },
        py::arg("num_bytes"), py::arg("device") = -1);
  m.def("allocate_device",
        [](int64_t nbytes, int64_t device) {
          return mxk::allocate_device(nbytes, device);
        },
        py::arg("num_bytes"), py::arg("device") = -1);
  m.def("free_device", [](int64_t ptr) { mxk::free_device(ptr); });
  m.def("is_finegrained",
        [](int64_t ptr) { return mxk::is_finegrained(ptr); });
  m.def("get_ipc_handle", [](int64_t ptr) {
    const std::vector<uint8_t> h = mxk::get_ipc_handle(ptr);
    return py::bytes(reinterpret_cast<const char*>(h.data()), h.size());
  });
  m.def("open_ipc_handle", [](py::bytes handle) {
    char* buf = nullptr;
    Py_ssize_t size = 0;
    if (PyBytes_AsStringAndSize(handle.ptr(), &buf, &size) != 0)
      throw py::error_already_set();
    std::vector<uint8_t> h(reinterpret_cast<uint8_t*>(buf),
                           reinterpret_cast<uint8_t*>(buf) + size);
    return mxk::open_ipc_handle(h);
  });
  m.def("close_ipc_handle", [](int64_t ptr) { mxk::close_ipc_handle(ptr); });
  m.def("enable_peer_access",
        [](int64_t local_device, int64_t peer_device) {
          mxk::enable_peer_access(local_device, peer_device);
        },
        py::arg("local_device"), py::arg("peer_device"));
  m.def("wrap_device_ptr", &wrap_device_ptr);

  // ---------------- flags 区 ----------------
  m.def("local_flags_bytes", &mxk_local_flags_bytes);
  m.def("ranks_flags_bytes", &mxk_ranks_flags_bytes);
  m.def("alloc_local_flags",
        [](int64_t device) { return mxk_alloc_local_flags((int)device); });
  m.def("alloc_ranks_flags_fg",
        [](int64_t device) { return mxk_alloc_ranks_flags_fg((int)device); });

  // ---------------- SymmBuffer ----------------
  py::class_<mxk::SymmBuffer>(m, "SymmBuffer")
      .def_readwrite("launch_count", &mxk::SymmBuffer::launch_count)
      .def_readwrite("R", &mxk::SymmBuffer::R)
      .def_readwrite("rank", &mxk::SymmBuffer::rank)
      .def_readwrite("num_experts", &mxk::SymmBuffer::num_experts)
      .def_readwrite("cap", &mxk::SymmBuffer::cap)
      .def_readwrite("topk", &mxk::SymmBuffer::topk)
      .def_readwrite("hidden", &mxk::SymmBuffer::hidden)
      .def_readwrite("intermediate_hidden",
                     &mxk::SymmBuffer::intermediate_hidden)
      .def_readwrite("expert_window", &mxk::SymmBuffer::win)
      .def_readonly("layout", &mxk::SymmBuffer::L);

  m.def("create_symm_buffer", &mxk::create_symm_buffer,
        py::arg("sym_ptr"), py::arg("lf_ptr"), py::arg("rf_ptr"),
        py::arg("base_ptrs"), py::arg("rf_ptrs"), py::arg("rank"),
        py::arg("num_ranks"), py::arg("num_experts"),
        py::arg("num_max_tokens_per_rank"), py::arg("num_topk"),
        py::arg("hidden"), py::arg("intermediate_hidden"),
        py::arg("win_alloc"), py::arg("device") = -1,
        // 所有权归 Python 侧显式管理：SymmBuffer.destroy() 调
        // destroy_symm_buffer；若这里用 take_ownership，pybind 会在 _impl
        // 包装对象被回收时再 delete 一次，导致 double free。
        py::return_value_policy::reference);
  m.def("destroy_symm_buffer", &mxk::destroy_symm_buffer);

  m.def("fp8_mega_moe",
        [](torch::Tensor y_fp32, torch::Tensor l1, torch::Tensor l2,
           mxk::SymmBuffer* buf, int64_t num_tokens, double activation_clamp,
           int64_t win_cap, int64_t merge_grid, bool use_frag,
           int64_t skip_prologue) {
          TORCH_CHECK(y_fp32.is_cuda() &&
                          y_fp32.scalar_type() == torch::kFloat32,
                      "y_fp32 must be a CUDA float32 tensor");
          TORCH_CHECK(l1.is_cuda() && l2.is_cuda() &&
                          l1.scalar_type() == torch::kUInt8 &&
                          l2.scalar_type() == torch::kUInt8,
                      "l1/l2 weights must be CUDA uint8 tensors");
          TORCH_CHECK(l1.is_contiguous() && l2.is_contiguous(),
                      "l1/l2 weights must be contiguous");
          (void)activation_clamp;
          mxk::launch_fp8_mega_moe(buf, y_fp32.data_ptr<float>(),
                                   l1.data_ptr<uint8_t>(), l2.data_ptr<uint8_t>(),
                                   num_tokens, current_stream(), win_cap,
                                   merge_grid, use_frag, skip_prologue);
        },
        py::arg("y_fp32"), py::arg("l1_weights"), py::arg("l2_weights"),
        py::arg("sym_buffer"), py::arg("num_tokens"),
        py::arg("activation_clamp") = 1e30, py::arg("win_cap") = 0,
        py::arg("merge_grid") = 0, py::arg("use_frag") = false,
        py::arg("skip_prologue") = 0);

  // ---------------- mini_mega kernel 探针 ----------------
  m.def("mega_launch_empty",
        [](int grid, int threads, int64_t smem_bytes) {
          mxk::launch_empty(grid, threads, smem_bytes, current_stream());
        },
        py::arg("grid"), py::arg("threads"), py::arg("smem_bytes") = 0);
  m.def("fill_weights_lane",
        [](torch::Tensor w, int n_rows, int k_cols, int64_t seed) {
          TORCH_CHECK(w.is_cuda(), "w must be a CUDA tensor");
          TORCH_CHECK(w.scalar_type() == torch::kUInt8, "w must be uint8");
          TORCH_CHECK(w.is_contiguous(), "w must be contiguous");
          mxk::fill_weights_lane(w.data_ptr<uint8_t>(), n_rows, k_cols,
                                 (uint32_t)seed, current_stream());
        },
        py::arg("w"), py::arg("n_rows"), py::arg("k_cols"),
        py::arg("seed") = 0);
  m.def("mega_bridge_symbols", &mxk::bridge_symbols);

  // ---------------- 未实现（非 fp8xfp8 路径） ----------------
  m.def("transform_sf_into_required_layout",
        [](py::args, py::kwargs) -> torch::Tensor {
          not_implemented("transform_sf_into_required_layout");
        });
  m.def("read_route_tables", [](py::args, py::kwargs) {
    not_implemented("read_route_tables");
  });
}
