#include <torch/extension.h>

#include <cmath>

// Soft shrink with threshold 1.
// Specification: torch.sign(x) * (x.abs() - 1).clamp(min=0)
// Every value moves 1 toward zero; values with |x| <= 1 become 0.
torch::Tensor soft_shrink(torch::Tensor input) {
    TORCH_CHECK(input.device().is_cpu(), "soft_shrink is CPU only");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32,
                "soft_shrink expects a float32 tensor, got ", input.scalar_type());

    torch::Tensor x = input.contiguous();
    torch::Tensor out = torch::empty_like(x);

    const float* src = x.data_ptr<float>();
    float* dst = out.data_ptr<float>();
    const int64_t n = x.numel();

    for (int64_t i = 0; i < n; ++i) {
        const float v = src[i];
        float a = std::fabs(v) - 1.0f;
        if (a < 0.0f) {
            a = 0.0f;
        }
        if (v > 0.0f) {
            dst[i] = a;
        } else if (v < 0.0f) {
            dst[i] = -a;
        } else {
            dst[i] = v * a;  // 0 stays 0, NaN stays NaN
        }
    }
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("soft_shrink", &soft_shrink,
               "Move every value of a float32 CPU tensor 1 toward zero");
}
