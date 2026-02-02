#include "custom_layer.h"

#include <cstdio>
#include <mlx/array.h>
#include <mlx/random.h>

extern "C" int fg_get_rng_digest(char *buf, size_t buf_len) {
    if (!buf || buf_len == 0)
        return 1;
    try {
        auto seq = mlx::core::random::KeySequence::default_();
        auto key = seq.next();
        key.eval();
        if (key.size() != 2 || key.dtype() != mlx::core::uint32) {
            std::snprintf(buf, buf_len, "unavailable");
            return 0;
        }
        const uint32_t *data = key.data<uint32_t>();
        double sum = static_cast<double>(data[0]) + static_cast<double>(data[1]);
        std::snprintf(buf, buf_len, "(2,):%.6f", sum);
        return 0;
    } catch (...) {
        std::snprintf(buf, buf_len, "unavailable");
        return 1;
    }
}
