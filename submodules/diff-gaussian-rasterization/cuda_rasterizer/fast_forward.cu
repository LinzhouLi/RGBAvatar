
#include "rasterizer.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;


template<int C>
__global__ void fastForwardCuda(
    const int P,
    const float weight_threshold,
    const float* est_color,
    const float* est_weight,
    bool* initialized,
    float* feature_dc
) {
    auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

    bool init = initialized[idx];
    float weight = est_weight[idx];

    if (!init && weight > weight_threshold) {
        initialized[idx] = true;
        feature_dc[idx * C] = (est_color[idx * C] / weight - 0.5) / SH_C0;
        feature_dc[idx * C + 1] = (est_color[idx * C + 1] / weight - 0.5) / SH_C0;
        feature_dc[idx * C + 2] = (est_color[idx * C + 2] / weight - 0.5) / SH_C0;
    }
}


void CudaRasterizer::Rasterizer::fastForward(
    const int P,
    const float weight_threshold,
    const float* est_color,
    const float* est_weight,
    bool* initialized,
    float* feature_dc
) {
    fastForwardCuda<NUM_CHANNELS> << <(P + 255) / 256, 256 >> >(
        P,
        weight_threshold,
        est_color,
        est_weight,
        initialized,
        feature_dc
    );
}