#include "utils.h"
#include <cooperative_groups.h>
namespace cg = cooperative_groups;


__forceinline__ __device__ void atomicAddVec(glm::vec3* address, const glm::vec3& val) {
    atomicAdd(&address->x, val.x);
    atomicAdd(&address->y, val.y);
    atomicAdd(&address->z, val.z);
}


__forceinline__ __device__ void atomicAddVec(glm::vec4* address, const glm::vec4& val) {
    atomicAdd(&address->x, val.x);
    atomicAdd(&address->y, val.y);
    atomicAdd(&address->z, val.z);
    atomicAdd(&address->w, val.w);
}


template <typename T>
__forceinline__ __device__ void warp_shuffle_sum(T *value) {
    for (int i = 16; i > 0; i /= 2)
        *value += __shfl_down_sync(0xFFFFFFFF, *value, i);
}

template <typename T>
__forceinline__ __device__ void block_shuffle_sum(T *shared, T *value, int tid) {
    warp_shuffle_sum<T>(value);
    if (tid % 32 == 0) shared[tid / 32] = *value;
    __syncthreads();
    if (tid < 32) {
        *value = shared[tid];
        warp_shuffle_sum<T>(value);
    }
}

template <typename T>
__forceinline__ __device__ void global_shuffle_sum(T *shared, T *value, int tid, T *global) {
    block_shuffle_sum<T>(shared, value, tid);
    if (tid == 0) atomicAdd(global, *value);
}



__global__ void LinearBlendingCuda(
    const int B, const int N, const int L,
    const float* blending_weights, // [B, L]
    const glm::vec3* base_xyzs, // [N, 3]
    const glm::vec4* base_rots, // [N, 4]
    const glm::vec3* base_rgbs, // [N, 3]
    const glm::vec3* delta_xyzs, // [L, N, 3]
    const glm::vec4* delta_rots, // [L, N, 4]
    const glm::vec3* delta_rgbs, // [L, N, 3]
    glm::vec3* blended_xyzs, // [B, N, 3]
    glm::vec4* blended_rots, // [B, N, 4]
    glm::vec3* blended_rgbs // [B, N, 3]
) {
    auto idx = cg::this_grid().thread_rank();
    if (idx >= B * N)
        return;

    int gs_index = idx % N;
    int bs_index = idx / N;

    glm::vec3 blended_xyz = base_xyzs[gs_index];
    glm::vec4 blended_rot = base_rots[gs_index];
    glm::vec3 blended_rgb = base_rgbs[gs_index];

    for (int l = 0; l < L; l++) {
        float weight = blending_weights[bs_index * L + l];
        blended_xyz += weight * delta_xyzs[l * N + gs_index];
        blended_rot += weight * delta_rots[l * N + gs_index];
        blended_rgb += weight * delta_rgbs[l * N + gs_index];
    }

    blended_xyzs[idx] = blended_xyz;
    blended_rots[idx] = blended_rot;
    blended_rgbs[idx] = blended_rgb;
}


__global__ void LinearBlendingBackwardCuda(
    const int B, const int N, const int L,
    const float* blending_weights, // [B, L]
    const glm::vec3* grad_blended_xyzs, // [B, N, 3]
    const glm::vec4* grad_blended_rots, // [B, N, 4]
    const glm::vec3* grad_blended_rgbs, // [B, N, 3]
    const glm::vec3* delta_xyzs, // [L, N, 3]
    const glm::vec4* delta_rots, // [L, N, 4]
    const glm::vec3* delta_rgbs, // [L, N, 3]
    float* grad_blending_weights, // [B, L, N]
    glm::vec3* grad_delta_xyzs, // [L, N, 3]
    glm::vec4* grad_delta_rots, // [L, N, 4]
    glm::vec3* grad_delta_rgbs // [L, N, 3]
) {
    auto idx = cg::this_grid().thread_rank();
    if (idx >= B * L * N) return;

    int n = idx % N;
    int l = (idx / N) % L;
    int b = (idx / N) / L;

    float weight = blending_weights[b * L + l];
    glm::vec3 grad_blended_xyz = grad_blended_xyzs[b * N + n];
    glm::vec4 grad_blended_rot = grad_blended_rots[b * N + n];
    glm::vec3 grad_blended_rgb = grad_blended_rgbs[b * N + n];

    glm::vec3 delta_xyz = delta_xyzs[l * N + n];
    glm::vec4 delta_rot = delta_rots[l * N + n];
    glm::vec3 delta_rgb = delta_rgbs[l * N + n];

    float grad_weight = glm::dot(grad_blended_xyz, delta_xyz) 
        + glm::dot(grad_blended_rot, delta_rot) 
        + glm::dot(grad_blended_rgb, delta_rgb);
    
    grad_blending_weights[idx] = grad_weight;


    atomicAddVec(&grad_delta_xyzs[l * N + n], weight * grad_blended_xyz);
    atomicAddVec(&grad_delta_rots[l * N + n], weight * grad_blended_rot);
    atomicAddVec(&grad_delta_rgbs[l * N + n], weight * grad_blended_rgb);
}


__global__ void LinearBlendingBackwardCudaFast(
    const int B, const int N, const int L,
    const float* blending_weights, // [B, L]
    const glm::vec3* grad_blended_xyzs, // [B, N, 3]
    const glm::vec4* grad_blended_rots, // [B, N, 4]
    const glm::vec3* grad_blended_rgbs, // [B, N, 3]
    const glm::vec3* delta_xyzs, // [L, N, 3]
    const glm::vec4* delta_rots, // [L, N, 4]
    const glm::vec3* delta_rgbs, // [L, N, 3]
    float* grad_blending_weights, // [B, L]
    glm::vec3* grad_delta_xyzs, // [L, N, 3]
    glm::vec4* grad_delta_rots, // [L, N, 4]
    glm::vec3* grad_delta_rgbs // [L, N, 3]
) {
    __shared__ float shared_data[32];

    auto block = cg::this_thread_block();
    int tid = block.thread_index().x;
    int n = block.group_index().x * 1024 + tid;
    int b = block.group_index().y;
    int l = block.group_index().z;

    glm::vec3 grad_blended_xyz = glm::vec3(0.0f);
    glm::vec4 grad_blended_rot = glm::vec4(0.0f);
    glm::vec3 grad_blended_rgb = glm::vec3(0.0f);
    glm::vec3 delta_xyz = glm::vec3(0.0f);
    glm::vec4 delta_rot = glm::vec4(0.0f);
    glm::vec3 delta_rgb = glm::vec3(0.0f);

    if (n < N) {
        grad_blended_xyz = grad_blended_xyzs[b * N + n];
        grad_blended_rot = grad_blended_rots[b * N + n];
        grad_blended_rgb = grad_blended_rgbs[b * N + n];
        delta_xyz = delta_xyzs[l * N + n];
        delta_rot = delta_rots[l * N + n];
        delta_rgb = delta_rgbs[l * N + n];
    }

    float grad_weight = glm::dot(grad_blended_xyz, delta_xyz) 
        + glm::dot(grad_blended_rot, delta_rot) 
        + glm::dot(grad_blended_rgb, delta_rgb);
    global_shuffle_sum<float>(shared_data, &grad_weight, tid, &grad_blending_weights[b * L + l]);
    if (n >= N) return;
    
    float weight = blending_weights[b * L + l];
    // grad_blending_weights[b * L * N + l * N + n] = grad_weight;
    atomicAddVec(&grad_delta_xyzs[l * N + n], weight * grad_blended_xyz);
    atomicAddVec(&grad_delta_rots[l * N + n], weight * grad_blended_rot);
    atomicAddVec(&grad_delta_rgbs[l * N + n], weight * grad_blended_rgb);
}


void CudaUtils::LinearBlending::forward(
    const int B, const int N, const int L,
    const float* blending_weights,
    const float* base_xyzs,
    const float* base_rots,
    const float* base_rgbs,
    const float* delta_xyzs,
    const float* delta_rots,
    const float* delta_rgbs,
    float* blended_xyzs,
    float* blended_rots,
    float* blended_rgbs
) {
    const int P = B * N;
    LinearBlendingCuda<< <(P + 255) / 256, 256 >> >(
        B, N, L,
        blending_weights,
        (glm::vec3*)base_xyzs,
        (glm::vec4*)base_rots,
        (glm::vec3*)base_rgbs,
        (glm::vec3*)delta_xyzs,
        (glm::vec4*)delta_rots,
        (glm::vec3*)delta_rgbs,
        (glm::vec3*)blended_xyzs,
        (glm::vec4*)blended_rots,
        (glm::vec3*)blended_rgbs
    );
}


void CudaUtils::LinearBlending::backward(
    const int B, const int N, const int L,
    const float* blending_weights,
    const float* grad_blended_xyzs,
    const float* grad_blended_rots,
    const float* grad_blended_rgbs,
    const float* delta_xyzs,
    const float* delta_rots,
    const float* delta_rgbs,
    float* grad_blending_weights,
    float* grad_delta_xyzs,
    float* grad_delta_rots,
    float* grad_delta_rgbs
) {
    // const int P = B * N * L;
    // LinearBlendingBackwardCuda<< <(P + 255) / 256, 256 >> >(
    //     B, N, L,
    //     blending_weights,
    //     (glm::vec3*)grad_blended_xyzs,
    //     (glm::vec4*)grad_blended_rots,
    //     (glm::vec3*)grad_blended_rgbs,
    //     (glm::vec3*)delta_xyzs,
    //     (glm::vec4*)delta_rots,
    //     (glm::vec3*)delta_rgbs,
    //     grad_blending_weights,
    //     (glm::vec3*)grad_delta_xyzs,
    //     (glm::vec4*)grad_delta_rots,
    //     (glm::vec3*)grad_delta_rgbs
    // );

    dim3 grid((N + 1023) / 1024, B, L);
    dim3 block(1024);
    LinearBlendingBackwardCudaFast<< <grid, block >> >(
        B, N, L,
        blending_weights,
        (glm::vec3*)grad_blended_xyzs,
        (glm::vec4*)grad_blended_rots,
        (glm::vec3*)grad_blended_rgbs,
        (glm::vec3*)delta_xyzs,
        (glm::vec4*)delta_rots,
        (glm::vec3*)delta_rgbs,
        grad_blending_weights,
        (glm::vec3*)grad_delta_xyzs,
        (glm::vec4*)grad_delta_rots,
        (glm::vec3*)grad_delta_rgbs
    );
}