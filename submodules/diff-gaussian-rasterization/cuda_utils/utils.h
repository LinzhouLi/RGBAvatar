#ifndef CUDA_UTILS_H_INCLUDED
#define CUDA_UTILS_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace CudaUtils {

    class MatrixToQuaternion {
    public:
        static void forward(
            const int P,
            const float* matrix,
            float* quaternion
        );
        
        static void backward(
            const int P,
            const float* grad_quaternion,
            const float* matrix,
            float* grad_matrix
        );
    };

    class QuaternionMultiply {
    public:
        static void forward(
            const int P,
            const float* q1,
            const float* q2,
            float* product
        );
        
        static void backward(
            const int P,
            const float* grad_product,
            const float* q1,
            const float* q2,
            float* grad_q1,
            float* grad_q2
        );
    };

    class MeshBinding {
    public:
        static void forward(
            const int B, const int N, const int F,
            const float* face_verts,
            const float* face_tbns,
            const float* binding_face_barys,
            const int* binding_face_ids,
            const float* gs_xyzs,
            const float* gs_rots,
            float* transformed_gs_xyzs,
            float* transformed_gs_rots,
            float* binding_rotations
        );
        
        static void backward(
            const int B, const int N,
            const float* grad_transformed_gs_xyzs,
            const float* grad_transformed_gs_rots,
            const float* binding_rotations,
            float* grad_gs_xyzs,
            float* grad_gs_rots
        );
    };

    class LinearBlending{
    public:
        static void forward(
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
        );
        
        static void backward(
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
        );
}   ;

    void ComputeFaceTBN(
        const int B, const int F,
        const float* face_verts,
        const float* face_uvs,
        float* TBNs
    );

}

#endif