#include "utils.h"
#include <cooperative_groups.h>
namespace cg = cooperative_groups;


__device__ glm::vec4 MatrixToQuaternion(const glm::mat3& m) {
    float trace = m[0][0] + m[1][1] + m[2][2];

    int choice = 0;
    float max_value = m[0][0];
    if (m[1][1] > max_value) { choice = 1; max_value = m[1][1]; }
    if (m[2][2] > max_value) { choice = 2; max_value = m[2][2]; }
    if (trace > max_value) { choice = 3; }

    glm::vec4 q = glm::vec4(0.0f);
    if (choice < 3) {
        int i = choice;
        int j = (i + 1) % 3;
        int k = (j + 1) % 3;

        q[i + 1] = 1.0f - trace + 2.0f * m[i][i];
        q[j + 1] = m[j][i] + m[i][j];
        q[k + 1] = m[k][i] + m[i][k];
        q[0] = m[k][j] - m[j][k];
    } else {
        q[1] = m[2][1] - m[1][2];
        q[2] = m[0][2] - m[2][0];
        q[3] = m[1][0] - m[0][1];
        q[0] = 1.0f + trace;
    }
    return glm::normalize(q);
}


__device__ glm::vec4 QuaternionMultiply(const glm::vec4& p, const glm::vec4& q) {
    float r0 = p[0] * q[0] - p[1] * q[1] - p[2] * q[2] - p[3] * q[3];
    float r1 = p[0] * q[1] + q[0] * p[1] + p[2] * q[3] - p[3] * q[2];
    float r2 = p[0] * q[2] + q[0] * p[2] + p[3] * q[1] - p[1] * q[3];
    float r3 = p[0] * q[3] + q[0] * p[3] + p[1] * q[2] - p[2] * q[1];
    return glm::vec4(r0, r1, r2, r3);
}


__global__ void MeshBindingCuda(
    const int B, const int N, const int F,
    const glm::vec3* face_verts, // [B, F, 3, 3]
    const glm::mat3* face_tbns, // [B, F, 3, 3]
    const glm::vec3* binding_face_barys, // [N, 3]
    const int* binding_face_ids, // [N]
    const glm::vec3* gs_xyzs, // [B, N, 3]
    const glm::vec4* gs_rots, // [B, N, 4]
    glm::vec3* transformed_gs_xyzs,
    glm::vec4* transformed_gs_rots,
    glm::mat3* binding_rotations
) {
    auto idx = cg::this_grid().thread_rank();
    if (idx >= B * N)
        return;

    int gs_index = idx % N;
    int bs_index = idx / N;

    const int binding_face_id = binding_face_ids[gs_index];
    const glm::vec3 binding_face_bary = binding_face_barys[gs_index];
    const glm::vec3 gs_xyz = gs_xyzs[idx];
    const glm::vec4 gs_rot = gs_rots[idx];

    const glm::vec3 v0 = face_verts[3 * (bs_index * F + binding_face_id)];
    const glm::vec3 v1 = face_verts[3 * (bs_index * F + binding_face_id) + 1];
    const glm::vec3 v2 = face_verts[3 * (bs_index * F + binding_face_id) + 2];

    const glm::mat3 binding_rotation = face_tbns[bs_index * F + binding_face_id];
    const glm::vec3 binding_offset = binding_face_bary.x * v0 + binding_face_bary.y * v1 + binding_face_bary.z * v2;

    glm::vec3 transformed_gs_xyz = glm::transpose(binding_rotation) * gs_xyz + binding_offset;
    glm::vec4 transformed_gs_rot = QuaternionMultiply(MatrixToQuaternion(binding_rotation), gs_rot);

    transformed_gs_xyzs[idx] = transformed_gs_xyz;
    transformed_gs_rots[idx] = transformed_gs_rot;
    binding_rotations[idx] = binding_rotation; // need transpose ?
}


__global__ void MeshBindingBackwardCuda(
    const int B, const int N,
    const glm::vec3* grad_transformed_gs_xyzs,
    const glm::vec4* grad_transformed_gs_rots,
    const glm::mat3* binding_rotations,
    glm::vec3* grad_gs_xyzs,
    glm::vec4* grad_gs_rots
) {
    auto idx = cg::this_grid().thread_rank();
    if (idx >= B * N)
        return;

    const glm::vec3 grad_transformed_gs_xyz = grad_transformed_gs_xyzs[idx];
    const glm::vec4 grad_transformed_gs_rot = grad_transformed_gs_rots[idx];
    const glm::mat3 binding_rotation = binding_rotations[idx]; // need transpose ?

    glm::vec3 grad_gs_xyz = binding_rotation * grad_transformed_gs_xyz;
    glm::vec4 grad_gs_rot = glm::vec4(0.0f);
    const glm::vec4 rot_q = MatrixToQuaternion(binding_rotation);
    grad_gs_rot[0] =  grad_transformed_gs_rot[0] * rot_q[0] + grad_transformed_gs_rot[1] * rot_q[1] 
        + grad_transformed_gs_rot[2] * rot_q[2] + grad_transformed_gs_rot[3] * rot_q[3];
    grad_gs_rot[1] = -grad_transformed_gs_rot[0] * rot_q[1] + grad_transformed_gs_rot[1] * rot_q[0] 
        + grad_transformed_gs_rot[2] * rot_q[3] - grad_transformed_gs_rot[3] * rot_q[2];
    grad_gs_rot[2] = -grad_transformed_gs_rot[0] * rot_q[2] - grad_transformed_gs_rot[1] * rot_q[3] 
        + grad_transformed_gs_rot[2] * rot_q[0] + grad_transformed_gs_rot[3] * rot_q[1];
    grad_gs_rot[3] = -grad_transformed_gs_rot[0] * rot_q[3] + grad_transformed_gs_rot[1] * rot_q[2] 
        - grad_transformed_gs_rot[2] * rot_q[1] + grad_transformed_gs_rot[3] * rot_q[0];

    grad_gs_xyzs[idx] = grad_gs_xyz;
    grad_gs_rots[idx] = grad_gs_rot;
}


void CudaUtils::MeshBinding::forward(
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
) {
    const int P = B * N;
    MeshBindingCuda<< <(P + 255) / 256, 256 >> >(
        B, N, F,
        (glm::vec3*)face_verts,
        (glm::mat3*)face_tbns,
        (glm::vec3*)binding_face_barys,
        binding_face_ids,
        (glm::vec3*)gs_xyzs,
        (glm::vec4*)gs_rots,
        (glm::vec3*)transformed_gs_xyzs,
        (glm::vec4*)transformed_gs_rots,
        (glm::mat3*)binding_rotations
    );
}


void CudaUtils::MeshBinding::backward(
    const int B, const int N,
    const float* grad_transformed_gs_xyzs,
    const float* grad_transformed_gs_rots,
    const float* binding_rotations,
    float* grad_gs_xyzs,
    float* grad_gs_rots
) {
    const int P = B * N;
    MeshBindingBackwardCuda<< <(P + 255) / 256, 256 >> >(
        B, N,
        (glm::vec3*)grad_transformed_gs_xyzs,
        (glm::vec4*)grad_transformed_gs_rots,
        (glm::mat3*)binding_rotations,
        (glm::vec3*)grad_gs_xyzs,
        (glm::vec4*)grad_gs_rots
    );
}