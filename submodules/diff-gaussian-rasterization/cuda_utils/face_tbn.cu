#include "utils.h"
#include <cooperative_groups.h>
namespace cg = cooperative_groups;


__global__ void ComputeFaceTBNCuda(
    const int B, const int F,
    const glm::vec3* face_verts, // [B, F, 3, 3]
    const glm::vec2* face_uvs,   // [F, 3, 2]
    glm::mat3* TBNs
) {
    auto idx = cg::this_grid().thread_rank();
    if (idx >= B * F)
        return;

    glm::vec3 v0 = face_verts[3 * idx];
    glm::vec3 v1 = face_verts[3 * idx + 1];
    glm::vec3 v2 = face_verts[3 * idx + 2];

    int face_index = idx % F;
    glm::vec2 uv0 = face_uvs[3 * face_index];
    glm::vec2 uv1 = face_uvs[3 * face_index + 1];
    glm::vec2 uv2 = face_uvs[3 * face_index + 2];

    glm::vec3 edge1 = v1 - v0;
    glm::vec3 edge2 = v2 - v0;

    glm::vec2 deltaUV1 = uv1 - uv0;
    glm::vec2 deltaUV2 = uv2 - uv0;

    float f = 1.0f / (deltaUV1.x * deltaUV2.y - deltaUV1.y * deltaUV2.x);
    glm::vec3 normal = glm::cross(edge1, edge2);
    glm::vec3 tangent = (edge1 * deltaUV2.y - edge2 * deltaUV1.y) * f;
    glm::vec3 bitangent = (edge2 * deltaUV1.x - edge1 * deltaUV2.x) * f;

    normal = glm::normalize(normal);
    tangent = glm::normalize(tangent);
    bitangent = glm::normalize(bitangent);
    TBNs[idx] = glm::transpose(glm::mat3(tangent, bitangent, normal)); // glm uses column-major
}


void CudaUtils::ComputeFaceTBN(
    const int B, const int F,
    const float* face_verts,
    const float* face_uvs,
    float* TBNs
) {
    const int P = B * F;
    ComputeFaceTBNCuda<< <(P + 255) / 256, 256 >> >(
        B, F,
        (glm::vec3*)face_verts,
        (glm::vec2*)face_uvs,
        (glm::mat3*)TBNs
    );
}