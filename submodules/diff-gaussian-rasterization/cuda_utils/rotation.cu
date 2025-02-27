#include "utils.h"
#include <cooperative_groups.h>
namespace cg = cooperative_groups;


__global__ void MatrixToQuaternionCuda(
    const int P,
    const glm::mat3* matrix,
    glm::vec4* quaternion
) {
    auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

    glm::mat3 m = matrix[idx]; // do not need transpose. why ???
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

    quaternion[idx] = q; // caution: do not normalize quaternion here
}


__global__ void MatrixToQuaternionBackwardCuda(
    const int P,
    const glm::vec4* grad_quaternion,
    const glm::mat3* matrix,
    glm::mat3* grad_matrix
) {
    auto idx = cg::this_grid().thread_rank();
    if (idx >= P)
        return;

    glm::vec4 grad_q = grad_quaternion[idx];
    glm::mat3 m = matrix[idx]; // do not need transpose. why ???
    float trace = m[0][0] + m[1][1] + m[2][2];

    int choice = 0;
    float max_value = m[0][0];
    if (m[1][1] > max_value) { choice = 1; max_value = m[1][1]; }
    if (m[2][2] > max_value) { choice = 2; max_value = m[2][2]; }
    if (trace > max_value) { choice = 3; }

    glm::mat3 grad_m = glm::mat3(0.0f);
    if (choice < 3) {
        int i = choice;
        int j = (i + 1) % 3;
        int k = (j + 1) % 3;

        grad_m[k][j] = grad_q[0];
        grad_m[j][k] = -grad_q[0];
        grad_m[k][i] = grad_q[k + 1];
        grad_m[i][k] = -grad_q[k + 1];
        grad_m[j][i] = grad_q[j + 1];
        grad_m[i][j] = -grad_q[j + 1];
        grad_m[i][i] = 2.0f * grad_q[i + 1];
        grad_m[0][0] -= grad_q[i + 1];
        grad_m[1][1] -= grad_q[i + 1];
        grad_m[2][2] -= grad_q[i + 1]; 
    } else {
        grad_m[2][1] = grad_q[1];
        grad_m[1][2] = -grad_q[1];
        grad_m[0][2] = grad_q[2];
        grad_m[2][0] = -grad_q[2];
        grad_m[1][0] = grad_q[3];
        grad_m[0][1] = -grad_q[3];
        grad_m[0][0] = grad_q[0];
        grad_m[1][1] = grad_q[0];
        grad_m[2][2] = grad_q[0];
    }

    grad_matrix[idx] = glm::transpose(grad_m); // glm uses column-major
}


__global__ void QuaternionMultiplyCuda(
    const int P,
    const glm::vec4* ps,
    const glm::vec4* qs,
    glm::vec4* product
) {
    auto idx = cg::this_grid().thread_rank();
    if (idx >= P)
        return;

    glm::vec4 p = ps[idx];
    glm::vec4 q = qs[idx];
    float r0 = p[0] * q[0] - p[1] * q[1] - p[2] * q[2] - p[3] * q[3];
    float r1 = p[0] * q[1] + q[0] * p[1] + p[2] * q[3] - p[3] * q[2];
    float r2 = p[0] * q[2] + q[0] * p[2] + p[3] * q[1] - p[1] * q[3];
    float r3 = p[0] * q[3] + q[0] * p[3] + p[1] * q[2] - p[2] * q[1];
    product[idx] = glm::vec4(r0, r1, r2, r3);
}


__global__ void QuaternionMultiplyBackwardCuda(
    const int P,
    const glm::vec4* grad_product,
    const glm::vec4* ps,
    const glm::vec4* qs,
    glm::vec4* grad_ps,
    glm::vec4* grad_qs
) {
    auto idx = cg::this_grid().thread_rank();
    if (idx >= P)
        return;

    glm::vec4 p = ps[idx];
    glm::vec4 q = qs[idx];
    glm::vec4 grad_r = grad_product[idx];

    glm::vec4 grad_p = glm::vec4(0.0f);
    glm::vec4 grad_q = glm::vec4(0.0f);

    grad_p[0] = grad_r[0] * q[0] + grad_r[1] * q[1] + grad_r[2] * q[2] + grad_r[3] * q[3];
    grad_q[0] = grad_r[0] * p[0] + grad_r[1] * p[1] + grad_r[2] * p[2] + grad_r[3] * p[3];

    grad_p[1] = -grad_r[0] * q[1] + grad_r[1] * q[0] - grad_r[2] * q[3] + grad_r[3] * q[2];
    grad_q[1] = -grad_r[0] * p[1] + grad_r[1] * p[0] + grad_r[2] * p[3] - grad_r[3] * p[2];

    grad_p[2] = -grad_r[0] * q[2] + grad_r[1] * q[3] + grad_r[2] * q[0] - grad_r[3] * q[1];
    grad_q[2] = -grad_r[0] * p[2] - grad_r[1] * p[3] + grad_r[2] * p[0] + grad_r[3] * p[1];

    grad_p[3] = -grad_r[0] * q[3] - grad_r[1] * q[2] + grad_r[2] * q[1] + grad_r[3] * q[0];
    grad_q[3] = -grad_r[0] * p[3] + grad_r[1] * p[2] - grad_r[2] * p[1] + grad_r[3] * p[0];

    grad_ps[idx] = grad_p;
    grad_qs[idx] = grad_q;
}


void CudaUtils::MatrixToQuaternion::forward(
    const int P,
    const float* matrix,
    float* quaternion
) {
    MatrixToQuaternionCuda<< <(P + 255) / 256, 256 >> >(
        P, 
        (glm::mat3*)matrix, 
        (glm::vec4*)quaternion
    );
}


void CudaUtils::MatrixToQuaternion::backward(
    const int P,
    const float* grad_quaternion,
    const float* matrix,
    float* grad_matrix
) {
    MatrixToQuaternionBackwardCuda<< <(P + 255) / 256, 256 >> >(
        P, 
        (glm::vec4*)grad_quaternion, 
        (glm::mat3*)matrix, 
        (glm::mat3*)grad_matrix
    );
}


void CudaUtils::QuaternionMultiply::forward(
    const int P,
    const float* q1,
    const float* q2,
    float* product
) {
    QuaternionMultiplyCuda<< <(P + 255) / 256, 256 >> >(
        P, 
        (glm::vec4*)q1, 
        (glm::vec4*)q2, 
        (glm::vec4*)product
    );
}


void CudaUtils::QuaternionMultiply::backward(
    const int P,
    const float* grad_product,
    const float* q1,
    const float* q2,
    float* grad_q1,
    float* grad_q2
) {
    QuaternionMultiplyBackwardCuda<< <(P + 255) / 256, 256 >> >(
        P, 
        (glm::vec4*)grad_product, 
        (glm::vec4*)q1, 
        (glm::vec4*)q2, 
        (glm::vec4*)grad_q1, 
        (glm::vec4*)grad_q2
    );
}