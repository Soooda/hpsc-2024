#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

using namespace std;

float** init(int row, int col) {
    float** ret;
    cudaMallocManaged(&ret, row * sizeof(float*));
    for (int i = 0; i < row; i++) {
        cudaMallocManaged(&ret[i], col * sizeof(float*));
        for (int j = 0; j < col; j++) {
            ret[i][j] = 0.0;
        }
    }
    return ret;
}

void copy(float** from, float** to, int nx, int ny) {
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            to[j][i] = from[j][i];
        }
    }
}

void cuda_free(float** matrix, int row) {
    for (int i = 0; i < row; i++) {
        cudaFree(matrix[i]);
    }
    cudaFree(matrix);
}

__global__ void compute_b(float** u, float** v, float** b, int nx, int ny, double dx, double dy, double dt, double rho) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i > 0 && i < nx - 1 && j > 0 && j < ny - 1) {
        b[j][i] = rho * (1.0 / dt * ((u[j][i + 1] - u[j][i - 1]) / (2 * dx) + (v[j + 1][i] - v[j - 1][i]) / (2 * dy)) - pow((u[j][i + 1] - u[j][i - 1]) / (2 * dx), 2) - 2 * ((u[j + 1][i] - u[j - 1][i]) / (2 * dy) * (v[j][i + 1] - v[j][i - 1]) / (2 * dx)) - pow((v[j + 1][i] - v[j - 1][i]) / (2 * dy), 2));
    }
}

__global__ void compute_p(float** p, float** pn, float** b, int nx, int ny, double dx, double dy) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i > 0 && i < nx - 1 && j > 0 && j < ny - 1) {
        p[j][i] = (pow(dy, 2) * (pn[j][i + 1] + pn[j][i - 1]) + pow(dx, 2) * (pn[j + 1][i] + pn[j - 1][i]) - b[j][i] * pow(dx, 2) * pow(dy, 2)) / (2 * (pow(dx, 2) + pow(dy, 2)));
    }
}

__global__ void set_boundary_p(float** p, int nx, int ny) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < ny) {
        p[j][0] = p[j][1];
        p[j][nx-1] = p[j][nx-2];
    }
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nx) {
        p[0][i] = p[1][i];
        p[ny-1][i] = 0.0;
    }
}

__global__ void compute_uv(float** u, float** v, float** un, float** vn, float** p, int nx, int ny, double dx, double dy, double dt, double rho, double nu) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i > 0 && i < nx - 1 && j > 0 && j < ny - 1) {
        u[j][i] = un[j][i] - un[j][i] * dt / dx * (un[j][i] - un[j][i - 1]) - un[j][i] * dt / dy * (un[j][i] - un[j - 1][i]) - dt / (2 * rho * dx) * (p[j][i + 1] - p[j][i - 1]) + nu * dt / pow(dx, 2) * (un[j][i + 1] - 2 * un[j][i] + un[j][i - 1]) + nu * dt / pow(dy, 2) * (un[j + 1][i] - 2 * un[j][i] + un[j - 1][i]);
        v[j][i] = vn[j][i] - vn[j][i] * dt / dx * (vn[j][i] - vn[j][i - 1]) - vn[j][i] * dt / dy * (vn[j][i] - vn[j - 1][i]) - dt / (2 * rho * dy) * (p[j + 1][i] - p[j - 1][i]) + nu * dt / pow(dx, 2) * (vn[j][i + 1] - 2 * vn[j][i] + vn[j][i - 1]) + nu * dt / pow(dy, 2) * (vn[j + 1][i] - 2 * vn[j][i] + vn[j - 1][i]);

    }
}

__global__ void set_boundary_uv(float** u, float** v, int nx, int ny) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < ny) {
        u[j][0] = 0.0;
        u[j][nx-1] = 0.0;
        v[j][0] = 0.0;
        v[j][nx-1] = 0.0;
    }
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nx) {
        u[0][i] = 0.0;
        u[ny-1][i] = 1.0;
        v[0][i] = 0.0;
        v[ny-1][i] = 0.0;
    }
}

int main() {
    int nx = 41;
    int ny = 41;
    int nt = 500;
    int nit = 50;
    double dx = 2.0 / (nx - 1);
    double dy = 2.0 / (ny - 1);
    double dt = 0.01;
    double rho = 1.0;
    double nu = 0.02;

    float** d_u = init(ny, nx);
    float** d_v = init(ny, nx);
    float** d_p = init(ny, nx);
    float** d_b = init(ny, nx);
    float** d_un = init(ny, nx);
    float** d_vn = init(ny, nx);
    float** d_pn = init(ny, nx);

    dim3 blockSize(16, 16);
    dim3 gridSize((nx + blockSize.x - 1) / blockSize.x, (ny + blockSize.y - 1) / blockSize.y);

    ofstream ufile("u.dat");
    ofstream vfile("v.dat");
    ofstream pfile("p.dat");

    for (int n = 0; n < nt; n++) {
        compute_b<<<gridSize, blockSize>>>(d_u, d_v, d_b, nx, ny, dx, dy, dt, rho);
        cudaDeviceSynchronize();

        for (int it = 0; it < nit; it++) {
            copy(d_p, d_pn, nx, ny);
            compute_p<<<gridSize, blockSize>>>(d_p, d_pn, d_b, nx, ny, dx, dy);
            cudaDeviceSynchronize();
        }

        set_boundary_p<<<gridSize, blockSize>>>(d_p, nx, ny);
        cudaDeviceSynchronize();

        copy(d_u, d_un, nx, ny);
        copy(d_v, d_vn, nx, ny);

        compute_uv<<<gridSize, blockSize>>>(d_u, d_v, d_un, d_vn, d_p, nx, ny, dx, dy, dt, rho, nu);
        cudaDeviceSynchronize();

        set_boundary_uv<<<gridSize, blockSize>>>(d_u, d_v, nx, ny);
        cudaDeviceSynchronize();

        if (n % 10 == 0) {
            for (int j = 0; j < ny; j++) {
                for (int i = 0; i < nx; i++) {
                    ufile << d_u[j][i] << " ";
                    vfile << d_v[j][i] << " ";
                    pfile << d_p[j][i] << " ";
                }
            }
            ufile << "\n";
            vfile << "\n";
            pfile << "\n";
        }
    }

    cuda_free(d_u, ny);
    cuda_free(d_v, ny);
    cuda_free(d_p, ny);
    cuda_free(d_b, ny);
    cuda_free(d_un, ny);
    cuda_free(d_vn, ny);
    cuda_free(d_pn, ny);

    ufile.close();
    vfile.close();
    pfile.close();

    return 0;
}
