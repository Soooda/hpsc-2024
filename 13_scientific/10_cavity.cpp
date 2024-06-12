#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>
#include <cmath>

using namespace std;
typedef vector<vector<float>> Matrix;

Matrix init(int row, int col) {
    return Matrix(row, vector<float>(col, 0));
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

    Matrix u = init(ny, nx);
    Matrix v = init(ny, nx);
    Matrix p = init(ny, nx);
    Matrix b = init(ny, nx);

    ofstream ufile("u.dat");
    ofstream vfile("v.dat");
    ofstream pfile("p.dat");

    for (int n = 0; n < nt; n++) {
        for (int j = 1; j < ny - 1; j++) {
            for (int i = 1; i < nx - 1; i++) {
                // Compute b[j][i]
                b[j][i] = rho * (1.0 / dt * ((u[j][i + 1] - u[j][i - 1]) / (2 * dx) + (v[j + 1][i] - v[j - 1][i]) / (2 * dy)) - pow((u[j][i + 1] - u[j][i - 1]) / (2 * dx), 2) - 2 * ((u[j + 1][i] - u[j - 1][i]) / (2 * dy) * (v[j][i + 1] - v[j][i - 1]) / (2 * dx)) - pow((v[j + 1][i] - v[j - 1][i]) / (2 * dy), 2));
            }
        }

        for (int it = 0; it < nit; it++) {
            auto pn = p;
            for (int j = 1; j < ny - 1; j++) {
                for (int i = 1; i < nx - 1; i++) {
                    // Compute p[j][i]
                    p[j][i] = (pow(dy, 2) * (pn[j][i + 1] + pn[j][i - 1]) + pow(dx, 2) * (pn[j + 1][i] + pn[j - 1][i]) - b[j][i] * pow(dx, 2) * pow(dy, 2)) / (2 * (pow(dx, 2) + pow(dy, 2)));
                }
            }
        }

        for (int j = 0; j < ny; j++) {
            // Compute p[j][0] and p[j][nx-1]
            p[j][0] = p[j][1];
            p[j][nx-1] = p[j][nx-2];
        }
        for (int i = 0; i < nx; i++) {
            // Compute p[0][i] and p[ny-1][i]
            p[0][i] = p[1][i];
            p[ny-1][i] = 0.0;
        }

        auto un = u;
        auto vn = v;

        for (int j = 1; j < ny - 1; j++) {
            for (int i = 1; i < nx - 1; i++) {
                // Compute u[j][i] and v[j][i]
                u[j][i] = un[j][i] - un[j][i] * dt / dx * (un[j][i] - un[j][i - 1]) - un[j][i] * dt / dy * (un[j][i] - un[j - 1][i]) - dt / (2 * rho * dx) * (p[j][i + 1] - p[j][i - 1]) + nu * dt / pow(dx, 2) * (un[j][i + 1] - 2 * un[j][i] + un[j][i - 1]) + nu * dt / pow(dy, 2) * (un[j + 1][i] - 2 * un[j][i] + un[j - 1][i]);
                v[j][i] = vn[j][i] - vn[j][i] * dt / dx * (vn[j][i] - vn[j][i - 1]) - vn[j][i] * dt / dy * (vn[j][i] - vn[j - 1][i]) - dt / (2 * rho * dy) * (p[j + 1][i] - p[j - 1][i]) + nu * dt / pow(dx, 2) * (vn[j][i + 1] - 2 * vn[j][i] + vn[j][i - 1]) + nu * dt / pow(dy, 2) * (vn[j + 1][i] - 2 * vn[j][i] + vn[j - 1][i]);
            }
        }

        for (int j = 0; j < ny; j++) {
            // Compute u[j][0], u[j][nx-1], v[j][0], v[j][nx-1]
            u[j][0] = 0.0;
            u[j][nx-1] = 0.0;
            v[j][0] = 0.0;
            v[j][nx-1] = 0.0;
        }
        for (int i = 0; i < nx; i++) {
            // Compute u[0][i], u[ny-1][i], v[0][i], v[ny-1][i]
            u[0][i] = 0.0;
            u[ny-1][i] = 1.0;
            v[0][i] = 0.0;
            v[ny-1][i] = 0.0;
        }

        if (n % 10 == 0) {
            for (int j = 0; j < ny; j++) {
                for (int i = 0; i < nx; i++) {
                    ufile << u[j][i] << " ";
                    vfile << v[j][i] << " ";
                    pfile << p[j][i] << " ";
                }
            }
            ufile << "\n";
            vfile << "\n";
            pfile << "\n";
        }
    }

    ufile.close();
    vfile.close();
    pfile.close();
    return 0;
}
