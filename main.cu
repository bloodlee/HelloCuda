#include <iostream>
#include "common/book.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "common/cpu_bitmap.h"
#include "common/cpu_anim.h"

using namespace std;

__global__ void add(int a, int b, int* c)
{
    *c = a + b;
}

__global__ void add2(int* const a, int* const b, int* const c, int N) {
    unsigned int tid = blockIdx.x;
    if (tid < N) {
        c[tid] = a[tid] + b[tid];
    }
}

int main1() {
    int c;
    int* dev_c;

    HANDLE_ERROR(cudaMalloc((void**)&dev_c, sizeof(int)));

    add<<<1, 1>>>(2, 7, dev_c);

    HANDLE_ERROR(cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost));

    cout << "2 + 7 = " << c << endl;

    cudaFree(dev_c);

    int count;
    HANDLE_ERROR(cudaGetDeviceCount(&count));

    cout << "Has " << count << " device(s)" << endl;

    for (int i = 0; i < count; ++i) {
        cudaDeviceProp prop;

        HANDLE_ERROR(cudaGetDeviceProperties(&prop, i));
        cout << prop.name << endl;
        cout << prop.totalGlobalMem / 1024.0 / 1024.0 << endl;
        cout << prop.maxThreadsPerBlock << endl;
        cout << "compute capability: " << prop.major << "." << prop.minor << endl;
    }

    return 0;
}

int main2() {
    const int N = 20;
    int a[N], b[N], c[N];
    int* dev_a, * dev_b, * dev_c;

    HANDLE_ERROR(cudaMalloc((void**)&dev_a, N * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_b, N * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_c, N * sizeof(int)));

    for (int i = 0; i < N; ++i) {
        a[i] = -i;
        b[i] = i * i;
    }

    HANDLE_ERROR(cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice));

    add2<<<N, 1 >>>(dev_a, dev_b, dev_c, N);

    HANDLE_ERROR(cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost));

    for (int i = 0; i < N; ++i) {
        printf("%d + %d = %d\n", a[i], b[i], c[i]);
    }

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return 0;
}

#define DIM 1000

struct cuComplex {
    float r;
    float i;

    __device__ cuComplex(float a, float b) : r(a), i(b) {}

    __device__ float magnitude2() {
        return r * r + i * i;
    }

    __device__ cuComplex operator*(const cuComplex& a) {
        return cuComplex(r * a.r - i * a.i, i * a.r + r * a.i);
    }

    __device__ cuComplex operator+(const cuComplex& a) {
        return cuComplex(r + a.r, i + a.i);
    }
};

__device__ int julia(int x, int y) {
    const float scale = 1.5;

    float jx = scale * (float)(DIM / 2 - x) / (DIM / 2);
    float jy = scale * (float)(DIM / 2 - y) / (DIM / 2);

    cuComplex c(-0.8, 0.156);
    cuComplex a(jx, jy);

    int i = 0;
    for (i = 0; i < 200; ++i) {
        a = a * a + c;
        if (a.magnitude2() > 1000) {
            return 0;
        }
    }

    return 1;
}

__global__ void kernel(unsigned char* ptr) {
    int x = blockIdx.x;
    int y = blockIdx.y;

    int offset = x + y * gridDim.x;

    int juliaValue = julia(x, y);
    ptr[offset * 4 + 0] = 255 * juliaValue;
    ptr[offset * 4 + 1] = 0;
    ptr[offset * 4 + 2] = 0;
    ptr[offset * 4 + 3] = 255;
}

int main3() {
    CPUBitmap bitmap(DIM, DIM);
    unsigned char* dev_bitmap;

    HANDLE_ERROR(cudaMalloc((void**)&dev_bitmap, bitmap.image_size()));

    dim3 grid(DIM, DIM);
    kernel<<<grid, 1>>> (dev_bitmap);

    HANDLE_ERROR(cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost));

    bitmap.display_and_exit();

    HANDLE_ERROR(cudaFree(dev_bitmap));

    return 0;
}

int main() {
    return main3();
}