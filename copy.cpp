
//#define FLOAT
//#define FLOAT_ZERO_ONE
//#define FLOAT_ZERO_NOE
//#define FLOAT_ZERO_THREE
#define FLOAT_ZERO_THREE_MAT
#define FLOAT_ZERO_THREE_LOCAL_MAT
#define FLOAT_ZERO_THREE_MAT_TILED
//#define FLOAT_ZERO_THREE_LONG_MAT
//#define FLOAT_CPU
#define FLOAT_MAT_CPU
//#define DOUBLE
//#define ULTRA
//#define TEN
//#define TEN_ZERO
//#define BUF_SIZE 4096
//#define BUF_SIZE 8192
//#define BUF_SIZE 16384
//#define BUF_SIZE 32768
//#define BUF_SIZE 65536
//#define BUF_SIZE 131072
#define NROWS 2048
//#define BUF_SIZE 64*2048
#define BUF_SIZE 64*NROWS
//#define MAT_SIZE1 64*2048*2
#define MAT_SIZE1 64*NROWS*2
#define MAT_SIZE2 64*64
#define VERBOSITY_LEVEL 2
#define REPS 1000
//#define BUF_SIZE 1024

#include <iostream>
#include <string.h>
#include <sys/time.h>

#if defined(__APPLE__) || defined(APPLE)
    #include <OpenCL/OpenCL.h>
#else
    //#include <CL/opencl.h>
    #include <CL/cl.h>
    #include <CL/cl_ext.h>
    //#define FLOAT_ZERO_TWO
#endif
using namespace std;

#define STRINGIFY(src) #src

#define kernel_header "#pragma OPENCL EXTENSION cl_khr_fp64 : enable \n typedef struct tag_UltraFloat{float mantissa; int base; int exponent;} UltraFloat; typedef struct tag_TenFloat{float mantissa; int exponent;} TenFloat;"

inline const char* Kernels()
{
    static const char* kernels =
        kernel_header
        #include "kernel.cl"
        ;
    return kernels;
}

cl_context ctx;
cl_kernel float_kernel;
cl_kernel float_no_e_kernel;
cl_kernel float_mat_kernel;
cl_kernel float_mat_local_kernel;
cl_kernel float_mat_tiled_kernel;
cl_kernel e_rebalance;;
cl_kernel double_kernel;
cl_kernel ultra_kernel;
cl_kernel ten_kernel;
cl_command_queue queue;
size_t global_work_size = BUF_SIZE;
size_t local_work_size;
cl_int err_num;
cl_program prog;
void* mapPtrA;
void* mapPtrB;
void* mapPtrAb;
void* mapPtrBb;

cl_mem d_buf1;
cl_mem d_buf1b;
cl_mem d_buf2;
cl_mem d_buf2b;

// XXX do these but with ints instead of floats
struct UltraFloat
{
    cl_float mantissa;
    cl_int base;
    cl_int exponent;
};

struct TenFloat
{
    cl_float mantissa;
    // Base == 10;
    cl_int exponent;
};

int setup_context();
int compile_kernel(const char* kernel_name, cl_kernel &kernel);
int create_buffer(cl_mem &d_buf, void* buffer, size_t size);
int create_buffer_zero(cl_mem &d_buf, size_t size);
void* map_buffer_zero(cl_mem d_buf, size_t size);
int unmap_buffer_zero(cl_mem &d_buf, void* buffer);
int launch_kernel(cl_kernel kernel);
int read_buffer(cl_mem &d_buf, void* data, size_t size);

int main()
{
    setup_context();
    const char* kernel_name;
    size_t size;
    size_t size_e;

    float runtime = 0;
    timeval t1, t2;

    int* exponents = new int[BUF_SIZE];
    float* floatCPUDataset = new float[BUF_SIZE];
    float* floatZeroDataset4 = new float[MAT_SIZE1];
    float* floatZeroDataset4b = new float[MAT_SIZE2];
    int* floatZeroExponents4 = new int[MAT_SIZE1];
    int* floatZeroExponents4b = new int[MAT_SIZE2];
    cl_int size1 = sizeof(float) * MAT_SIZE1;
    cl_int size_e1 = sizeof(cl_int) * MAT_SIZE1;
    cl_int size2 = sizeof(float) * MAT_SIZE2;
    cl_int size_e2 = sizeof(cl_int) * MAT_SIZE2;

#if defined(FLOAT) || defined(DOUBLE)
    for (int i = 0; i < BUF_SIZE; i++)
    {
        exponents[i] = 0;
    }
#endif

    // ******* BURN RUN SO THE FIRST ONE DOESN'T SUCK *******
    float* burn_floatDataset = new float[BUF_SIZE];
    for (int i = 0; i < BUF_SIZE; i++)
    {
        burn_floatDataset[i] = 2;
    }
    kernel_name = "floatTest";
    compile_kernel(kernel_name, float_kernel);
    size = sizeof(float) * BUF_SIZE;

    gettimeofday(&t1, NULL);

    create_buffer(d_buf1, burn_floatDataset, size);
    create_buffer(d_buf2, exponents, sizeof(int) * BUF_SIZE);
    err_num  = clSetKernelArg(float_kernel, 0, sizeof(cl_mem), (void *) &d_buf1);
    err_num  |= clSetKernelArg(float_kernel, 1, sizeof(cl_mem), (void *) &d_buf2);
    if (err_num != CL_SUCCESS)
    {
        cout << "kernel arg set fail" << endl;
        exit(err_num);
    }
    launch_kernel(float_kernel);
    read_buffer(d_buf1, burn_floatDataset, size);
    read_buffer(d_buf2, exponents, sizeof(int) * BUF_SIZE);

    gettimeofday(&t2, NULL);
    runtime += (t2.tv_sec -t1.tv_sec) * 1000.0;
    runtime += (t2.tv_usec - t1.tv_usec) / 1000.0;

    // ******* END BURN RUN SO THE FIRST ONE DOESN'T SUCK *******

#ifdef FLOAT_CPU
    runtime = 0;

    gettimeofday(&t1, NULL);

    for (int iter = 0; iter < REPS; iter++) {
        for (int i = 0; i < BUF_SIZE; i++)
        {
            floatCPUDataset[i] = 10.0;
            exponents[i] = 0;
        }

#pragma omp parallel for
        for (int i = 0; i < BUF_SIZE; i++)
        {
            float temp_v = floatCPUDataset[i];
            for (int j = 0; j < 64; j++) {
                temp_v += temp_v * 0.001;
            }

            floatCPUDataset[i] += temp_v;
            exponents[i] += 1;
        }
    }

    gettimeofday(&t2, NULL);
    runtime += (t2.tv_sec -t1.tv_sec) * 1000.0;
    runtime += (t2.tv_usec - t1.tv_usec) / 1000.0;
    cout << "Float_CPU runtime: " << runtime/REPS << endl;

    if (VERBOSITY_LEVEL > 1)
    {
        cout << "Float_CPU results:" << endl;
        //for (int i = 0; i < BUF_SIZE; i++)
        for (int i = 0; i < 10; i++)
        {
            cout << floatCPUDataset[i];
            cout << "x10^";
            cout << exponents[i];
            cout << " ";
        }
        cout << endl;
    }
#endif

#ifdef FLOAT_MAT_CPU
    runtime = 0;

    gettimeofday(&t1, NULL);

    /*
    floatCPUDataset = new float[BUF_SIZE];
    float* floatCPUDatasetb = new float[BUF_SIZE];
    float* floatCPUExponentsb = new float[BUF_SIZE];
    */
    float* mat1 = new float[MAT_SIZE1];
    float* mat1e = new float[MAT_SIZE1];
    float* mat2 = new float[MAT_SIZE2];
    float* mat2e = new float[MAT_SIZE2];
    //int nRows = 2048;
    int nRows = NROWS;
    int nCols = 64;
    int m = 64;
    for (int i = 0; i < MAT_SIZE2; i++) {
        mat1[i] = i;
        mat1e[i] = 0;
    }
    for (int i = 0; i < MAT_SIZE2; i++) {
        mat2[i] = i;
        mat2e[i] = 0;
    }
    /*
    if (VERBOSITY_LEVEL > 1)
    {
        cout << "Float_CPU_Mat results:" << endl;
        for (int i = 0; i < MAT_SIZE1; i++)
        {
            cout << mat1[i];
            cout << " ";
        }
        cout << endl;
        for (int i = 0; i < MAT_SIZE2; i++)
        {
            cout << mat2[i];
            cout << " ";
        }
        cout << endl;
    }
    */
    for (int iter = 0; iter < REPS; iter++) {
        for (int i = 0; i < MAT_SIZE1; i++) {
            mat1[i] = i;
            mat1e[i] = 0;
        }
        for (int i = 0; i < MAT_SIZE2; i++) {
            mat2[i] = i;
            mat2e[i] = 0;
        }

#pragma omp parallel for
        /* Normal Matrix multiplication
        for (int i = 0; i < nRows; i++)
        {
            for (int j = 0; j < m; j++) {
                float sum = 0.0;
                for (int k = 0; k < nCols; k++) {
                    sum += mat1[i*nCols + k] * mat2[k*m + j];
                }
                mat1[MAT_SIZE1/2 + i*nCols + j] = sum;
            }
        }
        */
        /* Transposed Matrix Multiplication */
        for (int i = 0; i < nRows; i++)
        {
            for (int j = 0; j < m; j++) {
                float sum = 0.0;
                for (int k = 0; k < nCols; k++) {
                    sum += mat1[i*nCols + k] * mat2[j*m + k];
                }
                mat1[MAT_SIZE1/2 + i*nCols + j] = sum;
            }
        }
    }

    gettimeofday(&t2, NULL);
    runtime += (t2.tv_sec -t1.tv_sec) * 1000.0;
    runtime += (t2.tv_usec - t1.tv_usec) / 1000.0;
    cout << "Float_CPU_Mat runtime: " << runtime/REPS << endl;

    if (VERBOSITY_LEVEL > 1)
    {
        cout << "Float_CPU_Mat results:" << endl;
        //for (int i = 0; i < BUF_SIZE; i++)
        for (int i = 0; i < 20; i++)
        //for (int i = 0; i < MAT_SIZE1; i++)
        {
            cout << mat1[i + MAT_SIZE1/2 - 10];
            //cout << mat1[i];
            //cout << "x10^";
            //cout << exponents[i];
            cout << " ";
        }
        cout << endl;
        /*
        for (int i = 0; i < MAT_SIZE2; i++)
        {
            cout << mat2[i];
            //cout << "x10^";
            //cout << exponents[i];
            cout << " ";
        }
        cout << endl;
        */
    }
#endif

#ifdef FLOAT
    runtime = 0;

    gettimeofday(&t1, NULL);
    float* floatDataset = new float[BUF_SIZE];

    kernel_name = "floatTest";
    compile_kernel(kernel_name, float_kernel);
    size = sizeof(float) * BUF_SIZE;

    for (int iter = 0; iter < REPS; iter++)
    {
        for (int i = 0; i < BUF_SIZE; i++)
        {
            floatDataset[i] = 11.0;
            exponents[i] = 0;
        }

        create_buffer(d_buf1, floatDataset, size);
        create_buffer(d_buf2, exponents, sizeof(int) * BUF_SIZE);

        err_num  = clSetKernelArg(float_kernel, 0, sizeof(cl_mem), (void *) &d_buf1);
        err_num  |= clSetKernelArg(float_kernel, 1, sizeof(cl_mem), (void *) &d_buf2);
        if (err_num != CL_SUCCESS)
        {
            cout << "kernel arg set fail" << endl;
            exit(err_num);
        }

        launch_kernel(float_kernel);

        read_buffer(d_buf1, floatDataset, size);
        read_buffer(d_buf2, exponents, sizeof(int) * BUF_SIZE);
    }
    gettimeofday(&t2, NULL);
    runtime += (t2.tv_sec -t1.tv_sec) * 1000.0;
    runtime += (t2.tv_usec - t1.tv_usec) / 1000.0;
    cout << "Float runtime: " << runtime/REPS << endl;

    if (VERBOSITY_LEVEL > 1)
    {
        cout << "Float results:" << endl;
        //for (int i = 0; i < BUF_SIZE; i++)
        for (int i = 0; i < 10; i++)
        {
            cout << floatDataset[i];
            cout << "x10^";
            cout << exponents[i];
            cout << " ";
        }
        cout << endl;
    }
#endif

#ifdef FLOAT_ZERO_ONE
    runtime = 0;

    gettimeofday(&t1, NULL);

    float* floatZeroDataset;
    int* floatZeroExponents;
    size = sizeof(float) * BUF_SIZE;
    size_e = sizeof(cl_int) * BUF_SIZE;

    kernel_name = "floatTest";
    compile_kernel(kernel_name, float_kernel);

    create_buffer_zero(d_buf1, size);
    create_buffer_zero(d_buf2, sizeof(cl_int)*BUF_SIZE);
    for (int iter = 0; iter < REPS; iter++) {
        mapPtrA = (float*)map_buffer_zero(d_buf1, size);
        mapPtrB = (float*)map_buffer_zero(d_buf2, size_e);
        for (int i = 0; i < BUF_SIZE; i++)
        {
            ((float*)mapPtrA)[i] = 12.0;
            ((int*)mapPtrB)[i] = 0;
        }

        clEnqueueUnmapMemObject(queue, d_buf1, mapPtrA, 0, NULL, NULL);
        clEnqueueUnmapMemObject(queue, d_buf2, mapPtrB, 0, NULL, NULL);

        err_num  = clSetKernelArg(float_kernel, 0, sizeof(cl_mem), (void *) &d_buf1);
        err_num  |= clSetKernelArg(float_kernel, 1, sizeof(cl_mem), (void *) &d_buf2);
        if (err_num != CL_SUCCESS)
        {
            cout << "kernel arg set fail" << endl;
            exit(err_num);
        }

        launch_kernel(float_kernel);

        mapPtrA = (float*)map_buffer_zero(d_buf1, size);
        mapPtrB = (float*)map_buffer_zero(d_buf2, size_e);
    }
    if (VERBOSITY_LEVEL > 1)
    {
        cout << "Float_Zero results:" << endl;
        //for (int i = 0; i < BUF_SIZE; i++)
        for (int i = 0; i < 10; i++)
        {
            cout << ((float*)mapPtrA)[i];
            cout << "x10^";
            cout << ((int*)mapPtrB)[i];
            cout << " ";
        }
        cout << endl;
    }

    clEnqueueUnmapMemObject(queue, d_buf1, mapPtrA, 0, NULL, NULL);
    clEnqueueUnmapMemObject(queue, d_buf2, mapPtrB, 0, NULL, NULL);

    gettimeofday(&t2, NULL);
    runtime += (t2.tv_sec -t1.tv_sec) * 1000.0;
    runtime += (t2.tv_usec - t1.tv_usec) / 1000.0;
    cout << "Float_Zero runtime: " << runtime/REPS << endl;

#endif

#ifdef FLOAT_ZERO_NOE
    runtime = 0;

    gettimeofday(&t1, NULL);

    int floatZeroExponent = 1;
    floatZeroExponents = new int[0];
    size = sizeof(float) * BUF_SIZE;

    kernel_name = "floatTestNoE";
    compile_kernel(kernel_name, float_no_e_kernel);

    kernel_name = "eRebalance";
    compile_kernel(kernel_name, e_rebalance);

    create_buffer_zero(d_buf1, size);
    create_buffer_zero(d_buf2, sizeof(cl_int));

    mapPtrB = (int*)map_buffer_zero(d_buf2, sizeof(cl_int));
    ((int*)mapPtrB)[0] = 1;
    clEnqueueUnmapMemObject(queue, d_buf2, mapPtrB, 0, NULL, NULL);

    for (int iter = 0; iter < REPS; iter++) {
        mapPtrA = (float*)map_buffer_zero(d_buf1, size);
        mapPtrB = (int*)map_buffer_zero(d_buf2, sizeof(cl_int));
        for (int i = 0; i < BUF_SIZE; i++)
        {
            ((float*)mapPtrA)[i] = 12.0;
        }
        ((int*)mapPtrB)[0] = 1;
        floatZeroExponent = 1;

        clEnqueueUnmapMemObject(queue, d_buf1, mapPtrA, 0, NULL, NULL);
        clEnqueueUnmapMemObject(queue, d_buf2, mapPtrB, 0, NULL, NULL);

        err_num  = clSetKernelArg(float_no_e_kernel, 0, sizeof(cl_mem), (void *) &d_buf1);
        err_num  |= clSetKernelArg(float_no_e_kernel, 1, sizeof(cl_int), (void *) &floatZeroExponent);
        if (err_num != CL_SUCCESS)
        {
            cout << "kernel arg set fail" << endl;
            exit(err_num);
        }

        launch_kernel(float_no_e_kernel);

        err_num  = clSetKernelArg(e_rebalance, 0, sizeof(cl_mem), (void *) &d_buf1);
        err_num  |= clSetKernelArg(e_rebalance, 1, sizeof(cl_mem), (void *) &d_buf2);
        if (err_num != CL_SUCCESS)
        {
            cout << "kernel arg set fail" << endl;
            exit(err_num);
        }

        launch_kernel(e_rebalance);

        mapPtrB = (int*)map_buffer_zero(d_buf2, sizeof(cl_int));
        //floatZeroExponent = ((int*)mapPtrB)[0];
        clEnqueueUnmapMemObject(queue, d_buf2, mapPtrB, 0, NULL, NULL);

        mapPtrA = (float*)map_buffer_zero(d_buf1, size);
        clEnqueueUnmapMemObject(queue, d_buf1, mapPtrA, 0, NULL, NULL);
        //mapPtrB = (int*)map_buffer_zero(d_buf2, sizeof(cl_int));
        //clEnqueueUnmapMemObject(queue, d_buf2, mapPtrB, 0, NULL, NULL);
    }
    mapPtrA = (float*)map_buffer_zero(d_buf1, size);
    mapPtrB = (int*)map_buffer_zero(d_buf2, sizeof(cl_int));
    if (VERBOSITY_LEVEL > 1)
    {
        cout << "Float_Zero_NoE results:" << endl;
        //for (int i = 0; i < BUF_SIZE; i++)
        for (int i = 0; i < 10; i++)
        {
            cout << ((float*)mapPtrA)[i];
            cout << "x10^";
            cout << ((int*)mapPtrB)[0];
            //cout << floatZeroExponent;
            cout << " ";
        }
        cout << endl;
    }

    clEnqueueUnmapMemObject(queue, d_buf1, mapPtrA, 0, NULL, NULL);

    gettimeofday(&t2, NULL);
    runtime += (t2.tv_sec -t1.tv_sec) * 1000.0;
    runtime += (t2.tv_usec - t1.tv_usec) / 1000.0;
    cout << "Float_Zero_NoE runtime: " << runtime/REPS << endl;

#endif

#ifdef FLOAT_ZERO_TWO
    runtime = 0;

    gettimeofday(&t1, NULL);

    float* floatZeroDataset2;
    int* floatZeroExponents2;
    size = sizeof(float) * BUF_SIZE;
    size_e = sizeof(cl_int) * BUF_SIZE;

    kernel_name = "floatTest";
    compile_kernel(kernel_name, float_kernel);

    create_buffer_zero(d_buf1, size);
    create_buffer_zero(d_buf2, sizeof(cl_int)*BUF_SIZE);
    for (int iter = 0; iter < REPS; iter++) {
        mapPtrA = (float*)map_buffer_zero(d_buf1, size);
        mapPtrB = (float*)map_buffer_zero(d_buf2, size_e);
        for (int i = 0; i < BUF_SIZE; i++)
        {
            ((float*)mapPtrA)[i] = 13.0;
            ((int*)mapPtrB)[i] = 0;
        }

        clEnqueueUnmapMemObject(queue, d_buf1, mapPtrA, 0, NULL, NULL);
        clEnqueueUnmapMemObject(queue, d_buf2, mapPtrB, 0, NULL, NULL);

        err_num  = clSetKernelArg(float_kernel, 0, sizeof(cl_mem), (void *) &d_buf1);
        err_num  |= clSetKernelArg(float_kernel, 1, sizeof(cl_mem), (void *) &d_buf2);
        if (err_num != CL_SUCCESS)
        {
            cout << "kernel arg set fail" << endl;
            exit(err_num);
        }

        launch_kernel(float_kernel);

        mapPtrA = (float*)map_buffer_zero(d_buf1, size);
        mapPtrB = (float*)map_buffer_zero(d_buf2, size_e);
    }
    if (VERBOSITY_LEVEL > 1)
    {
        cout << "Float_Zero_Two results:" << endl;
        //for (int i = 0; i < BUF_SIZE; i++)
        for (int i = 0; i < 10; i++)
        {
            cout << ((float*)mapPtrA)[i];
            cout << "x10^";
            cout << ((int*)mapPtrB)[i];
            cout << " ";
        }
        cout << endl;
    }

    clEnqueueUnmapMemObject(queue, d_buf1, mapPtrA, 0, NULL, NULL);
    clEnqueueUnmapMemObject(queue, d_buf2, mapPtrB, 0, NULL, NULL);

    gettimeofday(&t2, NULL);
    runtime += (t2.tv_sec -t1.tv_sec) * 1000.0;
    runtime += (t2.tv_usec - t1.tv_usec) / 1000.0;
    cout << "Float_Zero_Two runtime: " << runtime/1000 << endl;
#endif

#ifdef FLOAT_ZERO_THREE
    runtime = 0;

    gettimeofday(&t1, NULL);

    floatDataset = new float[BUF_SIZE];

    float* floatZeroDataset3;
    int* floatZeroExponents3;
    size = sizeof(float) * BUF_SIZE;
    size_e = sizeof(cl_int) * BUF_SIZE;

    kernel_name = "floatTest";
    compile_kernel(kernel_name, float_kernel);

    create_buffer_zero(d_buf1, size);
    create_buffer_zero(d_buf2, sizeof(cl_int)*BUF_SIZE);

    for (int iter = 0; iter < REPS; iter++) {
        for (int i = 0; i < BUF_SIZE; i++)
        {
            floatDataset[i] = 11.0;
            exponents[i] = 0;
        }
        mapPtrA = (float*)map_buffer_zero(d_buf1, size);
        mapPtrB = (float*)map_buffer_zero(d_buf2, size_e);
        memcpy(mapPtrA, floatDataset, size);
        memcpy(mapPtrB, exponents, sizeof(cl_int)*BUF_SIZE);
        clEnqueueUnmapMemObject(queue, d_buf1, mapPtrA, 0, NULL, NULL);
        clEnqueueUnmapMemObject(queue, d_buf2, mapPtrB, 0, NULL, NULL);

        err_num  = clSetKernelArg(float_kernel, 0, sizeof(cl_mem), (void *) &d_buf1);
        err_num  |= clSetKernelArg(float_kernel, 1, sizeof(cl_mem), (void *) &d_buf2);
        if (err_num != CL_SUCCESS)
        {
            cout << "kernel arg set fail" << endl;
            exit(err_num);
        }

        launch_kernel(float_kernel);

        mapPtrA = (float*)map_buffer_zero(d_buf1, size);
        mapPtrB = (float*)map_buffer_zero(d_buf2, size_e);
    }
    if (VERBOSITY_LEVEL > 1)
    {
        cout << "Float_Zero_Three results:" << endl;
        //for (int i = 0; i < BUF_SIZE; i++)
        for (int i = 0; i < 10; i++)
        {
            cout << ((float*)mapPtrA)[i];
            cout << "x10^";
            cout << ((int*)mapPtrB)[i];
            cout << " ";
        }
        cout << endl;
    }

    clEnqueueUnmapMemObject(queue, d_buf1, mapPtrA, 0, NULL, NULL);
    clEnqueueUnmapMemObject(queue, d_buf2, mapPtrB, 0, NULL, NULL);

    gettimeofday(&t2, NULL);
    runtime += (t2.tv_sec -t1.tv_sec) * 1000.0;
    runtime += (t2.tv_usec - t1.tv_usec) / 1000.0;
    cout << "Float_Zero_Three runtime: " << runtime/REPS << endl;
#endif

#ifdef FLOAT_ZERO_THREE_LOCAL_MAT
    runtime = 0;

    gettimeofday(&t1, NULL);


    kernel_name = "floatMatLocalTest";
    compile_kernel(kernel_name, float_mat_local_kernel);

    create_buffer_zero(d_buf1, size1);
    create_buffer_zero(d_buf2, size_e1);
    create_buffer_zero(d_buf1b, size2);
    create_buffer_zero(d_buf2b, size_e2);

    for (int iter = 0; iter < REPS; iter++) {
        for (int i = 0; i < MAT_SIZE1; i++)
        {
            floatZeroDataset4[i] = i;
            floatZeroExponents4[i] = 0;
        }
        for (int i = 0; i < MAT_SIZE2; i++)
        {
            floatZeroDataset4b[i] = i;
            floatZeroExponents4b[i] = 0;
        }
        mapPtrA = (float*)map_buffer_zero(d_buf1, size1);
        mapPtrB = (float*)map_buffer_zero(d_buf2, size_e1);
        mapPtrAb = (float*)map_buffer_zero(d_buf1b, size2);
        mapPtrBb = (float*)map_buffer_zero(d_buf2b, size_e2);
        memcpy(mapPtrA, floatZeroDataset4, size1);
        memcpy(mapPtrB, floatZeroExponents4, size_e1);
        memcpy(mapPtrAb, floatZeroDataset4b, size2);
        memcpy(mapPtrBb, floatZeroExponents4b, size_e2);
        clEnqueueUnmapMemObject(queue, d_buf1, mapPtrA, 0, NULL, NULL);
        clEnqueueUnmapMemObject(queue, d_buf2, mapPtrB, 0, NULL, NULL);
        clEnqueueUnmapMemObject(queue, d_buf1b, mapPtrAb, 0, NULL, NULL);
        clEnqueueUnmapMemObject(queue, d_buf2b, mapPtrBb, 0, NULL, NULL);

        err_num  = clSetKernelArg(float_mat_local_kernel, 0, sizeof(cl_mem), (void *) &d_buf1);
        err_num  |= clSetKernelArg(float_mat_local_kernel, 1, sizeof(cl_mem), (void *) &d_buf2);
        err_num  = clSetKernelArg(float_mat_local_kernel, 2, sizeof(cl_mem), (void *) &d_buf1b);
        err_num  |= clSetKernelArg(float_mat_local_kernel, 3, sizeof(cl_mem), (void *) &d_buf2b);
        if (err_num != CL_SUCCESS)
        {
            cout << "kernel arg set fail" << endl;
            exit(err_num);
        }

        launch_kernel(float_mat_local_kernel);

        mapPtrA = (float*)map_buffer_zero(d_buf1, size1);
        mapPtrB = (float*)map_buffer_zero(d_buf2, size_e1);
        mapPtrAb = (float*)map_buffer_zero(d_buf1b, size2);
        mapPtrBb = (float*)map_buffer_zero(d_buf2b, size_e2);
    }
    if (VERBOSITY_LEVEL > 1)
    {
        cout << "Float_Zero_Three_Local_Mat results:" << endl;
        //for (int i = 0; i < BUF_SIZE; i++)
        for (int i = 0; i < 20; i++)
        //for (int i = 0; i < MAT_SIZE1; i++)
        {
            cout << ((float*)mapPtrA)[i + MAT_SIZE1/2 - 10];
            //cout << ((float*)mapPtrA)[i];
            //cout << "x10^";
            //cout << ((int*)mapPtrB)[i + MAT_SIZE1/2];
            cout << " ";
        }
        cout << endl;
    }

    clEnqueueUnmapMemObject(queue, d_buf1, mapPtrA, 0, NULL, NULL);
    clEnqueueUnmapMemObject(queue, d_buf2, mapPtrB, 0, NULL, NULL);
    clEnqueueUnmapMemObject(queue, d_buf1b, mapPtrAb, 0, NULL, NULL);
    clEnqueueUnmapMemObject(queue, d_buf2b, mapPtrBb, 0, NULL, NULL);

    gettimeofday(&t2, NULL);
    runtime += (t2.tv_sec -t1.tv_sec) * 1000.0;
    runtime += (t2.tv_usec - t1.tv_usec) / 1000.0;
    cout << "Float_Zero_Three_Local_Mat runtime: " << runtime/1000 << endl;
#endif

#ifdef FLOAT_ZERO_THREE_MAT
    runtime = 0;

    gettimeofday(&t1, NULL);


    kernel_name = "floatMatTest";
    compile_kernel(kernel_name, float_mat_kernel);

    create_buffer_zero(d_buf1, size1);
    create_buffer_zero(d_buf2, size_e1);
    create_buffer_zero(d_buf1b, size2);
    create_buffer_zero(d_buf2b, size_e2);

    for (int iter = 0; iter < REPS; iter++) {
        for (int i = 0; i < MAT_SIZE1; i++)
        {
            floatZeroDataset4[i] = i;
            floatZeroExponents4[i] = 0;
        }
        for (int i = 0; i < MAT_SIZE2; i++)
        {
            floatZeroDataset4b[i] = i;
            floatZeroExponents4b[i] = 0;
        }
        mapPtrA = (float*)map_buffer_zero(d_buf1, size1);
        mapPtrB = (float*)map_buffer_zero(d_buf2, size_e1);
        mapPtrAb = (float*)map_buffer_zero(d_buf1b, size2);
        mapPtrBb = (float*)map_buffer_zero(d_buf2b, size_e2);
        memcpy(mapPtrA, floatZeroDataset4, size1);
        memcpy(mapPtrB, floatZeroExponents4, size_e1);
        memcpy(mapPtrAb, floatZeroDataset4b, size2);
        memcpy(mapPtrBb, floatZeroExponents4b, size_e2);
        clEnqueueUnmapMemObject(queue, d_buf1, mapPtrA, 0, NULL, NULL);
        clEnqueueUnmapMemObject(queue, d_buf2, mapPtrB, 0, NULL, NULL);
        clEnqueueUnmapMemObject(queue, d_buf1b, mapPtrAb, 0, NULL, NULL);
        clEnqueueUnmapMemObject(queue, d_buf2b, mapPtrBb, 0, NULL, NULL);

        int kernelNCols = 64;

        err_num  = clSetKernelArg(float_mat_kernel, 0, sizeof(cl_mem), (void *) &d_buf1);
        err_num  |= clSetKernelArg(float_mat_kernel, 1, sizeof(cl_mem), (void *) &d_buf2);
        err_num  |= clSetKernelArg(float_mat_kernel, 2, sizeof(cl_mem), (void *) &d_buf1b);
        err_num  |= clSetKernelArg(float_mat_kernel, 3, sizeof(cl_mem), (void *) &d_buf2b);
        err_num  |= clSetKernelArg(float_mat_kernel, 4, sizeof(cl_int), (void *) &kernelNCols);
        if (err_num != CL_SUCCESS)
        {
            cout << "kernel arg set fail" << endl;
            exit(err_num);
        }

        launch_kernel(float_mat_kernel);

        mapPtrA = (float*)map_buffer_zero(d_buf1, size1);
        mapPtrB = (float*)map_buffer_zero(d_buf2, size_e1);
        mapPtrAb = (float*)map_buffer_zero(d_buf1b, size2);
        mapPtrBb = (float*)map_buffer_zero(d_buf2b, size_e2);
    }
    if (VERBOSITY_LEVEL > 1)
    {
        cout << "Float_Zero_Three_Mat results:" << endl;
        //for (int i = 0; i < BUF_SIZE; i++)
        for (int i = 0; i < 20; i++)
        //for (int i = 0; i < MAT_SIZE1; i++)
        {
            cout << ((float*)mapPtrA)[i + MAT_SIZE1/2 - 10];
            //cout << ((float*)mapPtrA)[i];
            //cout << "x10^";
            //cout << ((int*)mapPtrB)[i + MAT_SIZE1/2];
            cout << " ";
        }
        cout << endl;
    }

    clEnqueueUnmapMemObject(queue, d_buf1, mapPtrA, 0, NULL, NULL);
    clEnqueueUnmapMemObject(queue, d_buf2, mapPtrB, 0, NULL, NULL);
    clEnqueueUnmapMemObject(queue, d_buf1b, mapPtrAb, 0, NULL, NULL);
    clEnqueueUnmapMemObject(queue, d_buf2b, mapPtrBb, 0, NULL, NULL);

    gettimeofday(&t2, NULL);
    runtime += (t2.tv_sec -t1.tv_sec) * 1000.0;
    runtime += (t2.tv_usec - t1.tv_usec) / 1000.0;
    cout << "Float_Zero_Three_Mat runtime: " << runtime/1000 << endl;
#endif

#ifdef FLOAT_ZERO_THREE_MAT_TILED
    runtime = 0;

    gettimeofday(&t1, NULL);

    floatZeroDataset4 = new float[MAT_SIZE1];
    floatZeroDataset4b = new float[MAT_SIZE2];
    floatZeroExponents4 = new int[MAT_SIZE1];
    floatZeroExponents4b = new int[MAT_SIZE2];

    size1 = sizeof(float) * MAT_SIZE1;
    size_e1 = sizeof(cl_int) * MAT_SIZE1;
    size2 = sizeof(float) * MAT_SIZE2;
    size_e2 = sizeof(cl_int) * MAT_SIZE2;

    kernel_name = "floatMatTiledTest";
    compile_kernel(kernel_name, float_mat_tiled_kernel);

    create_buffer_zero(d_buf1, size1);
    create_buffer_zero(d_buf2, size_e1);
    create_buffer_zero(d_buf1b, size2);
    create_buffer_zero(d_buf2b, size_e2);

    for (int iter = 0; iter < REPS; iter++) {
        for (int i = 0; i < MAT_SIZE1; i++)
        {
            floatZeroDataset4[i] = i;
            floatZeroExponents4[i] = 0;
        }
        for (int i = 0; i < MAT_SIZE2; i++)
        {
            floatZeroDataset4b[i] = i;
            floatZeroExponents4b[i] = 0;
        }
        mapPtrA = (float*)map_buffer_zero(d_buf1, size1);
        mapPtrB = (float*)map_buffer_zero(d_buf2, size_e1);
        mapPtrAb = (float*)map_buffer_zero(d_buf1b, size2);
        mapPtrBb = (float*)map_buffer_zero(d_buf2b, size_e2);
        memcpy(mapPtrA, floatZeroDataset4, size1);
        memcpy(mapPtrB, floatZeroExponents4, size_e1);
        memcpy(mapPtrAb, floatZeroDataset4b, size2);
        memcpy(mapPtrBb, floatZeroExponents4b, size_e2);
        clEnqueueUnmapMemObject(queue, d_buf1, mapPtrA, 0, NULL, NULL);
        clEnqueueUnmapMemObject(queue, d_buf2, mapPtrB, 0, NULL, NULL);
        clEnqueueUnmapMemObject(queue, d_buf1b, mapPtrAb, 0, NULL, NULL);
        clEnqueueUnmapMemObject(queue, d_buf2b, mapPtrBb, 0, NULL, NULL);

        int temp_block_number = 4;
        int temp_block_size = 16;

        err_num  = clSetKernelArg(float_mat_tiled_kernel, 0, sizeof(cl_mem), (void *) &d_buf1);
        err_num  |= clSetKernelArg(float_mat_tiled_kernel, 1, sizeof(cl_mem), (void *) &d_buf2);
        err_num  |= clSetKernelArg(float_mat_tiled_kernel, 2, sizeof(cl_mem), (void *) &d_buf1b);
        err_num  |= clSetKernelArg(float_mat_tiled_kernel, 3, sizeof(cl_mem), (void *) &d_buf2b);
        err_num  |= clSetKernelArg(float_mat_tiled_kernel, 4, sizeof(cl_int), (void *) &temp_block_number);
        err_num  |= clSetKernelArg(float_mat_tiled_kernel, 5, sizeof(cl_int), (void *) &temp_block_size);
        if (err_num != CL_SUCCESS)
        {
            cout << "kernel arg set fail" << endl;
            exit(err_num);
        }

        launch_kernel(float_mat_tiled_kernel);

        mapPtrA = (float*)map_buffer_zero(d_buf1, size1);
        mapPtrB = (float*)map_buffer_zero(d_buf2, size_e1);
        mapPtrAb = (float*)map_buffer_zero(d_buf1b, size2);
        mapPtrBb = (float*)map_buffer_zero(d_buf2b, size_e2);
    }
    if (VERBOSITY_LEVEL > 1)
    {
        cout << "Float_Zero_Three_Mat_Tiled results:" << endl;
        //for (int i = 0; i < BUF_SIZE; i++)
        for (int i = 0; i < 20; i++)
        //for (int i = 0; i < MAT_SIZE1; i++)
        {
            cout << ((float*)mapPtrA)[i + MAT_SIZE1/2 - 10];
            //cout << ((float*)mapPtrA)[i];
            //cout << "x10^";
            //cout << ((int*)mapPtrB)[i + MAT_SIZE1/2];
            cout << " ";
        }
        cout << endl;
    }

    clEnqueueUnmapMemObject(queue, d_buf1, mapPtrA, 0, NULL, NULL);
    clEnqueueUnmapMemObject(queue, d_buf2, mapPtrB, 0, NULL, NULL);
    clEnqueueUnmapMemObject(queue, d_buf1b, mapPtrAb, 0, NULL, NULL);
    clEnqueueUnmapMemObject(queue, d_buf2b, mapPtrBb, 0, NULL, NULL);

    gettimeofday(&t2, NULL);
    runtime += (t2.tv_sec -t1.tv_sec) * 1000.0;
    runtime += (t2.tv_usec - t1.tv_usec) / 1000.0;
    cout << "Float_Zero_Three_Mat_Tiled runtime: " << runtime/1000 << endl;
#endif


#ifdef FLOAT_ZERO_THREE_LONG_MAT
    runtime = 0;

    gettimeofday(&t1, NULL);

    floatDataset = new float[BUF_SIZE*2];

    size = sizeof(float) * BUF_SIZE*2;
    size_e = sizeof(cl_int) * BUF_SIZE*2;

    kernel_name = "floatTest";
    compile_kernel(kernel_name, float_kernel);

    create_buffer_zero(d_buf1, size);
    create_buffer_zero(d_buf2, size_e);

    for (int iter = 0; iter < REPS; iter++) {
        for (int i = 0; i < BUF_SIZE*2; i++)
        {
            floatDataset[i] = 11.0;
            exponents[i] = 0;
        }
        mapPtrA = (float*)map_buffer_zero(d_buf1, size);
        mapPtrB = (float*)map_buffer_zero(d_buf2, size_e);
        memcpy(mapPtrA, floatDataset, size);
        memcpy(mapPtrB, exponents, size_e);
        clEnqueueUnmapMemObject(queue, d_buf1, mapPtrA, 0, NULL, NULL);
        clEnqueueUnmapMemObject(queue, d_buf2, mapPtrB, 0, NULL, NULL);

        err_num  = clSetKernelArg(float_kernel, 0, sizeof(cl_mem), (void *) &d_buf1);
        err_num  |= clSetKernelArg(float_kernel, 1, sizeof(cl_mem), (void *) &d_buf2);
        if (err_num != CL_SUCCESS)
        {
            cout << "kernel arg set fail" << endl;
            exit(err_num);
        }

        launch_kernel(float_kernel);

        mapPtrA = (float*)map_buffer_zero(d_buf1, size);
        mapPtrB = (float*)map_buffer_zero(d_buf2, size_e);
    }
    if (VERBOSITY_LEVEL > 1)
    {
        cout << "Float_Zero_Three_Long results:" << endl;
        //for (int i = 0; i < BUF_SIZE; i++)
        for (int i = 0; i < 10; i++)
        {
            cout << ((float*)mapPtrA)[i];
            cout << "x10^";
            cout << ((int*)mapPtrB)[i];
            cout << " ";
        }
        cout << endl;
    }

    clEnqueueUnmapMemObject(queue, d_buf1, mapPtrA, 0, NULL, NULL);
    clEnqueueUnmapMemObject(queue, d_buf2, mapPtrB, 0, NULL, NULL);

    gettimeofday(&t2, NULL);
    runtime += (t2.tv_sec -t1.tv_sec) * 1000.0;
    runtime += (t2.tv_usec - t1.tv_usec) / 1000.0;
    cout << "Float_Zero_Three_Long runtime: " << runtime/1000 << endl;
#endif


#ifdef DOUBLE
    runtime = 0;

    gettimeofday(&t1, NULL);
    kernel_name = "doubleTest";
    compile_kernel(kernel_name, double_kernel);

    double* doubleDataset = new double[BUF_SIZE];

    for (int iter = 0; iter < 1000; iter++) {
        for (int i = 0; i < BUF_SIZE; i++)
        {
            doubleDataset[i] = 14.0;
            exponents[i] = 0;
        }
        size = sizeof(double) * BUF_SIZE;

        create_buffer(d_buf1, doubleDataset, size);
        create_buffer(d_buf2, exponents, sizeof(int) * BUF_SIZE);
        err_num  = clSetKernelArg(double_kernel, 0, sizeof(cl_mem), (void *) &d_buf1);
        err_num  |= clSetKernelArg(double_kernel, 1, sizeof(cl_mem), (void *) &d_buf2);
        if (err_num != CL_SUCCESS)
        {
            cout << "kernel arg set fail" << endl;
            exit(err_num);
        }

        launch_kernel(double_kernel);

        read_buffer(d_buf1, doubleDataset, size);
        read_buffer(d_buf2, exponents, sizeof(int) * BUF_SIZE);
    }

    gettimeofday(&t2, NULL);
    runtime += (t2.tv_sec -t1.tv_sec) * 1000.0;
    runtime += (t2.tv_usec - t1.tv_usec) / 1000.0;
    cout << "Double runtime: " << runtime/1000 << endl;

    if (VERBOSITY_LEVEL > 1)
    {
        cout << "Double results:" << endl;
        //for (int i = 0; i < BUF_SIZE; i++)
        for (int i = 0; i < 10; i++)
        {
            cout << doubleDataset[i];
            cout << "x10^";
            cout << exponents[i];
            cout << " ";
        }
        cout << endl;
    }
#endif

#ifdef ULTRA
    runtime = 0;

    gettimeofday(&t1, NULL);

    UltraFloat* ultraDataset = new UltraFloat[BUF_SIZE];
    kernel_name = "ultraTest";
    compile_kernel(kernel_name, ultra_kernel);

    for (int iter = 0; iter < 1000; iter++) {
        for (int i = 0; i < BUF_SIZE; i++)
        {
            ultraDataset[i].mantissa = 15.0;
            ultraDataset[i].base = 10;
            ultraDataset[i].exponent = 0;
        }
        size = sizeof(UltraFloat) * BUF_SIZE;

        create_buffer(d_buf1, ultraDataset, size);
        err_num  = clSetKernelArg(ultra_kernel, 0, sizeof(cl_mem), (void *) &d_buf1);
        if (err_num != CL_SUCCESS)
        {
            cout << "kernel arg set fail" << endl;
            exit(err_num);
        }
        launch_kernel(ultra_kernel);
        read_buffer(d_buf1, ultraDataset, size);
    }

    gettimeofday(&t2, NULL);
    runtime += (t2.tv_sec -t1.tv_sec) * 1000.0;
    runtime += (t2.tv_usec - t1.tv_usec) / 1000.0;
    cout << "Ultra runtime: " << runtime/REPS << endl;

    if (VERBOSITY_LEVEL > 1)
    {
        cout << "Ultra dataset" << endl;
        //for (int i = 0; i < BUF_SIZE; i++)
        for (int i = 0; i < 10; i++)
        {
            cout << ultraDataset[i].mantissa;
            cout << "x";
            cout << ultraDataset[i].base;
            cout << "^";
            cout << ultraDataset[i].exponent;
            cout << " ";
        }
        cout << endl;
    }
#endif

#ifdef TEN
    runtime = 0;

    gettimeofday(&t1, NULL);

    TenFloat* tenDataset = new TenFloat[BUF_SIZE];
    size = sizeof(TenFloat) * BUF_SIZE;
    kernel_name = "tenTest";
    compile_kernel(kernel_name, ten_kernel);

    for (int iter = 0; iter < REPS; iter++) {
        for (int i = 0; i < BUF_SIZE; i++)
        {
            tenDataset[i].mantissa = 16.0;
            tenDataset[i].exponent = 0;
        }

        create_buffer(d_buf1, tenDataset, size);
        err_num  = clSetKernelArg(ten_kernel, 0, sizeof(cl_mem), (void *) &d_buf1);
        if (err_num != CL_SUCCESS)
        {
            cout << "kernel arg set fail" << endl;
            exit(err_num);
        }

        launch_kernel(ten_kernel);
        read_buffer(d_buf1, tenDataset, size);
    }

    gettimeofday(&t2, NULL);
    runtime += (t2.tv_sec -t1.tv_sec) * 1000.0;
    runtime += (t2.tv_usec - t1.tv_usec) / 1000.0;
    cout << "Ten runtime: " << runtime/REPS << endl;

    if (VERBOSITY_LEVEL > 1)
    {
        cout << "Ten dataset" << endl;
        //for (int i = 0; i < BUF_SIZE; i++)
        for (int i = 0; i < 10; i++)
        {
            cout << tenDataset[i].mantissa;
            cout << "x10^";
            cout << tenDataset[i].exponent;
            cout << " ";
        }
        cout << endl;
    }
#endif

#ifdef TEN_ZERO
    runtime = 0;

    gettimeofday(&t1, NULL);

    TenFloat* tenZeroDataset = new TenFloat[BUF_SIZE];
    for (int i = 0; i < BUF_SIZE; i++)
    {
        tenZeroDataset[i].mantissa = 16.0;
        tenZeroDataset[i].exponent = 0.0;
    }

    kernel_name = "tenTest";
    compile_kernel(kernel_name, ten_kernel);

    size = sizeof(TenFloat) * BUF_SIZE;
    create_buffer_zero(d_buf1, size);

    for (int iter = 0; iter < REPS; iter++) {
        mapPtrA = (TenFloat*)map_buffer_zero(d_buf1, size);
        for (int i = 0; i < BUF_SIZE; i++)
        {
            ((TenFloat*)mapPtrA)[i].mantissa = 16.0;
            ((TenFloat*)mapPtrA)[i].exponent = 1;
        }
        clEnqueueUnmapMemObject(queue, d_buf1, mapPtrA, 0, NULL, NULL);

        err_num  = clSetKernelArg(ten_kernel, 0, sizeof(cl_mem), (void *) &d_buf1);
        if (err_num != CL_SUCCESS)
        {
            cout << "kernel arg set fail" << endl;
            exit(err_num);
        }

        launch_kernel(ten_kernel);

        mapPtrA = (TenFloat*)map_buffer_zero(d_buf1, size);
    }

    if (VERBOSITY_LEVEL > 1)
    {
        cout << "Ten Zero dataset" << endl;
        //for (int i = 0; i < BUF_SIZE; i++)
        for (int i = 0; i < 10; i++)
        {
            cout << ((TenFloat*)mapPtrA)[i].mantissa;
            cout << "x10^";
            cout << ((TenFloat*)mapPtrA)[i].exponent;
            cout << " ";
        }
        cout << endl;
    }
    clEnqueueUnmapMemObject(queue, d_buf1, mapPtrA, 0, NULL, NULL);
    gettimeofday(&t2, NULL);
    runtime += (t2.tv_sec -t1.tv_sec) * 1000.0;
    runtime += (t2.tv_usec - t1.tv_usec) / 1000.0;
    cout << "Ten Zero runtime: " << runtime/REPS << endl;
#endif
    return 0;
}

void* map_buffer_zero(cl_mem d_buf, size_t size)
{
    void* temp = clEnqueueMapBuffer(  queue,
                                d_buf,
                                CL_TRUE,
                                CL_MAP_READ | CL_MAP_WRITE,
                                0,
                                size,
                                0,
                                NULL,
                                NULL,
                                NULL);
    if (err_num != CL_SUCCESS)
    {
        cout << "map buffer fail" << endl;
        exit(err_num);
    }
    return temp;
}

int unmap_buffer_zero(cl_mem &d_buf, void* data)
{
     err_num = clEnqueueUnmapMemObject( queue,
                                d_buf,
                                data,
                                0,
                                NULL,
                                NULL);
    if (err_num != CL_SUCCESS)
    {
        cout << "unmap buffer fail" << endl;
        exit(err_num);
    }
    return 0;
}

int create_buffer_zero(cl_mem &d_buf, size_t size)
{
    d_buf = clCreateBuffer( ctx,
                            CL_MEM_READ_ONLY  | CL_MEM_ALLOC_HOST_PTR,
                            size,
                            NULL,
                            &err_num);
    if (err_num != CL_SUCCESS)
    {
        cout << "make buffer fail" << endl;
        exit(err_num);
    }
    return 0;
/*
    host_buf = (float* )clEnqueueMapBuffer(   queue,
                                    d_buf,
                                    CL_TRUE,
                                    CL_MAP_READ | CL_MAP_WRITE,
                                    0,
                                    size,
                                    //data,
                                    0,
                                    NULL,
                                    NULL,
                                    NULL
            );
    if (err_num != CL_SUCCESS)
    {
        cout << "make buffer fail" << endl;
        exit(err_num);
    }
*/
    //return 0;
}
int create_buffer(cl_mem &d_buf, void* data, size_t size)
{
    d_buf = clCreateBuffer( ctx,
                    CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                    size,
                    data,
                    &err_num);
    if (err_num != CL_SUCCESS)
    {
        cout << "make buffer fail" << endl;
        exit(err_num);
    }
    /*
    err_num = clEnqueueWriteBuffer( queue,
                                    d_buf,
                                    CL_TRUE,
                                    0,
                                    size,
                                    data,
                                    0,
                                    NULL,
                                    NULL
            );
    if (err_num != CL_SUCCESS)
    {
        cout << "make buffer fail" << endl;
        exit(err_num);
    }
            */
    return 0;
}

int read_buffer(cl_mem &d_buf, void* data, size_t size)
{
    err_num = clEnqueueReadBuffer(  queue,
                                    d_buf,
                                    CL_TRUE,
                                    0,
                                    size,
                                    data,
                                    0,
                                    NULL,
                                    NULL
            );
    if (err_num != CL_SUCCESS)
    {
        cout << "read fail" << endl;
        exit(err_num);
    }
    return 0;
}

int launch_kernel(cl_kernel kernel)
{
    err_num = clEnqueueNDRangeKernel(   queue,
                                        kernel,
                                        1,
                                        0,
                                        &global_work_size,
                                        &local_work_size,
                                        0,
                                        NULL,
                                        NULL);
    if (err_num != CL_SUCCESS)
    {
        cout << "kernel launch fail" << endl;
        exit(err_num);
    }
    clFinish(queue);
    return 0;
}


int setup_context()
{
    cl_platform_id plat = NULL;
    cl_device_id *devices = NULL;
    cl_device_id device = NULL;
    cl_uint dev_count = 0;
    err_num = CL_SUCCESS;

    err_num = clGetPlatformIDs(1, &plat, NULL);
    if (err_num != CL_SUCCESS)
    {
        cout << "Plat fail" << endl;
        exit(err_num);
    }

    // Dev setup
    err_num = clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 0, NULL, &dev_count);
    devices = (cl_device_id *)malloc(dev_count * sizeof(cl_device_id));
    err_num = clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, dev_count, devices, NULL);

    device = devices[0]; // XXX set back down to 0
    if (err_num != CL_SUCCESS)
    {
        cout << "Dev fail" << endl;
        exit(err_num);
    }

    // Context setup
    // 1 == my device count (arbitrary)
    ctx = clCreateContext(0, 1, &device, NULL, NULL, &err_num);
    if (err_num != CL_SUCCESS)
    {
        cout << "Ctx fail" << endl;
        exit(err_num);
    }

    // get device info
    size_t returned_size = 0;
    cl_char vendor_name[1024] = {0};
    cl_char device_name[1024] = {0};
    size_t wg_max;
    err_num = clGetDeviceInfo(device, CL_DEVICE_VENDOR, sizeof(vendor_name),
                             vendor_name, &returned_size);
    err_num |= clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name),
                              device_name, &returned_size);
    err_num |= clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(wg_max),
                              &wg_max, &returned_size);
    if (err_num != CL_SUCCESS)
    {
        cout << "Name fetch fail" << endl;
        exit(err_num);
    }
    //printf("Connecting to %s %s...\n", vendor_name, device_name);
    cout << "Connecting to " << vendor_name << " " << device_name << "..." << endl;
    //printf("Max work group size: %"PRIuPTR"\n", wg_max);
    cout << "Max work group size: " << wg_max << endl;
    //local_work_size = wg_max;
    local_work_size = 256;

    // queue setup
    queue = clCreateCommandQueue(   ctx,
                                    device,
                                    0,
                                    &err_num);
    if (err_num != CL_SUCCESS)
    {
        cout << "queue fail" << endl;
        exit(err_num);
    }

    // prog setup
    const char* source = Kernels();
    prog = clCreateProgramWithSource(ctx, 1, &source, NULL, &err_num);
    if (err_num != CL_SUCCESS)
    {
        cout << "compile fail" << endl;
        exit(err_num);
    }

    // build program
    err_num = clBuildProgram(prog, 1, &device, "-cl-fast-relaxed-math -cl-mad-enable", NULL, NULL);
    if (err_num != CL_SUCCESS)
    {
        cout << "build fail " << err_num << endl;
        // Determine the size of the log
        size_t log_size;
        clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

        // Allocate memory for the log
        char *log = (char *) malloc(log_size);

        // Get the log
        clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);

        // Print the log
        //printf("%s\n", log);
        cout << log << endl;
        exit(err_num);
    }
    return 0;
}

int compile_kernel(const char* kernel_name, cl_kernel &kernel)
{
    // kernel setup
    kernel = clCreateKernel(prog, kernel_name, &err_num);
    if (err_num != CL_SUCCESS)
    {
        cout << "make kernel fail" << endl;
        cout << err_num << endl;
        exit(err_num);
    }
    return 0;
}
