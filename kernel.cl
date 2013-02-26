STRINGIFY(

__kernel void floatTest(__global float* buf1,
                        __global int* buf2
                        )
{
    int index = get_global_id(0);
    float temp1 = buf1[index];
    int temp2 = buf2[index];
    buf1[index] = temp1 + 1.0;
    buf2[index] = temp2 + 1;
};

__kernel void doubleTest(__global double* buf1,
                        __global int* buf2
                        )
{
    int index = get_global_id(0);
    double temp1 = buf1[index];
    int temp2 = buf2[index];
    buf1[index] = temp1 + 1.0;
    buf2[index] = temp2 + 1;
};

__kernel void ultraTest(__global UltraFloat* buf1
                        )
{
    int index = get_global_id(0);
    UltraFloat temp = buf1[index];
    float mantissa = temp.mantissa;
    int base = temp.base;
    int exponent = temp.exponent;
    temp.mantissa = mantissa + 1.0;
    //temp.base = base + 1;
    temp.exponent = exponent + 1;
    buf1[index] = temp;
};

__kernel void tenTest(__global TenFloat* buf1
                        )
{
    int index = get_global_id(0);
    TenFloat temp = buf1[index];
    float mantissa = temp.mantissa;
    int exponent = temp.exponent;
    temp.mantissa = mantissa + 1.0;
    temp.exponent = exponent + 1;
    buf1[index] = temp;
};

);
