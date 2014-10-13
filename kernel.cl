STRINGIFY(

__kernel void floatMatTest(__global float* buf1,
                        __global int* buf2,
                        __global float* buf1b,
                        __global int* buf2b,
                        int nCols
                        )
{
    int index = get_global_id(0);
    int a_offset = (index/64)*64;
    int b_offset = (index%64)*64;
    float sum = 0.0;
    //for (int j = 0; j < nCols; j++) {
        //sum += buf1[a_offset + j] * buf1b[b_offset + j];
    //}
    for (int j = 0; j < nCols;) {
        sum += buf1[a_offset + j] * buf1b[b_offset + j];
        j++;
        sum += buf1[a_offset + j] * buf1b[b_offset + j];
        j++;
        sum += buf1[a_offset + j] * buf1b[b_offset + j];
        j++;
        sum += buf1[a_offset + j] * buf1b[b_offset + j];
        j++;
        sum += buf1[a_offset + j] * buf1b[b_offset + j];
        j++;
        sum += buf1[a_offset + j] * buf1b[b_offset + j];
        j++;
        sum += buf1[a_offset + j] * buf1b[b_offset + j];
        j++;
        sum += buf1[a_offset + j] * buf1b[b_offset + j];
        j++;
    }
    int store_index = index + get_global_size(0);
    buf1[store_index] = sum;
};

__kernel void floatMatLocalTest(__global float* buf1,
                                __global int* buf2,
                                __global float* buf1b,
                                __global int* buf2b
                        )
{
    int index = get_global_id(0);
    // we're using half of buf1 for source, half for sink, global work size is equal to half buf1 size
    int store_index = get_global_id(0) + get_global_size(0);
    __local float aScratch[256];
    __local float bScratch[64*64];
    aScratch[get_local_id(0)] = buf1[get_global_id(0)];
    for (int i = 0; i < 4; i++) {
      bScratch[get_local_id(0) + i*256] = buf1b[get_local_id(0) + i*256];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    float sum = 0.0;
    for (int j = 0; j < 64; j++) {
        sum += aScratch[(get_local_id(0)/64)*64 + j] * bScratch[(get_local_id(0)%64)*64 + j];
    }
    buf1[store_index] = sum;
};

__kernel void floatMatTiledTest(__global float* buf1,
                                __global int* buf2,
                                __global float* buf1b,
                                __global int* buf2b,
                                int block_number,
                                int block_size
                                )
{
  __local float aScratch[256];
  __local float bScratch[256];
  int groupI = get_group_id(0);
  int localI = get_local_id(0);
  // we're using half of buf1 for source, half for sink, global work size is equal to half buf1 size
  int store_index = get_global_size(0) + (groupI/4)*64*16 + (groupI%4)*16 + (localI/16)*16*4 + localI;
  float sum = 0.0;
  /*
  for (int blockI = 0; blockI < 4; blockI++) {
    aScratch[get_local_id(0)] = buf1[(groupI/4)*64*16 + blockI*16 + (localI/16)*16*4 + localI%16];
    bScratch[get_local_id(0)] = buf1b[(groupI%4)*16*64 + blockI*16 + (localI/16)*16*4 + localI%16];
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int lineI = 0; lineI < 16; lineI++) {
      sum += aScratch[(localI/16) * 16 + lineI] * bScratch[(localI%16) * 16 + lineI];
    }
  }
  */
  int a_offset = (groupI/4)*64*16 + (localI/16)*16*4 + localI%16;
  int b_offset = (groupI%4)*16*64 + (localI/16)*16*4 + localI%16;
  int as_offset = (localI/16) * 16;
  int bs_offset = (localI%16) * 16;
  int blockI = 0;
  for (int blockI = 0; blockI < block_number; blockI++) {
  //for (int blockI = 0; blockI < 4; blockI++) {
    aScratch[localI] = buf1[a_offset + blockI*16];
    bScratch[localI] = buf1b[b_offset + blockI*16];
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int lineI = 0; lineI < block_size;) {
    //for (int lineI = 0; lineI < 16; lineI++) {
      sum += aScratch[as_offset + lineI] * bScratch[bs_offset + lineI];
      lineI++;
      sum += aScratch[as_offset + lineI] * bScratch[bs_offset + lineI];
      lineI++;
      sum += aScratch[as_offset + lineI] * bScratch[bs_offset + lineI];
      lineI++;
      sum += aScratch[as_offset + lineI] * bScratch[bs_offset + lineI];
      lineI++;
      sum += aScratch[as_offset + lineI] * bScratch[bs_offset + lineI];
      lineI++;
      sum += aScratch[as_offset + lineI] * bScratch[bs_offset + lineI];
      lineI++;
      sum += aScratch[as_offset + lineI] * bScratch[bs_offset + lineI];
      lineI++;
      sum += aScratch[as_offset + lineI] * bScratch[bs_offset + lineI];
      lineI++;
    }
  }
  buf1[store_index] = sum;
};

__kernel void floatTest(__global float* buf1,
                        __global int* buf2
                        )
{
    int index = get_global_id(0);
    float temp1 = buf1[index];
    int temp2 = buf2[index];
    for (int j = 0; j < 64; j++) {
        temp1 *= 0.8;
    }
    if (temp1 < 0.001) {
      temp1 *= 1000.0;
      temp2 += 3;
    }
    buf1[index] = temp1;
    //buf1[index] = temp1 + 1.0;
    buf2[index] = temp2;
};

__kernel void floatTestNoE(__global float* buf1,
                        int e
                        )
{
    int index = get_global_id(0);
    float temp1 = buf1[index];
    for (int j = 0; j < 64; j++) {
        temp1 += temp1 * 0.001;
    }
    buf1[index] += temp1;
};

__kernel void eRebalance( __global float* buf1,
                          __global int* e
                          )
{
    int index = get_global_id(0);
    float temp1 = buf1[index];
    int temp2 = e[0];
    float max_n = temp1;
    if (index == 0) {
      float max_n = temp1;
      for (int i = 0; i < get_global_size(0); i++) {
        if (buf1[i] > max_n) {
          max_n = buf1[i];
        }
      }
    }
    buf1[index] = temp1;
    if (index == 0) {
      //e[0] = temp2 + 1;
      e[0] = (int)max_n;
    }
};

__kernel void doubleTest(__global double* buf1,
                        __global int* buf2
                        )
{
    int index = get_global_id(0);
    double temp1 = buf1[index];
    int temp2 = buf2[index];
    for (int j = 0; j < 64; j++) {
        temp1 += temp1 * 0.001;
    }
    buf1[index] += temp1;
    //buf1[index] = temp1 + 1.0;
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
    //temp.mantissa = mantissa + 1.0;
    for (int j = 0; j < 64; j++) {
        temp.mantissa += temp.mantissa * 0.001;
    }
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
    for (int j = 0; j < 64; j++) {
      mantissa += mantissa * 0.001;
    }
    temp.mantissa += mantissa;
    temp.exponent = exponent + 1;
    buf1[index] = temp;
};

);
