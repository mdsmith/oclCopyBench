STRINGIFY(

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
);
