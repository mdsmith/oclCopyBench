oclCopyBench
============

Compile:
--------

make

Usage:
------

./copy


Current Tests:
--------------

*   Float + Exponent Arrays
*   Double + Exponent Arrays
*   Struct with three elements Array
*   Struct with two elements Array

Whether or not it makes sense to replace mantissa & exponent arrays with a
single struct array. This goes for both copying the data to the GPU and
accessing it once it is already there.

Current Results:
----------------

Currently the struct arrays are about twice as fast. More testing is needed
to isolate whether or not this is from the number of arrays that need to be
copied (my hypothesis). This would suggest that the time to access the data
from memory is trivial if the memory is only being accessed once.

If this is the case the next test will be to experiment with the different
methods of moving memory to and from the GPU, which may at some point involve
an APU and zero copy transfers (perhaps even a unified memory space).
