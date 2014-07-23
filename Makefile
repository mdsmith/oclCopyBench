UNAME := $(shell uname)
CXX = clang++
ifeq ($(UNAME), Linux)
CXX = g++
endif

ifeq ($(UNAME), Linux)
linker_flags = -lOpenCL
else
linker_flags = -framework OpenCL
endif

platform_extras = 

ifdef AMDAPPSDKROOT
platform_includes = $(AMDAPPSDKROOT)/include
platform_links = $(AMDAPPSDKROOT)/lib
platform_extras += -I$(platform_includes) -L$(platform_links) -O3
endif

# omp and clang don't get along
#all_flags = -O3 -Wall -Werror -Wno-unknown-pragmas

.PHONY: all clean
all: regular
# zero and pinned planned

#core_objs =

###### Core Test ######

copy.o: copy.cpp
	$(CXX) -c -o copy.o copy.cpp $(platform_extras)

regular: copy.o
	$(CXX) -o copy $^ $(linker_flags)

#$(CXX) $(linker_flags) -o copy $^

###### ETC ######

%.o: %.cpp
	$(CXX) $(linker_flags) -c -o $@ $<

clean:
	-rm -f *.o copy
