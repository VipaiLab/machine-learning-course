TF_ROOT=../tf_dist

BIN_SRCS=test.cpp  camera.cpp
COMM_SRCS+=tensorflow_mtcnn.cpp comm_lib.cpp  utils.cpp



#
# opencv settings
#

OPENCV_LDFLAGS=$(shell pkg-config --libs opencv)
OPENCV_CFLAGS=$(shell pkg-config --cflags opencv)


CXXFLAGS += $(OPENCV_CFLAGS) 
LIBS += $(OPENCV_LDFLAGS)

#
# tensorflow ettings
#


TF_INCS += -I$(TF_ROOT)/include
CXXFLAGS += $(TF_INCS) 
LIBS+=-Wl,-rpath,$(TF_ROOT)/lib -L$(TF_ROOT)/lib -ltensorflow


CXX := g++
CXXFLAGS+= -Wall  -ggdb -std=c++11 


COMM_OBJS=$(COMM_SRCS:.cpp=.o)
BIN_OBJS=$(BIN_SRCS:.cpp=.o)
BIN_EXES=$(BIN_SRCS:.cpp=)


default : $(BIN_EXES)

$(BIN_EXES) : $(COMM_OBJS)

$(BIN_EXES):%:%.o


%:%.o
	$(CXX) $< -o $@ $(LDFLAGS) $(COMM_OBJS) $(LIBS)

%.o : %.cpp 
	$(CXX) $(CXXFLAGS) -c $< -o $@ 

clean:
	rm -f $(BIN_EXES) *.o

.PHONY : all clean

