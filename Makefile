CC = gcc
CCC= g++
GPUCC= nvcc
PARAMETERS = -O3 -lm 
CFLAGS = -arch sm_30

main: main.o io.o nb_cuda.o
	$(GPUCC) $(PARAMETERS) -o nb main.o io.o nb_cuda.o

main.o: main.cpp
	$(GPUCC) $(PARAMETERS) -c main.cpp -o main.o

io.o: io.cpp io.hpp
	$(GPUCC) $(PARAMETERS) -c io.cpp -o io.o

nb_cuda.o: nb_cuda.cu nb_cuda.h
	$(GPUCC)  $(CFLAGS) -c nb_cuda.cu -o nb_cuda.o
	

clean:
	rm -f *.o nb *~
