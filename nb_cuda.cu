#include <cuda.h>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <cfloat>
#include "io.hpp"
#include "evaluate.h"

#define CUDA_CHECK_RETURN(value) { \
               cudaError_t _m_cudaStat = value;\
               if (_m_cudaStat != cudaSuccess) {\
                       fprintf(stderr, "Error %s at line %d in file %s\n",\
                                       cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);\
                                       exit(1);\
               }}

#define SIZE_TRAIN 16
#define SIZE_CLASS 256
/*
 __global__ void learning_kernel(double *probClass, double *probMatrix, int *freqClassVector, double *matrixTermFreq, double *totalFreqClassVector, int numClasses, int numDocs, int numTerms, int totalTerms, double alpha) {

 int idx = blockIdx.x*blockDim.x + threadIdx.x;
 int i;
 double prob;
 if(idx < numTerms){
 if(idx == 0){
 for(i = 0; i < numClasses; i++){
 probClass[i] = ((double)(freqClassVector[i]+alpha))/((double)(numDocs+alpha*numClasses));
 }
 }
 for(i = 0; i < numClasses; i++){
 prob = (matrixTermFreq[i*numTerms + idx] + alpha) / (totalFreqClassVector[i] + alpha*totalTerms);
 probMatrix[i*numTerms + idx] = prob;
 }
 }
 }
 */
/*
 __global__ void trainning_kernel(double *probClass, double *probMatrix, int *docTestIndexVector, int *docTestVector, double *docTestFreqVector, double *probClasse, int numClasses, int numTerms, int numDocsTest, double* modeloNB, double lambda, double *freqTermVector, double totalTermFreq) {
 int idx = blockIdx.x*blockDim.x + threadIdx.x;
 int d,c,t;
 int term;
 double freq, prob, sumProb;

 double  maiorProb;

 if(idx < numDocsTest){
 d = idx;
 for(c=0;c<numClasses;c++){
 sumProb = log(probClass[c]);
 int inicio = docTestIndexVector[d];
 int fim = docTestIndexVector[d+1];
 for(t=inicio;t<fim;t++){
 term = docTestVector[t];
 freq = docTestFreqVector[t];
 prob = probMatrix[c*numTerms + term];
 prob = lambda*(freqTermVector[term]/totalTermFreq) + (1.0 - lambda)*prob;
 //if(freqTermVector[term] != 0)
 sumProb += freq*log(prob);
 }
 if(c == 0){
 maiorProb = sumProb;
 }
 if(sumProb > maiorProb){
 maiorProb = sumProb;
 }

 modeloNB[d*numClasses + c] = sumProb;
 }
 probClasse[d] = maiorProb;
 }
 }
 */
__global__ void trainning_kernel2(int *freqClassVector, double *matrixTermFreq,
		double* totalFreqClassVector, int *docTestIndexVector,
		int *docTestVector, double *docTestFreqVector, double *probClasse,
		int numClasses, int numTerms, int numDocsTest, double *freqTermVector,
		double totalTermFreq, int totalTerms, double lambda, double alpha,
		int numDocs, double *modeloNB) {

	int vecs, len, term;
	double freq;
	double prob, nt, maiorProb;
	extern __shared__ double temp[]; // used to hold segment of the vector (size nthreads)
	// plus 3 integers (vecs, len, partial sum) at the end
	int tid = threadIdx.x;

	if (tid == 0) {  // thread 0 calculates vecs and len
		//vecs - vector size
		temp[blockDim.x + 1] = (docTestIndexVector[blockIdx.x + 1]
				- docTestIndexVector[blockIdx.x]);
		// len - number of segments (size nthreads) of the vector
		if(temp[blockDim.x + 1] > blockDim.x)
    		temp[blockDim.x + 2] = ceil(temp[blockDim.x + 1] / (float) blockDim.x);
    else
    		temp[blockDim.x + 2] = 1.0;

		maiorProb = -99999.9;
	}
	__syncthreads();

	vecs = temp[blockDim.x + 1]; // communicate vecs and len's values to other threads
	len = (int) temp[blockDim.x + 2];

	for (int c = 0; c < numClasses; c++) {
		if (tid == 0) {
			// partial sum initialization
			temp[blockDim.x + 3] = log(
					(freqClassVector[c] + alpha)
							/ (numDocs + alpha * numClasses));
		}
		__syncthreads();
		for (int b = 0; b < len; b++) { // loop through 'len' segments
			// first, each thread loads data into shared memory
			if ((b * blockDim.x + tid) >= vecs) // check if outside 'vec' boundary
				temp[tid] = 0.0;
			else {
				term = docTestVector[docTestIndexVector[blockIdx.x]
						+ b * blockDim.x + tid];
        if(freqTermVector[term] != 0){
			      freq = docTestFreqVector[docTestIndexVector[blockIdx.x]
					        + b * blockDim.x + tid];
			      prob = (matrixTermFreq[c * numTerms + term] + alpha)
					        / (totalFreqClassVector[c] + alpha * totalTerms);
			      nt = freqTermVector[term] / totalTermFreq;
			      prob = lambda * nt + (1.0 - lambda) * prob;
			      temp[tid] = freq * log(prob);
        }
        else{
          temp[tid] = 0.0;
        }
			}
			__syncthreads();

			// next, perform binary tree reduction on shared memory
			for (int d = blockDim.x >> 1; d > 0; d >>= 1) {
				if (tid < d)
					temp[tid] += (tid + d) >= vecs ? 0.0 : temp[tid + d];
				__syncthreads();
			}

			// first thread puts partial result into shared memory
			if (tid == 0) {
				temp[blockDim.x + 3] += temp[0];
			}
			__syncthreads();
		}
		// finally, first thread puts result into global memory
		if (tid == 0) {
			modeloNB[blockIdx.x * numClasses + c] = temp[blockDim.x + 3];
			if (c == 0) {
				maiorProb = temp[blockDim.x + 3];
			} else if (temp[blockDim.x + 3] > maiorProb) {
				maiorProb = temp[blockDim.x + 3];
			}
		}
		__syncthreads();
	}
	if (tid == 0) {
		probClasse[blockIdx.x] = maiorProb;
	}
}

__global__ void super_parent_freq(int *docIndexVector, int *docVector,
		double *docFreqVector, int *docClassVector, double *totalTermClassSp,
		int numTerms, int numDocs, int totalTerms, int numClasses) {

	int sp = blockIdx.x * blockDim.x + threadIdx.x;
	int d, t, t2;
	int term;
	double freq;

	if (sp < numTerms) {
		for (int c = 0; c < numClasses; c++)
			totalTermClassSp[c * numTerms + sp] = 0.0;
		for (d = 0; d < numDocs; d++) {
			int clas = docClassVector[d];
			int inicio = docIndexVector[d];
			int fim = docIndexVector[d + 1];

			for (t = inicio; t < fim; t++) {
				term = docVector[t];
				freq = docFreqVector[t];
				if (term == sp && freq > 0) {
					for (t2 = inicio; t2 < fim; t2++) {
						term = docVector[t2];
						freq = docFreqVector[t2];
						if (term != sp)
							totalTermClassSp[clas * numTerms + sp] =
									totalTermClassSp[clas * numTerms + sp]
											+ freq;
					}
				}
			}
		}
	}
}

__global__ void find_sp_kernel(int *docIndexVector, int *docVector,
		double *docFreqVector, int numClasses, int numTerms, int numDocs,
		int totalTerms, int *hasSp, int sp) {

	int vecs, len, term;
	double freq;
	__shared__ int aux[2]; // used to hold segment of the vector (size nthreads)
	// plus 3 integers (vecs, len, partial sum) at the end
	int tid = threadIdx.x;

	if (tid == 0) {  // thread 0 calculates vecs and len
		//vecs - vector size
		aux[blockDim.x + 1] = (docIndexVector[blockIdx.x + 1]
				- docIndexVector[blockIdx.x]);
		// len - number of segments (size nthreads) of the vector
		if(aux[blockDim.x + 1] > blockDim.x)
    		aux[blockDim.x + 2] = ceil(aux[blockDim.x + 1] / (float) blockDim.x);
    else
    		aux[blockDim.x + 2] = 1.0;
		hasSp[blockIdx.x] = 0;
	}
	__syncthreads();

	vecs = aux[blockDim.x + 1]; // communicate vecs and len's values to other threads
	len = aux[blockDim.x + 2];

	for (int b = 0; b < len; b++) { // loop through 'len' segments
		// first, each thread loads data into shared memory
		if ((b * blockDim.x + tid) < vecs) { // check if outside 'vec' boundary
			term = docVector[docIndexVector[blockIdx.x] + b * blockDim.x + tid];
			freq = docFreqVector[docIndexVector[blockIdx.x] + b * blockDim.x
					+ tid];
			if (term == sp && freq > 0) {
				hasSp[blockIdx.x] = 1;
			}
		}
		__syncthreads();
	}
}

__global__ void super_parent_train(int *docIndexVector, int *docVector,
		double *docFreqVector, int *docClassVector, double *totalTermClassSp,
		int numTerms, int numDocs, int totalTerms, int numClasses,
		double* probSp, int sp, double alpha, int *hasSp) {

	int termId = blockIdx.x * blockDim.x + threadIdx.x;
	int i, d, t;
	int term;
	double freq;

	if (termId < numTerms) {
		for (i = 0; i < numClasses; i++) {
			probSp[i * numTerms + termId] = 0;
		}

		//Calculo da Frequencia de um termo dado Super pai e a Classe
		for (d = 0; d < numDocs; d++) {
			if (hasSp[d] == 1) {
				int clas = docClassVector[d];
				int inicio = docIndexVector[d];
				int fim = docIndexVector[d + 1];

				//Procurando Super Pai no documento
				for (t = inicio; t < fim; t++) {
					term = docVector[t];
					freq = docFreqVector[t];
					if (term == termId && freq > 0) {
						probSp[clas * numTerms + termId] = probSp[clas
								* numTerms + termId] + freq;
					}
				}
			}
		}

		for (i = 0; i < numClasses; i++) {
			probSp[i * numTerms + termId] = (probSp[i * numTerms + termId]
					+ alpha)
					/ (totalTermClassSp[i * numTerms + sp]
							+ alpha * (double) totalTerms);

		}
	}
}

/*
 __global__ void super_parent_train(int *docIndexVector, int *docVector,
 double *docFreqVector, int *docClassVector, double *totalTermClassSp,
 int numTerms, int numDocs, int totalTerms, int numClasses,
 double* probSp, int sp, double alpha) {

 int termId = blockIdx.x * blockDim.x + threadIdx.x;
 int i, d, t, t2;
 int term, term2;
 double freq;

 if (termId < numTerms) {
 for (i = 0; i < numClasses; i++) {
 probSp[i * numTerms + termId] = 0;
 }

 //Calculo da Frequencia de um termo dado Super pai e a Classe
 for (d = 0; d < numDocs; d++) {
 int clas = docClassVector[d];
 int inicio = docIndexVector[d];
 int fim = docIndexVector[d + 1];

 //Procurando Super Pai no documento
 for (t = inicio; t < fim; t++) {
 term = docVector[t];
 freq = docFreqVector[t];
 if (term == sp && freq > 0) {
 for (t2 = inicio; t2 < fim; t2++) {
 term2 = docVector[t2];
 freq = docFreqVector[t2];
 if (term2 == termId && freq > 0) {
 probSp[clas * numTerms + termId] = probSp[clas
 * numTerms + termId] + freq;
 }
 }
 }
 }
 }

 for (i = 0; i < numClasses; i++) {
 probSp[i * numTerms + termId] = (probSp[i * numTerms + termId]
 + alpha)
 / (totalTermClassSp[i * numTerms + sp]
 + alpha * (double) totalTerms);

 }
 }
 }
 */
/*
 __global__ void super_parent_predict(double *probSp, double *probMatrix, int *docTestIndexVector, int *docTestVector, double *docTestFreqVector, double *probClassSp, int numClasses, int numTerms, int numDocsTest, double *freqTermVector, double totalTermFreq, int totalTerms, int sp, double *modeloNB, double lambda){
 int idx = blockIdx.x*blockDim.x + threadIdx.x;
 int d,c,t;
 int term;
 double prob, nt, freq;
 double sumProb;

 double  maiorProb;
 if(idx < numDocsTest){
 d = idx;
 for(c=0;c<numClasses;c++){
 sumProb = modeloNB[d*numClasses + c];
 int inicio = docTestIndexVector[d];
 int fim = docTestIndexVector[d+1];
 for(t=inicio;t<fim;t++){
 term = docTestVector[t];
 freq = docTestFreqVector[t];
 prob = probMatrix[c*numTerms + term];
 if(term != sp){
 nt =  freqTermVector[term]/totalTermFreq;
 prob = log(lambda*nt + (1.0 - lambda)*probSp[c*numTerms + term]) - log(lambda*nt + (1.0-lambda)*prob);
 sumProb = sumProb + freq*prob;
 }
 }
 if(c == 0){
 maiorProb = sumProb;
 }
 if(sumProb > maiorProb){
 maiorProb = sumProb;
 }
 }
 probClassSp[d] = maiorProb;
 }
 }
 */

__global__ void super_parent_predict2(double *matrixTermFreq,
		double* totalFreqClassVector, int *docTestIndexVector,
		int *docTestVector, double *docTestFreqVector, double *probClassSp,
		int numClasses, int numTerms, int numDocsTest, double *freqTermVector,
		double totalTermFreq, int totalTerms, double lambda, double alpha,
		int sp, double *modeloNB, double *probSp){

	int vecs, len, term;
	double freq;
	double prob, nt;//, maiorProb;
	//int bestClass;
	extern __shared__ double temp[]; // used to hold segment of the vector (size nthreads)
	// plus 3 integers (vecs, len, partial sum) at the end
	int tid = threadIdx.x;

	if (tid == 0) {  // thread 0 calculates vecs and len
		//vecs - vector size
		temp[blockDim.x + 1] = (docTestIndexVector[blockIdx.x + 1]
				- docTestIndexVector[blockIdx.x]);
		// len - number of segments (size nthreads) of the vector
		if(temp[blockDim.x + 1] > blockDim.x)
    		temp[blockDim.x + 2] = ceil(temp[blockDim.x + 1] / (float) blockDim.x);
    else
    		temp[blockDim.x + 2] = 1.0;
		// partial sum initialization
		//temp[blockDim.x + 3] = 0.0;
	}
	__syncthreads();

	vecs = temp[blockDim.x + 1]; // communicate vecs and len's values to other threads
	len = (int) temp[blockDim.x + 2];

	for (int c = 0; c < numClasses; c++) {
		if (tid == 0) {
			// partial sum initialization
			temp[blockDim.x + 3] = modeloNB[blockIdx.x * numClasses + c];
		}
		__syncthreads();
		for (int b = 0; b < len; b++) { // loop through 'len' segments
			// first, each thread loads data into shared memory
			if ((b * blockDim.x + tid) >= vecs) // check if outside 'vec' boundary
				temp[tid] = 0.0;
			else {
				term = docTestVector[docTestIndexVector[blockIdx.x]
						+ b * blockDim.x + tid];
        if(freqTermVector[term] != 0){
  				freq = docTestFreqVector[docTestIndexVector[blockIdx.x]
  						+ b * blockDim.x + tid];
  				prob = (matrixTermFreq[c * numTerms + term] + alpha)
  						/ (totalFreqClassVector[c] + alpha * totalTerms);
  				nt = freqTermVector[term] / totalTermFreq;
  				if (term != sp) {
  					prob = log(
  							lambda * nt
  									+ (1.0 - lambda)
  											* probSp[c * numTerms + term])
  							- log(lambda * nt + (1.0 - lambda) * prob);
  					temp[tid] = freq * prob;
  				} else {
  					temp[tid] = 0.0;
  				}
        }
        else{
          temp[tid] = 0.0;
        }
			}
			__syncthreads();

			// next, perform binary tree reduction on shared memory
			for (int d = blockDim.x >> 1; d > 0; d >>= 1) {
				if (tid < d)
					temp[tid] += (tid + d) >= vecs ? 0.0 : temp[tid + d];
				__syncthreads();
			}

			// first thread puts partial result into shared memory
			if (tid == 0) {
				temp[blockDim.x + 3] += temp[0];
			}
			__syncthreads();
		}
		// finally, first thread puts result into global memory
		/*
		if (tid == 0) {
			if (c == 0) {
				maiorProb = temp[blockDim.x + 3];
				bestClass = c;
			} else if (temp[blockDim.x + 3] > maiorProb) {
				maiorProb = temp[blockDim.x + 3];
				bestClass = c;
			}
		}
		__syncthreads();
		*/
		if (tid == 0) {
			probClassSp[blockIdx.x * numClasses + c] = temp[blockDim.x + 3];
		}
		__syncthreads();
	}

}

__global__ void super_parent_best_child(double *probSp, double *matrixTermFreq,
		double *totalFreqClassVector, int *docTestIndexVector,
		int *docTestVector, double *docTestFreqVector, double *docClassChild,
		int numClasses, int numTerms, int numDocsTest, double *freqTermVector,
		double totalTermFreq, int totalTerms, int sp, double *modeloNB, int d,
		double lambda, double alpha, int c) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int t;
	int term;
	double prob, nt;
	double sumProb, freq;

	if (idx < numTerms) {
		sumProb = modeloNB[d * numClasses + c];
		int inicio = docTestIndexVector[d];
		int fim = docTestIndexVector[d + 1];
		for (t = inicio; t < fim; t += 1) {
			term = docTestVector[t];
			if (term == idx && idx != sp && freqTermVector[term] != 0) {
				freq = docTestFreqVector[t];
				prob = (matrixTermFreq[c * numTerms + term] + alpha)
						/ (totalFreqClassVector[c] + alpha * totalTerms);
				nt = freqTermVector[term] / totalTermFreq;
				prob = log(
						lambda * nt
								+ (1.0 - lambda) * probSp[c * numTerms + term])
						- log(lambda * nt + (1.0 - lambda) * prob);
				sumProb = sumProb + freq * prob;
			}
		}
		docClassChild[d * numTerms + idx] = sumProb;
	}
}
/*
__global__ void super_parent_update(double *probSp, double *matrixTermFreq,
		double *totalFreqClassVector, int *docTestIndexVector,
		int *docTestVector, double *docTestFreqVector, int numClasses,
		int numTerms, int numDocsTest, double *freqTermVector,
		int totalTermFreq, int totalTerms, int filho, double *modeloNB, int d,
		double lambda, double alpha, int c) {

	int vecs, len, term;
	double freq;
	double prob, nt;
	__shared__ int aux[2];
	int tid = threadIdx.x;

	if (tid == 0) {  // thread 0 calculates vecs and len
		//vecs - vector size
		aux[blockDim.x + 1] =
				(docTestIndexVector[d + 1] - docTestIndexVector[d]);
		// len - number of segments (size nthreads) of the vector
		aux[blockDim.x + 2] = ceil(aux[blockDim.x + 1] / (float) blockDim.x);
	}
	__syncthreads();

	vecs = aux[blockDim.x + 1]; // communicate vecs and len's values to other threads
	len = (int) aux[blockDim.x + 2];

	for (int b = 0; b < len; b++) { // loop through 'len' segments
		// first, each thread loads data into shared memory
		if ((b * blockDim.x + tid) < vecs) { // check if outside 'vec' boundary
			term = docTestVector[docTestIndexVector[d] + b * blockDim.x + tid];
			freq = docTestFreqVector[docTestIndexVector[d] + b * blockDim.x
					+ tid];
			prob = (matrixTermFreq[c * numTerms + term] + alpha)
					/ (totalFreqClassVector[c] + alpha * totalTerms);
			nt = freqTermVector[term] / totalTermFreq;
			if (term == filho) {
				prob = log(
						lambda * nt
								+ (1.0 - lambda) * probSp[c * numTerms + term])
						- log(lambda * nt + (1.0 - lambda) * prob);
				modeloNB[d * numClasses + c] += freq * prob;
			}
		}
		__syncthreads();
	}
}
*/
//Segunda opcao!
__global__ void super_parent_update(double *probSp, double *matrixTermFreq,
		double *totalFreqClassVector, int *docTestIndexVector,
		int *docTestVector, double *docTestFreqVector, int numClasses,
		int numTerms, int numDocsTest, double *freqTermVector,
		double totalTermFreq, int totalTerms, int filho, double *modeloNB, int d,
		double lambda, double alpha, int c) {

	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	int t;
	int term;
	double prob, nt;
	double freq;

	if(tid < numTerms){
		int inicio = docTestIndexVector[d];
		int fim = docTestIndexVector[d + 1];
		for (t = inicio; t < fim; t++) {
			term = docTestVector[t];
			if (tid == term && term == filho && freqTermVector[term] != 0) {
				freq = docTestFreqVector[t];
				prob = (matrixTermFreq[c * numTerms + term] + alpha) / (totalFreqClassVector[c] + alpha * totalTerms);
				nt = freqTermVector[term] / totalTermFreq;
				prob = log(lambda * nt + (1.0 - lambda) * probSp[c * numTerms + term]) - log(lambda * nt + (1.0 - lambda) * prob);
				modeloNB[d * numClasses + c] += freq * prob;
			}
		}
	}
}

extern "C" {

double nb_gpu(const char* filenameTreino, const char* filenameTeste,
		int numDocs, int numClasses, int numTerms, int numDocsTest,
		int numTermsTest, double alpha, double lambda, int cudaDevice) {

	cudaDeviceReset();
	cudaSetDevice(cudaDevice);
	clock_t begin, endT, end;
	//double iTreino, fTreino;
	begin = clock();
	int block_size, n_blocks;
	int *docTestIndexVector = (int*) malloc((numDocsTest + 1) * sizeof(int)); //Alterei numDocs para numDocsTest
	int *docTestVector = NULL;
	double *docTestFreqVector = NULL;
	int *docClassVector = (int*) malloc(numDocs * sizeof(int));

	int *freqClassVector = (int*) malloc(numClasses * sizeof(int));
	double *totalFreqClassVector = (double*) malloc(
			numClasses * sizeof(double));
	double *matrixTermFreq = (double*) malloc(
			(numTerms * numClasses) * sizeof(double));
	double *freqTermVector = (double*) malloc((numTerms) * sizeof(double));
	double totalTermFreq = 0.0;
	int totalTerms = 0;

	for (int i = 0; i < numClasses; i++) {
		totalFreqClassVector[i] = 0.0;
		freqClassVector[i] = 0;
		for (int j = 0; j < numTerms; j++) {
			matrixTermFreq[i * numTerms + j] = 0.0;
		}
	}
	for (int j = 0; j < numTerms; j++) {
		freqTermVector[j] = 0.0;
	}
	//iTreino = tempoAtual();

	int *docIndexVector = (int*) malloc((numDocs + 1) * sizeof(int));
	int *docVector = NULL;
	double *docFreqVector = NULL;

	set<int> vocabulary;
	docVector = readTrainDataSP(filenameTreino, docIndexVector,
			totalFreqClassVector, freqClassVector, freqTermVector,
			&totalTermFreq, numClasses, numTerms, &totalTerms, matrixTermFreq,
			vocabulary, &docFreqVector, docClassVector);

	double *matrixTermFreq_D;
	cudaMalloc((void **) &matrixTermFreq_D,
			sizeof(double) * (numTerms * numClasses));
	cudaMemcpy(matrixTermFreq_D, matrixTermFreq,
			sizeof(double) * (numTerms * numClasses), cudaMemcpyHostToDevice);

//REMOVER
//        double *probClass = (double*) malloc (numClasses*sizeof(double));
//        double *probMatrix = (double*) malloc (numClasses*numTerms*sizeof(double));

	int *freqClassVector_D;
	cudaMalloc((void **) &freqClassVector_D, sizeof(int) * numClasses);
	cudaMemcpy(freqClassVector_D, freqClassVector, sizeof(int) * numClasses,
			cudaMemcpyHostToDevice);
	double *totalFreqClassVector_D;
	cudaMalloc((void **) &totalFreqClassVector_D, sizeof(double) * numClasses);
	cudaMemcpy(totalFreqClassVector_D, totalFreqClassVector,
			sizeof(double) * numClasses, cudaMemcpyHostToDevice);

//REMOVER
//        double *probClass_D;
//        cudaMalloc ((void **) &probClass_D, sizeof(double)*numClasses);
//        double *probMatrix_D;
//        cudaMalloc ((void **) &probMatrix_D, sizeof(double)*(numClasses*numTerms));

	double *freqTermVector_D;
	cudaMalloc((void **) &freqTermVector_D, sizeof(double) * numTerms);
	cudaMemcpy(freqTermVector_D, freqTermVector, sizeof(double) * numTerms,
			cudaMemcpyHostToDevice);

//REMOVER
//        block_size = 384;
//        n_blocks = (numTerms+1)/block_size + ((numTerms+1)%block_size == 0 ? 0:1);
//        learning_kernel<<< n_blocks, block_size >>>(probClass_D,probMatrix_D,freqClassVector_D,matrixTermFreq_D,totalFreqClassVector_D,numClasses,numDocs,numTerms,totalTerms, alpha);
//        cudaDeviceSynchronize();

//        cudaFree(freqClassVector_D);
//        free(freqClassVector);

//        cudaFree(totalFreqClassVector_D);
//        free(totalFreqClassVector);

//        cudaFree(matrixTermFreq_D);
//        free(matrixTermFreq);

//        free(probClass);

	/* ============================ TESTE ================================*/
	int *realClass = (int*) malloc((numDocsTest + 1) * sizeof(int));

	docTestVector = readTestData(filenameTeste, docTestIndexVector, realClass,
			&docTestFreqVector);

	int *docTestIndexVector_D;
	cudaMalloc((void **) &docTestIndexVector_D,
			sizeof(int) * (numDocsTest + 1));
	cudaMemcpy(docTestIndexVector_D, docTestIndexVector,
			sizeof(int) * (numDocsTest + 1), cudaMemcpyHostToDevice);
	int *docTestVector_D;
	cudaMalloc((void **) &docTestVector_D,
			sizeof(int) * docTestIndexVector[numDocsTest]);
	cudaMemcpy(docTestVector_D, docTestVector,
			sizeof(int) * docTestIndexVector[numDocsTest],
			cudaMemcpyHostToDevice);
	double *docTestFreqVector_D;
	cudaMalloc((void **) &docTestFreqVector_D,
			sizeof(double) * docTestIndexVector[numDocsTest]);
	cudaMemcpy(docTestFreqVector_D, docTestFreqVector,
			sizeof(double) * docTestIndexVector[numDocsTest],
			cudaMemcpyHostToDevice);

	double *probClasse = (double*) malloc((numDocsTest) * sizeof(double));
	double *probClasse_D;
	cudaMalloc((void **) &probClasse_D, sizeof(double) * (numDocsTest));

	double* modeloNB = (double*) malloc(
			(numClasses * (numDocsTest)) * sizeof(double));
	double* modeloNB_D;
	cudaMalloc((void **) &modeloNB_D,
			sizeof(double) * (numClasses * (numDocsTest)));

//        block_size = 384;
//        n_blocks = (numDocsTest+1)/block_size + ((numDocsTest+1)%block_size == 0 ? 0:1);
//        trainning_kernel<<< n_blocks, block_size >>>(probClass_D, probMatrix_D, docTestIndexVector_D, docTestVector_D, docTestFreqVector_D, probClasse_D, numClasses, numTerms, numDocsTest, modeloNB_D, lambda, freqTermVector_D, totalTermFreq);
//        cudaDeviceSynchronize();
	block_size = SIZE_CLASS;
	n_blocks = numDocsTest;
	trainning_kernel2<<<n_blocks, block_size, (block_size + 3) * sizeof(double)>>>(
			freqClassVector_D, matrixTermFreq_D, totalFreqClassVector_D,
			docTestIndexVector_D, docTestVector_D, docTestFreqVector_D,
			probClasse_D, numClasses, numTerms, numDocsTest, freqTermVector_D,
			totalTermFreq, totalTerms, lambda, alpha, numDocs, modeloNB_D);
	cudaDeviceSynchronize();

	cudaMemcpy(probClasse, probClasse_D, sizeof(double) * (numDocsTest),
			cudaMemcpyDeviceToHost);
	cudaMemcpy(modeloNB, modeloNB_D,
			sizeof(double) * (numClasses * numDocsTest),
			cudaMemcpyDeviceToHost);

	double valorFinal, maiorProb;
	int maiorClasseProb;
	int *predictClass = (int*) malloc((numDocsTest) * sizeof(int));

	for (int d = 0; d < numDocsTest; d++) {
		maiorProb = -9999999.0;
		for (int c = 0; c < numClasses; c++) {
			if (modeloNB[d * numClasses + c] > maiorProb) {
				maiorClasseProb = c;
				maiorProb = modeloNB[d * numClasses + c];
			}
		}
		predictClass[d] = maiorClasseProb;
	}
	valorFinal = evaluate(realClass, predictClass, numDocsTest, 1);
	cerr << "Resultado Naive Bayes "
			<< evaluate(realClass, predictClass, numDocsTest, 1) * 100 << " "
			<< evaluate(realClass, predictClass, numDocsTest, 0) * 100 << endl;

  cout << "Resultado Naive Bayes "
      << evaluate(realClass, predictClass, numDocsTest, 1) * 100 << " "
      << evaluate(realClass, predictClass, numDocsTest, 0) * 100 << endl;

	cudaFree(probClasse_D);
//        cudaFree(probClass_D);

	/* ============================ SP-TAN ================================*/

	int *docIndexVector_D;
	cudaMalloc((void **) &docIndexVector_D, (numDocs + 1) * sizeof(int));
	cudaMemcpy(docIndexVector_D, docIndexVector, (numDocs + 1) * sizeof(int),
			cudaMemcpyHostToDevice);
	int *docVector_D;
	cudaMalloc((void **) &docVector_D, sizeof(int) * docIndexVector[numDocs]);
	cudaMemcpy(docVector_D, docVector, sizeof(int) * docIndexVector[numDocs],
			cudaMemcpyHostToDevice);
	double *docFreqVector_D;
	cudaMalloc((void **) &docFreqVector_D,
			sizeof(double) * docIndexVector[numDocs]);
	cudaMemcpy(docFreqVector_D, docFreqVector,
			sizeof(double) * docIndexVector[numDocs], cudaMemcpyHostToDevice);
	int *docClassVector_D;
	cudaMalloc((void **) &docClassVector_D, sizeof(int) * numDocs);
	cudaMemcpy(docClassVector_D, docClassVector, sizeof(int) * numDocs,
			cudaMemcpyHostToDevice);

	free(docIndexVector);
	free(docVector);
	free(docFreqVector);
	free(docClassVector);

	double *totalTermClassSp_D;
	cudaMalloc((void **) &totalTermClassSp_D,
			sizeof(double) * numClasses * numTerms);
	double * totalTermClassSp = (double*) malloc(
			numClasses * numTerms * sizeof(double));
	double *probSp = (double*) malloc(numClasses * numTerms * sizeof(double));
	int sp;
	double *probSp_D;
	cudaMalloc((void **) &probSp_D, sizeof(double) * numClasses * numTerms);

	double *probClassSp = (double*) malloc(
			numDocsTest * numClasses * sizeof(double));
	double *probClassSp_D;
	cudaMalloc((void **) &probClassSp_D,
			sizeof(double) * numDocsTest * numClasses);

	double *docClassChild = (double*) malloc(
			numDocsTest * numTerms * sizeof(double));
	double *docClassChild_D;
	cudaMalloc((void **) &docClassChild_D,
			sizeof(double) * numDocsTest * numTerms);

	block_size = 384;
	n_blocks = (numTerms + 1) / block_size
			+ ((numTerms + 1) % block_size == 0 ? 0 : 1);
	super_parent_freq<<<n_blocks, block_size>>>(docIndexVector_D, docVector_D,
			docFreqVector_D, docClassVector_D, totalTermClassSp_D, numTerms,
			numDocs, totalTerms, numClasses);
//	cudaDeviceSynchronize();
	cudaMemcpy(totalTermClassSp, totalTermClassSp_D,
			sizeof(double) * numClasses * numTerms, cudaMemcpyDeviceToHost);
	for (int c = 0; c < numClasses; c++) {
		cerr << c << " " << totalTermClassSp[c * numTerms + 0] << endl;
	}
	free(totalTermClassSp);

	double *highestProbs = (double*) malloc(
			(numDocsTest) * numClasses * sizeof(double));
	int *bestSp = (int*) malloc((numDocsTest) * numClasses * sizeof(int));
	for (int d = 0; d < numDocsTest; d++) {
		for (int c = 0; c < numClasses; c++) {
			highestProbs[d * numClasses + c] = -999999.9;
			bestSp[d * numClasses + c] = -1;
		}
	}

	int *hasSp_D;
	cudaMalloc((void**) &hasSp_D, sizeof(int) * numDocs);
	int * hasSp = (int*) malloc(numDocs * sizeof(int));
	//cerr << "----SP----" << numberOfSp << endl;
	for (set<int>::iterator spIt = vocabulary.begin(); spIt != vocabulary.end();
			++spIt) {
		sp = *spIt;
		//Constroi a tabela de probabilidades para o Super Pai
		//block_size = 384;
		//n_blocks = (numTerms + 1) / block_size
		//		+ ((numTerms + 1) % block_size == 0 ? 0 : 1);
		//super_parent_train<<<n_blocks, block_size>>>(docIndexVector_D,
		//		docVector_D, docFreqVector_D, docClassVector_D,
		//		totalTermClassSp_D, numTerms, numDocs, totalTerms, numClasses,
		//		probSp_D, sp, alpha);

		block_size = SIZE_TRAIN;
		n_blocks = numDocs;
		find_sp_kernel<<<n_blocks, block_size>>>(docIndexVector_D, docVector_D,
				docFreqVector_D, numClasses, numTerms, numDocs, totalTerms,
				hasSp_D, sp);
		/*
		 if(sp == 2){
		 cerr<<"Hey\n";
		 //for(int dt=0;dt<numDocs;dt++){
		 //	hasSp[dt]=0;
		 //}
		 cudaMemcpy(hasSp, hasSp_D, sizeof(int)*numDocs, cudaMemcpyDeviceToHost);
		 for(int dt=0;dt<numDocs;dt++){
		 cerr<<dt<<" "<<hasSp[dt]<<endl;
		 }
		 }
		 */
		block_size = 384;
		n_blocks = (numTerms + 1) / block_size
				+ ((numTerms + 1) % block_size == 0 ? 0 : 1);
		super_parent_train<<<n_blocks, block_size>>>(docIndexVector_D,
				docVector_D, docFreqVector_D, docClassVector_D,
				totalTermClassSp_D, numTerms, numDocs, totalTerms, numClasses,
				probSp_D, sp, alpha, hasSp_D);

		cudaDeviceSynchronize();

		//Classificação dos documentos baseado no modelo de classificação construído dado o Super Pai
//            block_size = 384;
//            n_blocks = (numDocsTest+1)/block_size + ((numDocsTest+1)%block_size == 0 ? 0:1);
//            super_parent_predict<<< n_blocks, block_size >>>(probSp_D, probMatrix_D, docTestIndexVector_D, docTestVector_D, docTestFreqVector_D, probClassSp_D, numClasses, numTerms, numDocsTest, freqTermVector_D, totalTermFreq, totalTerms, sp, modeloNB_D, lambda);

		block_size = SIZE_CLASS;
		n_blocks = numDocsTest;
		super_parent_predict2<<<n_blocks, block_size,
				(block_size + 3) * sizeof(double)>>>(matrixTermFreq_D,
				totalFreqClassVector_D, docTestIndexVector_D, docTestVector_D,
				docTestFreqVector_D, probClassSp_D, numClasses, numTerms,
				numDocsTest, freqTermVector_D, totalTermFreq, totalTerms,
				lambda, alpha, sp, modeloNB_D, probSp_D);
		cudaDeviceSynchronize();

		//Avaliação da qualidade de classificação dado o Super Pai
		cudaMemcpy(probClassSp, probClassSp_D, numDocsTest * numClasses * sizeof(double),
				cudaMemcpyDeviceToHost);
		for (int d = 0; d < numDocsTest; d++) {
			for (int c = 0; c < numClasses; c++) {
				if (probClassSp[d * numClasses + c] > highestProbs[d * numClasses + c]) {
					highestProbs[d * numClasses + c] = probClassSp[d * numClasses + c];
					bestSp[d * numClasses + c] = sp;
				}
			}
		}
	}
	free(hasSp);
	endT = clock();

	double maxProb;
	int classe;
	for (int d = 0; d < numDocsTest; d++) {
		maxProb = -99999.9;
		for (int c = 0; c < numClasses; c++) {
			if (maxProb < highestProbs[d * numClasses + c]) {
				maxProb = highestProbs[d * numClasses + c];
				classe = c;
			}
		}
		predictClass[d] = classe;
	}
	cerr << "Melhor SP "
			<< evaluate(realClass, predictClass, numDocsTest, 1) * 100
			<< " " << evaluate(realClass, predictClass, numDocsTest, 0) * 100
			<< " time: " << double(endT - begin) << endl;

  cout << "Melhor SP "
      << evaluate(realClass, predictClass, numDocsTest, 1) * 100
      << " " << evaluate(realClass, predictClass, numDocsTest, 0) * 100
      << " time: " << double(endT - begin) << endl;

  //Escolha dos filhos favoritos
	for (int d = 0; d < numDocsTest; d++) {
		cerr << d << " * " << probClasse[d] << " ";
		for (int c = 0; c < numClasses; c++) {
			cerr << c << " " << highestProbs[d * numClasses + c] << " " << bestSp[d * numClasses + c] << " ";
			block_size = SIZE_TRAIN;
			n_blocks = numDocs;
			find_sp_kernel<<<n_blocks, block_size>>>(docIndexVector_D,
					docVector_D, docFreqVector_D, numClasses, numTerms,
					numDocs, totalTerms, hasSp_D, sp);
			block_size = 384;
			n_blocks = (numTerms + 1) / block_size
					+ ((numTerms + 1) % block_size == 0 ? 0 : 1);
			super_parent_train<<<n_blocks, block_size>>>(docIndexVector_D,
					docVector_D, docFreqVector_D, docClassVector_D,
					totalTermClassSp_D, numTerms, numDocs, totalTerms,
					numClasses, probSp_D, sp, alpha, hasSp_D);
			//cudaMemcpy(probSp, probSp_D, sizeof(double)*numTerms*numClasses,
			//    cudaMemcpyDeviceToHost);
			//cudaDeviceSynchronize();
			block_size = 384;
			n_blocks = (numTerms + 1) / block_size
					+ ((numTerms + 1) % block_size == 0 ? 0 : 1);
			super_parent_best_child<<<n_blocks, block_size>>>(probSp_D,
					matrixTermFreq_D, totalFreqClassVector_D,
					docTestIndexVector_D, docTestVector_D,
					docTestFreqVector_D, docClassChild_D, numClasses,
					numTerms, numDocsTest, freqTermVector_D, totalTermFreq,
					totalTerms, bestSp[d * numClasses + c], modeloNB_D, d,
					lambda, alpha, c);
			cudaMemcpy(docClassChild, docClassChild_D,
					sizeof(double) * numDocsTest * numTerms,
					cudaMemcpyDeviceToHost);

			for (int t = 0; t < numTerms; t++) {
				//Escolha do filho favorito
				if (probClasse[d]  < docClassChild[d * numTerms + t]
						&& t != bestSp[d * numClasses + c]) {
					block_size = 384;
					n_blocks = (numTerms + 1) / block_size
					+ ((numTerms + 1) % block_size == 0 ? 0 : 1);
					super_parent_update<<<n_blocks, block_size>>>(probSp_D,
							matrixTermFreq_D, totalFreqClassVector_D,
							docTestIndexVector_D, docTestVector_D,
							docTestFreqVector_D, numClasses, numTerms,
							numDocsTest, freqTermVector_D, totalTermFreq,
							totalTerms, t, modeloNB_D, d, lambda, alpha, c);
				}
			}
		}
		cerr << endl;
	}

	cudaMemcpy(modeloNB, modeloNB_D, sizeof(double) * numDocsTest * numClasses,
		cudaMemcpyDeviceToHost);

	for (int d = 0; d < numDocsTest; d++) {
		maiorProb = -9999999.0;
		for (int c = 0; c < numClasses; c++) {
			if (modeloNB[d * numClasses + c] > maiorProb) {
				maiorClasseProb = c;
				maiorProb = modeloNB[d * numClasses + c];
			}
		}
		cerr << d << " " << maiorClasseProb << " " << maiorProb << " " << realClass[d] << endl;
		predictClass[d] = maiorClasseProb;
		probClasse[d] = maiorProb;
	}
	valorFinal = evaluate(realClass, predictClass, numDocsTest, 1);

  cerr << "Filhos Favoritos " << evaluate(realClass, predictClass, numDocsTest, 1) * 100 << " "
    << evaluate(realClass, predictClass, numDocsTest, 0) * 100 << " ";

	//fTreino = tempoAtual();

	cout << "Filhos Favoritos " << evaluate(realClass, predictClass, numDocsTest, 1) * 100 << " "
			<< evaluate(realClass, predictClass, numDocsTest, 0) * 100 << " ";

	cudaFree(docIndexVector_D);
	cudaFree(docVector_D);
	cudaFree(docFreqVector_D);
	cudaFree(docClassVector_D);
	cudaFree(docTestIndexVector_D);
	cudaFree(docTestVector_D);
	cudaFree(docTestFreqVector_D);
	cudaFree(freqTermVector_D);
	free(freqTermVector);

	cudaFree(totalFreqClassVector_D);
	free(totalFreqClassVector);

	cudaFree(matrixTermFreq_D);
	free(matrixTermFreq);

	cudaFree(hasSp_D);
	cudaFree(totalTermClassSp_D);

	cudaFree(probClassSp_D);
	free(probClassSp);

	cudaFree(modeloNB_D);
	free(modeloNB);

	cudaFree(probSp_D);
	free(probSp);

	free(realClass);
	free(predictClass);
	free(highestProbs);
	free(bestSp);
	free(docTestIndexVector);
	free(docTestVector);
	free(docTestFreqVector);

	cudaFree(docClassChild_D);
	free(docClassChild);

	end = clock();
	cerr << "Time " << double(end - begin) << endl;
  cout << "Time " << double(end - begin) << endl;

	return valorFinal;
}
}
