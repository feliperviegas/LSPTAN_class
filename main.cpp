#include <stdio.h>
#include <stdlib.h>
#include "string.h"
#include "io.hpp"
#include "nb_cuda.h"

//#include "nb_cuda.h"

int main(int argc, char *argv[]) {

	int cudaDevice;
	double inicio, final;
	int numDocs, numClasses, numTerms;
	int numDocsTest, numTermsTest;
	double alpha = 1.0, lambda = 0.3;
	/***************************************/

	if (argc != 21) {
		printf(
				"\n\n./nb -c [cuda=1] -nd [NumDocs] -nc [numClasses] -nt [numTerms] -fl [fileTrainning] -ndT [NumDocsTest] -ntT [numTermsTest] -ft [fileTest] -a [alpha] -l [lambda]\n\n");
		exit(0);
	}

	//Parametros
	cudaDevice = atoi(argv[2]);
	numDocs = atoi(argv[4]);
	numClasses = atoi(argv[6]);
	numTerms = atoi(argv[8]);

	numDocsTest = atoi(argv[12]);
	numTermsTest = atoi(argv[14]);

	alpha = atof(argv[18]);
	lambda = atof(argv[20]);

	//Timing
	inicio = tempoAtual();

	double resultado = 0.0;

	//if(!cuda)docClasses = nb_cpu2(argv[10], argv[16], numDocs, numClasses, numTerms, numDocsTest,numTermsTest);
	//else 
	resultado = nb_gpu(argv[10], argv[16], numDocs, numClasses, numTerms,
			numDocsTest, numTermsTest, alpha, lambda, cudaDevice);

//    cout << resultado*100 << endl;
	/*
	 int i;
	 for(i=1;i<=numDocsTest;i++){
	 printf("%d CLASSE=%.0f CLASSE=%.0f:%.50lf\n",i,docClasses[i*3 + 0],docClasses[i*3 +1],docClasses[i*3 +2]);
	 }
	 */
	/*============================= DONE =================================*/
	final = tempoAtual();

	return 0;
}
