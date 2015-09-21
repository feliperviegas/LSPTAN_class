#include "stdio.h"

#ifndef COMPONENTES_H
#define COMPONENTES_H
#ifdef __cplusplus
extern "C" {
#endif


void trainning_kernel2(int *freqClassVector, double *matrixTermFreq,
		double* totalFreqClassVector, int *docTestIndexVector,
		int *docTestVector, double *docTestFreqVector, double *probClasse,
		int numClasses, int numTerms, int numDocsTest, double *freqTermVector,
		double totalTermFreq, int totalTerms, double lambda, double alpha,
		int numDocs, double *modeloNB);
void super_parent_freq(int *docIndexVector, int *docVector,
		double *docFreqVector, int *docClassVector, double *totalTermClassSp,
		int numTerms, int numDocs, int totalTerms, int numClasses);

void super_parent_predict2(double *matrixTermFreq, double* totalFreqClassVector,
		int *docTestIndexVector, int *docTestVector, double *docTestFreqVector,
		double *probClasseSp, int numClasses, int numTerms, int numDocsTest,
		double *freqTermVector, double totalTermFreq, int totalTerms,
		double lambda, double alpha, int sp, double *modeloNB, double *probSp);
void super_parent_best_child(double *probSp, double *matrixTermFreq,
		double *totalFreqClassVector, int *docTestIndexVector,
		int *docTestVector, double *docTestFreqVector, double *docClassChild,
		int numClasses, int numTerms, int numDocsTest, double *freqTermVector,
		double totalTermFreq, int totalTerms, int sp, double *modeloNB, int d,
		double lambda, double alpha, int c);
void super_parent_update(double *probSp, double *matrixTermFreq,
		double *totalFreqClassVector, int *docTestIndexVector,
		int *docTestVector, double *docTestFreqVector, int numClasses,
		int numTerms, int numDocsTest, double *freqTermVector,
		double totalTermFreq, int totalTerms, int filho, double *modeloNB, int d,
		double lambda, double alpha, int c);
double nb_gpu(const char* filenameTreino, const char* filenameTeste,
		int numDocs, int numClasses, int numTerms, int numDocsTest,
		int numTermsTest, double alpha, double lambda, int cudaDevice);
void find_sp_kernel(int *docIndexVector, int *docVector, double *docFreqVector, int numClasses, 
		int numTerms, int numDocs, int totalTerms, int *hasSp, int sp);
void super_parent_train(int *docIndexVector, int *docVector,
		double *docFreqVector, int *docClassVector, double *totalTermClassSp,
		int numTerms, int numDocs, int totalTerms, int numClasses,
		double* probSp, int sp, double alpha, int *hasSp);



#ifdef __cplusplus
}
#endif

#endif
