#ifndef IO_H__
#define IO_H__

#include <cmath>
#include <cstring>
#include <iostream>
#include <map>
#include <set>
#include <vector>
#include <utility>
#include <fstream>
#include <iomanip>
#include <locale>
#include <sstream>
#include <string>
#include <stdlib.h>
#include <unistd.h>
#include <algorithm>
#include <sys/time.h>
#include <sys/resource.h>

using namespace std;

struct Feature {
	int id;
	double weight;
};

struct comp {
	bool operator()(const Feature lhs, const Feature rhs) {
		return lhs.weight > rhs.weight;
	}
};

void stringTokenize(const std::string& str, std::vector<std::string>& tokens,
		const std::string& delimiters);
void temposExecucao(double *utime, double *stime, double *total_time);
double tempoAtual();
int* readTrainDataSP(const char* filename, int *docIndexVector,
		double *totalFreqClassVector, int *freqClassVector,
		double *freqTermVector, double *totalTermFreq, int numClasses,
		int numTerms, int *totalT, double* matrixTermFreq, set<int>& vocabulary,
		double *(*docFreqVector), int *docClassVector);
int* readTestData(const char* filename, int *docTestIndexVector, int *realClass,
		double *(*docTestFreqVector));

#endif
