dataset=$1;
alpha=$2;
lambda=$3;

libLinear="../liblinear-1.92";
svm_scale="$libLinear/svm-scale";

numAttributes=`awk -f attributes.awk $dataset`;
numClasses=`awk -F";" '{vet[$3]++;}END{cont=0;for(i in vet) cont++; print cont;}' $dataset`;
echo ${numAttributes} ${numClasses}

./hold-out -d $dataset -p 0.3 -i 1

awk -f tsalles2libSVM.awk treino.dat > treino-svm.dat
awk -f tsalles2libSVM.awk teste.dat > teste-svm.dat

$svm_scale -l 0 -u 1 -s scale${i} treino-svm.dat > treino.dat;
$svm_scale           -r scale${i} teste-svm.dat  > teste.dat ;

awk -f libSVM2tsalles.awk treino.dat > tmp; mv tmp treino.dat
awk -f libSVM2tsalles.awk teste.dat  > tmp; mv tmp teste.dat

numDocTreino=`awk -F";" 'END{print NR;}' treino.dat`;
numDocTeste=`awk -F";" 'END{print NR;}' teste.dat`;
echo "$numDocTreino - $numDocTeste";
time  ./nb -c 0 -nd $numDocTreino -nc $numClasses -nt $numAttributes -fl treino.dat -ndT $numDocTeste -ntT $numAttributes -ft teste.dat -a $alpha -l $lambda > "res_nb.dat" 2> log.dat


