dataset=$1;
alpha=$2;
lambda=$3;

numAttributes=`awk -f attributes.awk $dataset`;
numClasses=`awk -F";" '{vet[$3]++;}END{cont=0;for(i in vet) cont++; print cont;}' $dataset`;

./hold-out -d $dataset -p 0.3 -i 1

#./tf-idf -d treino.dat -t 0 > tmp; mv tmp treino.dat
numDocTreino=`awk -F";" 'END{print NR;}' treino.dat`;
numDocTeste=`awk -F";" 'END{print NR;}' teste.dat`;
echo "$numDocTreino - $numDocTeste";

time  ./nb -c 0 -nd $numDocTreino -nc $numClasses -nt $numAttributes -fl treino.dat -ndT $numDocTeste -ntT $numAttributes -ft teste.dat -a $alpha -l $lambda > "res_nb.dat"


