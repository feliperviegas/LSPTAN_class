dataset=$1;
freq=$2;
alpha=$3;
lambda=$4;
numAttributes=`awk -f attributes.awk $dataset`;
numClasses=`awk -F";" '{vet[$3]++;}END{cont=0;for(i in vet) cont++; print cont;}' $dataset`;

./hold-out -d $dataset -p 0.3 -i 1

awk -F";" -v freq=$freq '{for(i=4;i<=NF;i=i+2) dist[$i]+=1;}END{for(i in dist) if(dist[i] >= freq) print i" "dist[i];}' treino.dat | sort -n -k1 > filtro-atributos.dat

./filtra -d treino.dat -i filtro-atributos.dat > filtro-treino.dat
./filtra -d teste.dat -i filtro-atributos.dat > filtro-teste.dat

awk -F";" '{if(NF > 3) print $0;}' filtro-treino.dat > treino.dat
awk -F";" '{if(NF > 3) print $0;}' filtro-teste.dat > teste.dat

numDocTreino=`awk -F";" 'END{print NR;}' treino.dat`;
numDocTeste=`awk -F";" 'END{print NR;}' teste.dat`;
echo "$numDocTreino - $numDocTeste";

time  ./nb -c 1 -nd $numDocTreino -nc $numClasses -nt $numAttributes -fl treino.dat -ndT $numDocTeste -ntT $numAttributes -ft teste.dat -a $alpha -l $lambda> "res_nb.dat"

