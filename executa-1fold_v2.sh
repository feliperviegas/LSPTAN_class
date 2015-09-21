mb(){
i=$1;
./nb -d $k -nd $numDocTreino -nc $numClasses -nt $numAttributes -fl "treino${i}.dat" -ndV $numDocValidacao -ntV $numAttributes -ft "validacao${i}.dat" -a $alpha -l $lambda > "res_nb${i}.dat" 2> "log${i}.dat";
}
dataset=$1;
alpha=$2;
lambda=$3;
alphaNorm=$4;
i=$5;
k=$6;
numAttributes=`awk -f attributes.awk $dataset`;
numClasses=`awk -F";" '{vet[$3]++;}END{cont=0;for(i in vet) cont++; print cont;}' $dataset`;
echo $i;

#awk -F";" -v freq=$freq '{for(i=4;i<=NF;i=i+2) dist[$i]+=1;}END{for(i in dist) if(dist[i] >= freq) print i" "dist[i];}' treino-fold${i}.dat > filtro-atributos${i}.dat

#./filtra -d "treino-fold${i}.dat" -i "filtro-atributos${i}.dat" > "filtro-treino.dat"
#./filtra -d "validacao-fold${i}.dat" -i "filtro-atributos${i}.dat" > "filtro-validacao.dat"

#./tf-idf -d treino-fold${i}.dat -t 0 > treino-norm${i}.dat
#cat treino-norm${i}.dat > treino-length${i}.dat

#cat treino-fold${i}.dat > treino-length${i}.dat

cat treino-fold${i}.dat > treino-norm${i}.dat
./lengthNorm -d treino-norm${i}.dat -t 0 -a $alphaNorm > treino-length${i}.dat
awk -F";" '{if(NF > 3) print $0;}' treino-length${i}.dat > "treino${i}.dat"
awk -F";" '{if(NF > 3) print $0;}' validacao-fold${i}.dat > "validacao${i}.dat"

numDocTreino=`awk -F";" 'END{print NR;}' "treino${i}.dat"`;
numDocValidacao=`awk -F";" 'END{print NR;}' "validacao${i}.dat"`;
echo "$numDocTreino - $numDocValidacao";
mb $i
