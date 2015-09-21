mb(){
i=$1;
./nb -d $k -nd $numDocTreino -nc $numClasses -nt $numAttributes -fl "treino${i}.dat" -ndV $numDocValidacao -ntV $numAttributes -ft "validacao${i}.dat" -a $alpha -l $lambda > "res_nb${i}.dat" 2> "log${i}.dat";
}
dataset=$1;
alpha=$2;
lambda=$3;
i=$4;
k=0;
freq=15;
numAttributes=`awk -f attributes.awk $dataset`;
numClasses=`awk -F";" '{vet[$3]++;}END{cont=0;for(i in vet) cont++; print cont;}' $dataset`;
echo $i;

awk -F";" -v freq=$freq '{for(i=4;i<=NF;i=i+2) dist[$i]+=1;}END{for(i in dist) if(dist[i] >= freq) print i" "dist[i];}' treino-fold${i}.dat > filtro-atributos${i}.dat

./filtra -d "treino-fold${i}.dat" -i "filtro-atributos${i}.dat" > "filtro-treino.dat"
./filtra -d "validacao-fold${i}.dat" -i "filtro-atributos${i}.dat" > "filtro-validacao.dat"

awk -F";" '{if(NF > 3) print $0;}' "filtro-treino.dat" > "treino${i}.dat"
awk -F";" '{if(NF > 3) print $0;}' "filtro-validacao.dat" > "validacao${i}.dat"
  
numDocTreino=`awk -F";" 'END{print NR;}' "treino${i}.dat"`;
numDocValidacao=`awk -F";" 'END{print NR;}' "validacao${i}.dat"`;
echo "$numDocTreino - $numDocValidacao";
mb $i
