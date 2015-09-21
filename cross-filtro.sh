dataset=$1;
n=$2;
freq=$3
alpha=$4;
lambda=$5;
n1=`expr $n - 1`;

rm -f res_nb.dat ev_nb.dat log*.dat;

numAttributes=`awk -f attributes.awk $dataset`;
numClasses=`awk -F";" '{vet[$3]++;}END{cont=0;for(i in vet) cont++; print cont;}' $dataset`;
echo "$numAttributes - $numClasses";

./crossValidation -d $dataset -p $n -i 1;

for i in `seq 0 1 $n1`
do
  rm -f treino.dat
  echo $i;
  for j in `seq 0 1 $n1`
  do
    if [ "${i}" -eq "${j}" ]; then
      echo "$i - $j";
      cat "dados${j}.dat" > teste.dat;
    else
      cat "dados${j}.dat" >> treino.dat;
    fi
  done;

  awk -F";" -v freq=$freq '{for(i=4;i<=NF;i=i+2) dist[$i]+=1;}END{for(i in dist) if(dist[i] >= freq) print i" "dist[i];}' treino.dat > filtro-atributos.dat

  ./filtra -d treino.dat -i filtro-atributos.dat > filtro-treino.dat
  ./filtra -d teste.dat -i filtro-atributos.dat > filtro-teste.dat

  awk -F";" '{if(NF > 3) print $0;}' filtro-treino.dat > treino.dat
  awk -F";" '{if(NF > 3) print $0;}' filtro-teste.dat > teste.dat
  
  numDocTreino=`awk -F";" 'END{print NR;}' treino.dat`;
  numDocTeste=`awk -F";" 'END{print NR;}' teste.dat`;
  echo "$numDocTreino - $numDocTeste";

 ./nb -c 1 -nd $numDocTreino -nc $numClasses -nt $numAttributes -fl treino.dat -ndT $numDocTeste -ntT $numAttributes -ft teste.dat -a $alpha -l $lambda >> "res_nb.dat" 2> "log${i}.dat";
 
done;
awk -f mean_sd.awk "res_nb.dat" > "ev_nb.dat";
rm -f dados* treino* teste*;
