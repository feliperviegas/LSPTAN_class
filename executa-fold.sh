mb(){
i=$1;
./nb -d $k -nd $numDocTreino -nc $numClasses -nt $numAttributes -fl "treino${i}.dat" -ndT $numDocTeste -ntT $numAttributes -ft "teste${i}.dat" -a $alpha -l $lambda > "res_nb${i}.dat" 2> "log${i}.dat";
}
dataset=$1;
alpha=$2;
lambda=$3;
freq=$4;
numAttributes=`awk -f attributes.awk $dataset`;
numClasses=`awk -F";" '{vet[$3]++;}END{cont=0;for(i in vet) cont++; print cont;}' $dataset`;
i=0;
for j in `seq 1 1 1`
do
	for k in `seq 0 1 3`
	do
		echo $i;
		awk -F";" -v freq=$freq '{for(i=4;i<=NF;i=i+2) dist[$i]+=1;}END{for(i in dist) if(dist[i] >= freq) print i" "dist[i];}' treino-fold${i}.dat > filtro-atributos${i}.dat

		./filtra -d "treino-fold${i}.dat" -i "filtro-atributos${i}.dat" > "filtro-treino${i}.dat"
		./filtra -d "validacao-fold${i}.dat" -i "filtro-atributos${i}.dat" > "filtro-validacao${i}.dat"

		awk -F";" '{if(NF > 3) print $0;}' "filtro-treino${i}.dat" > "treino${i}.dat"
		awk -F";" '{if(NF > 3) print $0;}' "filtro-validacao${i}.dat" > "teste${i}.dat"
  
		numDocTreino=`awk -F";" 'END{print NR;}' "treino${i}.dat"`;
		numDocTeste=`awk -F";" 'END{print NR;}' "teste${i}.dat"`;
		echo "$numDocTreino - $numDocTeste";
		mb $i &
		i=`expr $i + 1`;
	done
	wait;
done

