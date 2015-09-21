mb(){
i=$1;
./nb -d $k -nd $numDocTreino -nc $numClasses -nt $numAttributes -fl "treino${i}.dat" -ndV $numDocValidacao -ntV $numAttributes -ft "validacao${i}.dat" -a $alpha -l $lambda > "res_nb${i}.dat" 2> "log${i}.dat";
}
dataset=$1;
alpha=$2;
lambda=$3;
i=$4;
k=$5;

libLinear="../liblinear-1.92";
svm_scale="$libLinear/svm-scale";

numAttributes=`awk -f attributes.awk $dataset`;
numClasses=`awk -F";" '{vet[$3]++;}END{cont=0;for(i in vet) cont++; print cont;}' $dataset`;
echo $i;

awk -f tsalles2libSVM.awk treino-fold${i}.dat > treino-svm${i}.dat
awk -f tsalles2libSVM.awk validacao-fold${i}.dat > validacao-svm${i}.dat

$svm_scale -l 0 -u 1 -s scale${i} treino-svm${i}.dat > tmp${i}; mv tmp${i} treino${i}.dat;
$svm_scale           -r scale${i} validacao-svm${i}.dat  > tmp${i}; mv tmp${i} validacao${i}.dat ;

awk -f libSVM2tsalles.awk treino${i}.dat > tmp${i}; mv tmp${i} treino${i}.dat
awk -f libSVM2tsalles.awk validacao${i}.dat  > tmp${i}; mv tmp${i} validacao${i}.dat

awk -F";" '{if(NF > 3) print $0;}' treino${i}.dat    > tmp${i}; mv tmp${i} treino${i}.dat
awk -F";" '{if(NF > 3) print $0;}' validacao${i}.dat > tmp${i}; mv tmp${i} validacao${i}.dat
  
numDocTreino=`awk -F";" 'END{print NR;}' treino${i}.dat`;
numDocValidacao=`awk -F";" 'END{print NR;}' validacao${i}.dat`;
echo "$numDocTreino - $numDocValidacao";
mb $i
