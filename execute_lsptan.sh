mb(){
i=$1;
./nb -d $k -nd $numDocTreino -nc $numClasses -nt $numAttributes -fl "_train.dat" -ndV $numDocValidacao -ntV $numAttributes -ft "_test.dat" -a ${alpha} -l ${lambda} > "res_lsptan.dat" 2> "log.dat";
}
trainData=$1;
testData=$2;
alpha=$3;
lambda=$4;
k=$5;
numAttributes=`awk -f attributes.awk ${trainData} ${testData}`;
numClasses=`awk -F";" '{print $3;}' ${trainData} ${testData} | sort | uniq -c | wc -l`;
echo "Number of Classes ${numClasses}";
echo "Number of Attributes ${numAttributes}";


echo "Iteration $i alpha ${alpha} lambda ${lambda}";

awk -F";" '{if(NF > 3) print $0;}' ${trainData} > "_train.dat";
awk -F";" '{if(NF > 3) print $0;}' ${testData} > tmp; mv tmp "_test.dat";

numDocTreino=`wc -l "_train.dat" | awk '{print $1;}'`;
numDocValidacao=`wc -l "_test.dat" | awk '{print $1;}'`;
echo "$numDocTreino - $numDocValidacao";
mb $i
