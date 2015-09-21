BEGIN{FS=" ";}
{
    id+=1;
    data=id";1;CLASS="$1;
    for(i=2;i<=NF;i++){
        split($i,vet,":");
        data=data";"vet[1]";"vet[2];
    }  
    print data;
}
