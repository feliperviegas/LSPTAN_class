BEGIN{FS=";";}
{
  for(i = 4; i <= NF; i = i +2){
    vet[$i] = $i;
  }
}
END{
  max = 0;
  for(i in vet){
    if(max < vet[i]){
      max = vet[i];
    }
  }
  print max;
}
