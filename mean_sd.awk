BEGIN{sum_mac = 0; sum_mic = 0;}
{
  mac[NR] = $1;
  sum_mac += $1;
  mic[NR] = $2;
  sum_mic += $2;

}
END{
  mean_mac = sum_mac/NR;
  sd_mac = 0;
  for(i in mac){
    sd_mac += (mac[i]- mean_mac)*(mac[i]- mean_mac);
  }
  sd_mac = sd_mac / (NR - 1);
  sd_mac = sqrt(sd_mac);
  print mean_mac" "sd_mac;
  
  mean_mic = sum_mic/NR;
  sd_mic = 0;
  for(i in mic){
    sd_mic += (mic[i]- mean_mic)*(mic[i]- mean_mic);
  }
  sd_mic = sd_mic / (NR - 1);
  sd_mic = sqrt(sd_mic);
  print mean_mic" "sd_mic;

}
