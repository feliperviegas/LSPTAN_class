BEGIN{FS=";";}
{
	class = $3;
	sub(/CLASS=/, "", class);
	data = class;
	for(i = 4; i <= NF; i+=2){
		term = $i;
		freq = $(i+1);
		data = data" "term":"freq;
	}
	print data;
}
