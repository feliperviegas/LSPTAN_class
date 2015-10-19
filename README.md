# LSPTAN_class

Implementation of LSPTAN_restricted using C/CUDA.


Data format
docId;Class;Year;termId;freq;termId;freq

If the collection does not have a year insert 1 instead.
The Class must start with 0.
The termId must star with 1.

To compile you must have CUDA installed.
Compile using make. 

execute the script:

bash execute_lsptan.sh <train> <test> <alpha parameter> <lambda parameter> <GPU device>

output file: res_lsptan.dat


If you have any questions, please contact me: feliperviegas@gmail.com


