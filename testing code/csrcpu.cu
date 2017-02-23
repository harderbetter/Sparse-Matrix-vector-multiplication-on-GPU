#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h> 
#include <sys/time.h>
void cpu(int num_rows,int *ptr, int *indices,float *data,float *x,float *y)
{ int i,jj;
 for( i=0;i<4;i++)
 {
	if(i< num_rows)
	{
		float dot=0;
		
		int row_start = ptr[i];
		int row_end = ptr[i+1];
		for (jj = row_start; jj<row_end; jj++)
			dot += data[jj] * x[indices[jj]];
		y[i] = dot;
		
	}
 }
}
int main()
{
	int num_rows=4;
	int ptr[] = {0,2,4,7,9};
	int indices[] = {1,2,0,2,0,2,3,1,3};
	float data[] = {2,6,1,7,5,3,9,5,3};
 
	float v[]={1,2,3,4};
 
	float *z;
	z= (float*)malloc(4*sizeof(float));
	memset(z,0,4*sizeof(float));

	
	cpu(num_rows,ptr,indices,data,v,z);
	
	
	
	printf("the outcome in cpu is \n");
	for(int i=0;i< num_rows; i++)
	{
		printf("%f \n",z[i]);
	}
}
