#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h> 
#include <sys/time.h>


__global__ void mv(int num_rows,int *ptr,int *indices,float *data,float *x,float *y)
{   int jj;
    int row= threadIdx.x + blockIdx.x*blockDim.x;
	if(row< num_rows){
		float dot=0;
		
		int row_start = ptr[row];
		int row_end = ptr[row+1];
		for ( jj = row_start; jj<row_end; jj++)
			dot += data[jj] * x[indices[jj]];
		y[row] = dot;
		
	}
}
int main()
{   
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop); 
	int num_rows=4;
	int ptr[] = {0,2,4,7,9};
	int indices[] = {1,2,0,2,0,2,3,1,3};
	float data[] = {2,6,1,7,5,3,9,5,3};
 
	float v[]={1,2,3,4};
 
	
 
	float *outcome;
	outcome= (float*)malloc(4*sizeof(float));
	memset(outcome,0,4*sizeof(float));
 

	int *p,*in;
	float *da,*x,*y;
 
	cudaMalloc(&p,5*sizeof(int));
	cudaMemcpy(p,ptr,5*sizeof(int),cudaMemcpyHostToDevice);
	cudaMalloc(&in,9*sizeof(int));
	cudaMemcpy(in,indices,9*sizeof(int),cudaMemcpyHostToDevice);
	cudaMalloc(&da,9*sizeof(float));
	cudaMemcpy(da,data,9*sizeof(float),cudaMemcpyHostToDevice);
	cudaMalloc(&x,4*sizeof(float));
	cudaMemcpy(x,v,4*sizeof(float),cudaMemcpyHostToDevice);
	cudaMalloc(&y,4*sizeof(float));
 
 
	dim3 blocks(1);
	dim3 threads(4);
	cudaEventRecord(start);	
	mv<<<blocks,threads>>>(num_rows,p,in,da,x,y);
	cudaEventRecord(stop);
	cudaMemcpy(outcome,y,sizeof(float)*4,cudaMemcpyDeviceToHost);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("calculate time on gpu (ms): %f\n", milliseconds);
	printf("the outcome is \n");
	for(int i=0;i< num_rows; i++)
	{
		printf("%f \n",outcome[i]);
	}
 
	free(outcome);
	cudaFree(p);
	cudaFree(in);
	cudaFree(da);
	cudaFree(x);
	cudaFree(y);
	cudaDeviceReset();
	return EXIT_SUCCESS;
 
}