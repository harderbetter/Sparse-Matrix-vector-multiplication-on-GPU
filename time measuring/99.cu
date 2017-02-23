
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
	int num_rows=1000;
	int ptr[1001];
	int indices[25000];
	float data[25000];
	int i,j,m,n,q;
	for( i=0; i<1001; i++)
	{
		ptr[i]=25*i;
	}

	
	for( j=0; j<25000;j++)
	{
		indices[j]=j;
	}
	
	for(m=0;m<1000;m++)
	{
		for(n=0;n<25;n++)
		{
			data[25*m+n]=n;
		}
	}
	
 
	float v[25000];
	
	for(q=0;q<25000;q++)
	{
		v[q]=q;
	}
 
	
	
	float *outcome;
	outcome= (float*)malloc(1000*sizeof(float));
	memset(outcome,0,1000*sizeof(float));
		int *p,*in;
	float *da,*x,*y;
 
	cudaMalloc(&p,1001*sizeof(int));
	cudaMemcpy(p,ptr,1001*sizeof(int),cudaMemcpyHostToDevice);
	cudaMalloc(&in,25000*sizeof(int));
	cudaMemcpy(in,indices,25000*sizeof(int),cudaMemcpyHostToDevice);
	cudaMalloc(&da,25000*sizeof(float));
	cudaMemcpy(da,data,25000*sizeof(float),cudaMemcpyHostToDevice);
	cudaMalloc(&x,25000*sizeof(float));
	cudaMemcpy(x,v,25000*sizeof(float),cudaMemcpyHostToDevice);
	cudaMalloc(&y,1000*sizeof(float));
 
 
	dim3 blocks(49);
	dim3 threads(25000);
	cudaEventRecord(start);	
	mv<<<blocks,threads>>>(num_rows,p,in,da,x,y);
	cudaEventRecord(stop);
	cudaMemcpy(outcome,y,sizeof(float)*1000,cudaMemcpyDeviceToHost);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("calculate time on gpu (ms): %f\n", milliseconds);
	
 
	free(outcome);
	cudaFree(p);
	cudaFree(in);
	cudaFree(da);
	cudaFree(x);
	cudaFree(y);
	cudaDeviceReset();
	return EXIT_SUCCESS;
	
	

}
