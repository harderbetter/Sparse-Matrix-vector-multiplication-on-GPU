#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h> 
#include <sys/time.h>


__global__ void mv(int num_rows,int *ptr,int *indices,float *data,float *x,float *y)
{   __shared__ float vals[128];
    int thread_id = threadIdx.x + blockIdx.x*blockDim.x;   // global thread index
    int warp_id = thread_id / 32;                         // global warp index
	int lane = thread_id & (32-1);                         // thread index within the warp
	// one warp per row
	int row = warp_id;
    int jj;
	if(row< num_rows){
		
		int row_start = ptr[row];
		int row_end = ptr[row+1];
		
		//compute running sum per thread
		vals[threadIdx.x] = 0;
		for ( jj = row_start + lane; jj<row_end; jj+=32)
			vals[threadIdx.x] += data[jj] * x[indices[jj]];
		//parallel reduction in shared memory
		if(lane < 16) vals[threadIdx.x] += vals[threadIdx.x +16];
		if(lane < 8) vals[threadIdx.x] += vals[threadIdx.x +8];
		if(lane < 4) vals[threadIdx.x] += vals[threadIdx.x +4];
		if(lane < 2) vals[threadIdx.x] += vals[threadIdx.x +2];
		if(lane < 1) vals[threadIdx.x] += vals[threadIdx.x +1];
		
		// first thread writes the result
		if(lane == 0)
		y[row] = vals[threadIdx.x];
		
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
	dim3 threads(4*32);
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