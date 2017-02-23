
#include <stdlib.h>
#include <stdio.h> 
#include <sys/time.h>
void cpu(int num_rows,int *ptr, int *indices,float *data,float *x,float *y)
{ int i,jj;
 for( i=0;i<1000;i++)
 {
	if(i< num_rows)
	{
		float dot=0;
		
		int row_start = ptr[i];
		int row_end = ptr[i+1];
		for ( jj = row_start; jj<row_end; jj++)
			{
				dot += data[jj] * x[indices[jj]];
			}
		y[i] = dot;
		
	}
 }
}
int main()
{   struct timeval begin, end;
	int num_rows=1000;
	int ptr[1001];
	int indices[25000];
	float data[25000];
	int i,j,m,n,p;
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
	
	for(p=0;p<25000;p++)
	{
		v[p]=p;
	}
 
	float *z;
	z= (float*)malloc(1000*sizeof(float));
	memset(z,0,1000*sizeof(float));

	gettimeofday(&begin, NULL);
	cpu(num_rows,ptr,indices,data,v,z);
	gettimeofday(&end, NULL);
	fprintf(stdout, "time run on cpu(ms)  = %lf\n", ((end.tv_sec-begin.tv_sec)*1000000.0 + (end.tv_usec-begin.tv_usec) * 1.0)/1000.0);

   
   

}
