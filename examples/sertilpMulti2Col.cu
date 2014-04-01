/*
author: Krzysztof Sopyla
mail: krzysztofsopyla@gmail.com
License: MIT
*/

#include <float.h>


texture<float,1,cudaReadModeElementType>  VecI_TexRef;
texture<float,1,cudaReadModeElementType>  VecJ_TexRef;

__constant__ float GAMMA=0.5f;


//define PREFETCH_SIZE 2
//__constant__ int PREFETCH_SIZE=2;
//__constant__ int THREAD_PER_ROW=2;
//__constant__ int LOG_THREADS=1; // LOG2(ThreadPerRow)
//__constant__ int SLICE_SIZE=16;

#define PREFETCH_SIZE 2
#define THREAD_PER_ROW 2

// LOG2(ThreadPerRow)
#define LOG_THREADS 1 
#define SLICE_SIZE 64 

__constant__ int STEP=(THREAD_PER_ROW*SLICE_SIZE);

//cuda kernel function for computing SVM RBF kernel in multi-class scenario with "one vs one" classification scheme, uses 
// Ellpack-R format for storing sparse matrix, uses ILP - prefetch vector elements in registers
// it is used in multiclass classifier "one vs one",
// arrays vals and colIdx should be aligned to PREFETCH_SIZE
//Params:
//vals - array of vectors values, 
//colIdx  - array of column indexes in ellpack-r format
//rowLength -array, contains number of nonzero elements in each row
//selfDot - array of precomputed self linear product 
//y - array of labels for sub-problem (array size =sub-problem size)
//results - array of results, only for sub-problem
//num_rows -number of vectors in whole dataset
//idxI - index number in sub-problem
//idxJ - index number in sub-problem
//idxI_ds - index number in whole dataset
//idxJ_ds - index number in whole dataset
//cls1_N_aligned - number of elements in class 1, aligned to warp size
//cls_start - array containing pointers to each class start
//cls_count - array containing size of each considered class (2 elements only)
//cls       - 2 element array containing considered class numbers
//gamma - gamma parameter for RBF 
extern "C" __global__ void rbfSERTILP2multi(const float * vals,
							   const int * colIdx, 
							   const int * rowLength, 
							   const int * sliceStart,
							   const float* selfDot,
							   const float* y,
							   float * results,
							   const int num_rows,
							   const int align, //SERTILP format align=threadsPerRow*sliceSize
							   const int idxI,//offset indices in a subproblem
							   const int idxJ,
							   const int idxI_ds,//true indices in a dataset
							   const int idxJ_ds,
							   const int cls1_N_aligned, //aligned to power of 2 class 1 size
							   const int * cls_start, //pointers where each class starts
							   const int * cls_count, //2-element array, number of elements in each class
							   const int * cls // 2-element array, with class number eg. cls[0]=2, cls[1]=5, binary classifiers between class 2 vs 5
							   )
{

	__shared__ float shISelfDot;
	__shared__ float shJSelfDot;
	__shared__ int shYI;
	__shared__ int shYJ;
	__shared__ int shClsSum;
	//__shared__ int shSliceStart;
	
	//shared memory for final reduction for THREAD_PER_ROW for each kernel column
	__shared__  float shDot[THREAD_PER_ROW*SLICE_SIZE*2];
	
	shDot[threadIdx.x]=0.0f;
	shDot[threadIdx.x+THREAD_PER_ROW*SLICE_SIZE]=0.0f;
	
	if(threadIdx.x==0)
	{
		shClsSum=num_rows; // cls_count[0]+cls_count[1];
		shYI = y[idxI];
		shYJ = y[idxJ];
		shISelfDot=selfDot[idxI_ds];
		shJSelfDot=selfDot[idxJ_ds];
		
		//TODO:check if is it correct?
		//shSliceStart=sliceStart[blockIdx.x];
	}
	__syncthreads();
	
	// global thread index
	//const unsigned int t   = blockDim.x * blockIdx.x + threadIdx.x;  

	const unsigned int th_group    = (blockDim.x * blockIdx.x + threadIdx.x)/THREAD_PER_ROW;  
	//thread group number, 
	//int th_group = t/THREAD_PER_ROW;
	//int th_mod = t%THREAD_PER_ROW;
	//const unsigned int th_mod    = (blockDim.x * blockIdx.x + threadIdx.x)%THREAD_PER_ROW;  
	//determines the class membership, (first cls1_N threads belongs to first class),0 - first class, 1- second class
	int th_cls = th_group/cls1_N_aligned;
	
	//thread offset in particular class
	int th_cls_offset = th_group - th_cls*(cls1_N_aligned);
	//or
	//int th_cls_offset = t - th_cls*cls1_N_aligned*THREAD_PER_ROW;
	
	//
	if(th_cls_offset<cls_count[th_cls])
	{		
		//int cls_nr = cls[th_cls];
		//true row index in a dataset
		//int row = th_cls_offset+cls_start[cls_nr];
		int row = th_cls_offset+cls_start[cls[th_cls]];

		// //slice number, which particular row belongs in
		//int sliceStartNr = row/SLICE_SIZE;
		//int rowSliceStart=sliceStart[sliceStartNr];
		//int rowSliceStart=sliceStart[row/SLICE_SIZE];
		// //offset of the row in slice
		//int sliceOffset = row% SLICE_SIZE;
		
		float preVals[PREFETCH_SIZE]={0.0};
		int preColls[PREFETCH_SIZE]={-1};
		
		//warning cuda doesn't initialize array with 0, 
		float dotI[PREFETCH_SIZE]={0};
		float dotJ[PREFETCH_SIZE]={0};

		int maxEl = rowLength[row];
		unsigned int j=0;
		//important!, explicit initialization with zeros
		for( j=0; j<PREFETCH_SIZE;j++)			
		{
			dotI[j]=0.0f;
			dotJ[j]=0.0f;	
		}
		
		unsigned int arIdx=0;
	
		for(unsigned int i=0; i<maxEl;i++)
		{
			#pragma unroll
			for( j=0; j<PREFETCH_SIZE;j++)			
			{
				arIdx = (i*PREFETCH_SIZE+j )*align+sliceStart[row/SLICE_SIZE]+( row% SLICE_SIZE)*THREAD_PER_ROW+threadIdx.x%THREAD_PER_ROW;
				preColls[j]=colIdx[arIdx];
				preVals[j]=vals[arIdx];
			}
			
			#pragma unroll
			for( j=0; j<PREFETCH_SIZE;j++){
				dotI[j]+=preVals[j]*tex1Dfetch(VecI_TexRef,preColls[j]);
				dotJ[j]+=preVals[j]*tex1Dfetch(VecJ_TexRef,preColls[j]);
			}
		}
		
		#pragma unroll
		for( j=1; j<PREFETCH_SIZE;j++){
			dotI[0]+=dotI[j];
			dotJ[0]+=dotJ[j];
		}
		
		//store i-collumn partial dot result
		shDot[threadIdx.x] = dotI[0];
		//store j-collumn partial dot result
		shDot[threadIdx.x+STEP] = dotJ[0];
		
		__syncthreads();
		
		for(j=1;j<=LOG_THREADS;j<<=1)
		{
			arIdx = 2*j*threadIdx.x;
			//if(arIdx< (cls_count[th_cls]*THREAD_PER_ROW )){
			if(arIdx< STEP){
				shDot[arIdx]+=shDot[arIdx+j];
				shDot[arIdx+STEP]+=shDot[arIdx+j+STEP];	
			}
			//todo: check if necessary
			__syncthreads();
		}
		//
		//if(th_mod==0)
		if(threadIdx.x%THREAD_PER_ROW==0)
		{
			//float dI = shDot[threadIdx.x];
			//float dJ = shDot[threadIdx.x+STEP];
			
			//index within a subset of two considered class
			//int rIdx =th_cls_offset+th_cls*cls_count[0];
			
			results[row]=y[row]*shYI*expf(-GAMMA*(selfDot[row]+shISelfDot-2*shDot[threadIdx.x]));
			results[row+shClsSum]=y[row]*shYJ*expf(-GAMMA*(selfDot[row]+shJSelfDot-2*shDot[threadIdx.x+STEP]));
			//for testing 
			// results[row]=dI;
			// results[row+shClsSum]=dJ;
			
		}
		
		
	}
}







