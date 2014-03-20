#include <float.h>
#define PREFETCH_SIZE 2

texture<float,1,cudaReadModeElementType>  VecI_TexRef;
texture<float,1,cudaReadModeElementType>  VecJ_TexRef;

__constant__ float GAMMA=0.5f;



//cuda kernel function for computing SVM RBF kernel in multi-class scenario with "one vs one" classification scheme, uses 
// Ellpack-R format for storing sparse matrix, uses ILP - prefetch vector elements in registers
// it is used in multiclass classifier "one vs one",
// arrays vals and colIdx should be aligned to PREFETCH_SIZE
//Params:
//vals - array of vectors values, 
//colIdx  - array of column indexes in ellpack-r format
//rowLength -array, contains number of nonzero elements in each row
//selfDot - array of precomputed self linear product 
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
extern "C" __global__ void rbfEllpackILPcol2multi(const float * vals,
									   const int * colIdx, 
									   const int * rowLength, 
									   const float* selfDot,
									   const int* y,
									   float * results,
									   const int num_rows,
									   const int idxI,//indices in subproblem
									   const int idxJ,
									   const int idxI_ds,//true indices
									   const int idxJ_ds,
									   const int cls1_N_aligned, //aligned class 1 size
									   const int * cls_start, //pointers where each class starts
									   const int * cls_count, //2-element array, number of elements in each class
									   const int * cls // 2-element array, with class number eg. cls[0]=2,cls[1]=5, binary classifiers between class 2 vs 5
									   )
{

	__shared__ float shISelfDot;
	__shared__ float shJSelfDot;
	__shared__ int shYI;
	__shared__ int shYJ;
	__shared__ int shClsSum;
	
	if(threadIdx.x==0)
	{
		shClsSum= cls_count[0]+cls_count[1];
		shYI = y[idxI];
		shYJ = y[idxJ];
		shISelfDot=selfDot[idxI_ds];
		shJSelfDot=selfDot[idxJ_ds];
	}
	__syncthreads();
	const unsigned int t   = blockDim.x * blockIdx.x + threadIdx.x;  // global thread index
		
	//thread class map, 0 - first class, 1- second class
	int th_cls = (t/cls1_N_aligned) ;//>0 ? 1:0;
	int th_cls_offset = t-th_cls*cls1_N_aligned;
	
	//int cls_sum=cls_count[0]+cls_count[1]; //=shClsSum
	
	
	if(th_cls_offset<cls_count[th_cls])
	{
		//true row index
		int cls_nr = cls[th_cls];
		int rowIdx = th_cls_offset+cls_start[cls_nr];
		
		float preVals[PREFETCH_SIZE]={0.0};
		int preColls[PREFETCH_SIZE]={-1};
		//warning cuda doesn't initialize whole array with 0, 
		float dotI[PREFETCH_SIZE]={0};
		float dotJ[PREFETCH_SIZE]={0};

		int maxEl = rowLength[rowIdx];
		unsigned int j=0;
		//important!, explicit initialization with zeros
		for( j=0; j<PREFETCH_SIZE;j++)			
		{
			dotI[j]=0.0f;
			dotJ[j]=0.0f;	
		}
		
		for(unsigned int i=0; i<maxEl;i++)
		{
			#pragma unroll
			for( j=0; j<PREFETCH_SIZE;j++)			
			{
				preColls[j]=colIdx[ (i*PREFETCH_SIZE+j)*num_rows+rowIdx];
				preVals[j]=vals[ (i*PREFETCH_SIZE+j)*num_rows+rowIdx];
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
		
		
		int rIdx = th_cls_offset+th_cls*cls_count[0];
		//rbf
		results[rIdx]=y[rIdx]*shYI*expf(-GAMMA*(selfDot[rowIdx]+shISelfDot-2*dotI[0]));
		results[rIdx+shClsSum]=y[rIdx]*shYJ*expf(-GAMMA*(selfDot[rowIdx]+shJSelfDot-2*dotJ[0]));
	}
}






