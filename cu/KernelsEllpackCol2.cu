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
	__shared__ int shRows;
	__shared__ int shClsSum;
	
	if(threadIdx.x==0)
	{
		shRows = num_rows;
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
		float dotI[PREFETCH_SIZE]={0,0};
		float dotJ[PREFETCH_SIZE]={0,0};

		int maxEl = rowLength[rowIdx];
		unsigned int j=0;
		
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
				// dotI[j]+=1;//preVals[j];
				// dotJ[j]+=1;//preVals[j];
			}
			
		}
		#pragma unroll
		for( j=1; j<PREFETCH_SIZE;j++){
			dotI[0]+=dotI[j];
			dotJ[0]+=dotJ[j];
		}
		
		
		int rIdx = th_cls_offset+th_cls*cls_count[0];
		//rbf
		// results[rIdx]=dotI[0];
		// results[rIdx+cls_sum]=dotJ[0];
		
		// results[rIdx]=y[rIdx];
		// results[rIdx+cls_sum]=y[rIdx];
		
		//rbf
		//results[rIdx]=selfDot[rowIdx]+shISelfDot;
		//results[rIdx+shClsSum]=selfDot[rowIdx]+shJSelfDot;
		// results[rIdx]=(dotI[0]);
		// results[rIdx+shClsSum]=(dotJ[0]);
		results[rIdx]=y[rIdx]*shYI*expf(-GAMMA*(selfDot[rowIdx]+shISelfDot-2*dotI[0]));
		results[rIdx+shClsSum]=y[rIdx]*shYJ*expf(-GAMMA*(selfDot[rowIdx]+shJSelfDot-2*dotJ[0]));
	}
	

}

extern "C" __global__ void rbfEllpackILPcol2multi2(const float * vals,
									   const int * colIdx, 
									   const int * rowLength, 
									   const float* selfDot,
									   const float* y,
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
	

	// __shared__ int shidxi;
	// __shared__ int shidxj;
	__shared__ float shISelfDot;
	__shared__ float shJSelfDot;
	__shared__ float shYI;
	__shared__ float shYJ;
	__shared__ int shRows;
	__shared__ int shClsSum;
	
	if(threadIdx.x==0)
	{
		shRows = num_rows;
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
	
	//thread in class offset
	int th_cls_offset = t-th_cls*cls1_N_aligned;
	
	if(th_cls_offset<cls_count[th_cls])
	{
		
		//true row index
		int cls_nr = cls[th_cls];
		int rowIdx = th_cls_offset+cls_start[cls_nr];
			
		float preVals[PREFETCH_SIZE]={0};
		int preColls[PREFETCH_SIZE]={-1};
		float dotI[PREFETCH_SIZE]={0};
		float dotJ[PREFETCH_SIZE]={0};

		int maxEl = rowLength[rowIdx];
	
		unsigned int j=0;
		//unsigned int arIdx=0;
		for(unsigned int i=0; i<maxEl;i++)
		{
			#pragma unroll 2
			for( j=0; j<PREFETCH_SIZE;j++)			
			{
				preColls[j]=colIdx[ (i*PREFETCH_SIZE+j)*shRows+rowIdx];
				preVals[j]=vals[ (i*PREFETCH_SIZE+j)*shRows+rowIdx];
			}
			
			#pragma unroll
			for( j=0; j<PREFETCH_SIZE;j++){
							
				dotI[j]+=preVals[j]*tex1Dfetch(VecI_TexRef,preColls[j]);
				dotJ[j]+=preVals[j]*tex1Dfetch(VecJ_TexRef,preColls[j]);
				
				//dotI[j]+=preVals[j];
				//dotJ[j]+=preVals[j];

			}
			
		}
				
		#pragma unroll
		for( j=1; j<PREFETCH_SIZE;j++){
			dotI[0]+=dotI[j];
			dotJ[0]+=dotJ[j];
		}
		
		//result and y index
		int rIdx = th_cls_offset+th_cls*cls_count[0];
		//rbf
		results[rIdx]=y[rIdx]*shYI*expf(-GAMMA*(selfDot[rowIdx]+shISelfDot-2*dotI[0]));
		results[rIdx+shClsSum]=y[rIdx]*shYJ*expf(-GAMMA*(selfDot[rowIdx]+shJSelfDot-2*dotJ[0]));
		
	}//end if	

}

//cuda kernel funtion for computing SVM RBF kernel, uses 
// Ellpack-R format for storing sparse matrix, labels are in texture cache,  uses ILP - prefetch vector elements in registers
// arrays vals and colIdx should be aligned to PREFETCH_SIZE
//Params:
//vals - array of vectors values
//colIdx  - array of column indexes in ellpack-r format
//rowLength -array, contains number of nonzero elements in each row
//selfDot - array of precomputed self linear product 
//results - array of results Linear Kernel
//num_rows -number of vectors
//mainVecIndex - main vector index, needed for retrieving its label
//gamma - gamma parameter for RBF 
extern "C" __global__ void rbfEllpackILPcol2(const float * vals,
									   const int * colIdx, 
									   const int * rowLength, 
									   const float* selfDot,
									   const float* y,
									   float * results,
									   const int num_rows,
									   const int indexI,
									   const int indexJ,
									   const float gamma)
{
	

	__shared__ float shGamma;
	__shared__ int shIdxI;
	__shared__ int shIdxJ;
	__shared__ float shISelfDot;
	__shared__ float shJSelfDot;
	__shared__ float shYI;
	__shared__ float shYJ;
	__shared__ int shRows;
	
	if(threadIdx.x==0)
	{
		shRows = num_rows;
		shIdxI=indexI;
		shIdxJ = indexJ;
		shGamma = gamma;
		shISelfDot = selfDot[shIdxI];
		shJSelfDot = selfDot[shIdxJ];
		shYI = y[shIdxI];
		shYJ = y[shIdxJ];
	}
	__syncthreads();
		
	const unsigned int row   = blockDim.x * blockIdx.x + threadIdx.x;  // global thread index

	if(row<shRows)
	{
		float preVals[PREFETCH_SIZE];
		int preColls[PREFETCH_SIZE];
		
		
		//float dot[2][PREFETCH_SIZE]={0};

		float dotI[PREFETCH_SIZE]={0,0};
		
		float dotJ[PREFETCH_SIZE]={0,0};

		int maxEl = rowLength[row];
	
		unsigned int j=0;
		//unsigned int arIdx=0;
		for(unsigned int i=0; i<maxEl;i++)
		{
			#pragma unroll 2
			for( j=0; j<PREFETCH_SIZE;j++)			
			{
				preColls[j]=colIdx[ (i*PREFETCH_SIZE+j)*shRows+row];
				preVals[j]=vals[ (i*PREFETCH_SIZE+j)*shRows+row];
				
				//arIdx = (i*PREFETCH_SIZE+j)*shRows+row;
				//preColls[j]=colIdx[arIdx];
				//preVals[j]=vals[arIdx];
			}
			
			#pragma unroll
			for( j=0; j<PREFETCH_SIZE;j++){
							
				dotI[j]+=preVals[j]*tex1Dfetch(VecI_TexRef,preColls[j]);
				dotJ[j]+=preVals[j]*tex1Dfetch(VecJ_TexRef,preColls[j]);
				
				//dotI[j]+=preVals[j];
				//dotJ[j]+=preVals[j];

			}
			
		}
				
		#pragma unroll
		for( j=1; j<PREFETCH_SIZE;j++){
			//dot[0][0]+=dot[0][j];
			//dot[1][0]+=dot[1][j];
			dotI[0]+=dotI[j];
			dotJ[0]+=dotJ[j];
		}
		
		//rbf
		results[row]=y[row]*shYI*expf(-shGamma*(selfDot[row]+shISelfDot-2*dotI[0]));
		results[row+shRows]=y[row]*shYJ*expf(-shGamma*(selfDot[row]+shJSelfDot-2*dotJ[0]));
		
		//results[row]=dotI[0];
		//results[row+shRows]=dotJ[0];
		//float yRow = y[row];
		//float selfDotRow = selfDot[row];
		//results[row]= yRow*shYI*expf(-shGamma*(selfDotRow+shISelfDot-2*dotI[0]));
		//results[row+shRows]=yRow*shYJ*expf(-shGamma*(selfDotRow+shJSelfDot-2*dotJ[0]));
		
	}	

}


extern "C" __global__ void rbfEllpackILPcol2_Prefetch2(const float * vals,
									   const int * colIdx, 
									   const int * rowLength, 
									   const float* selfDot,
									   const float* y,
									   float * results,
									   const int num_rows,
									   const int indexI,
									   const int indexJ,
									   const float gamma)
{
	

	__shared__ float shGamma;
	__shared__ int shIdxI;
	__shared__ int shIdxJ;
	__shared__ float shISelfDot;
	__shared__ float shJSelfDot;
	__shared__ float shYI;
	__shared__ float shYJ;
	__shared__ int shRows;
	
	if(threadIdx.x==0)
	{
		shRows = num_rows;
		shIdxI=indexI;
		shIdxJ = indexJ;
		shGamma = gamma;
		shISelfDot = selfDot[shIdxI];
		shJSelfDot = selfDot[shIdxJ];
		shYI = y[shIdxI];
		shYJ = y[shIdxJ];
	}
	__syncthreads();
		
	const unsigned int row   = blockDim.x * blockIdx.x + threadIdx.x;  // global thread index

	if(row<shRows)
	{
		float preVals[2];
		int preColls[2];
		float dotI[2]={0,0};	
		float dotJ[2]={0,0};

		int maxEl = rowLength[row];
		unsigned int arIdx = row;	
		for(unsigned int i=0; i<maxEl;i++)
		{
				//preColls[0]=colIdx[ i*2*shRows+row];
				//preVals[0]=vals[i*2*shRows+row];
				//preColls[1]=colIdx[ (i*2+1)*shRows+row];
				//preVals[1]=vals[(i*2+1)*shRows+row];

				//arIdx = i*2*shRows+row;
				preColls[0]=colIdx[arIdx];
				preVals[0]=vals[arIdx];
				arIdx+=shRows;
				preColls[1]=colIdx[ arIdx];
				preVals[1]=vals[arIdx];
				arIdx+=shRows;

				dotI[0]+=preVals[0]*tex1Dfetch(VecI_TexRef,preColls[0]);
				dotI[1]+=preVals[1]*tex1Dfetch(VecI_TexRef,preColls[1]);
				dotJ[0]+=preVals[0]*tex1Dfetch(VecJ_TexRef,preColls[0]);
				dotJ[1]+=preVals[1]*tex1Dfetch(VecJ_TexRef,preColls[1]);

		}
					
		dotI[0]+=dotI[1];
		dotJ[0]+=dotJ[1];
		
		//results[row]=y[row]*shYI*expf(-shGamma*(selfDot[row]+shISelfDot-2*dotI[0]));
		//results[row+shRows]=y[row]*shYJ*expf(-shGamma*(selfDot[row]+shJSelfDot-2*dotJ[0]));
		float yRow = y[row];
		float selfDotRow = selfDot[row];
		results[row]= yRow*shYI*expf(-shGamma*(selfDotRow+shISelfDot-2*dotI[0]));
		results[row+shRows]=yRow*shYJ*expf(-shGamma*(selfDotRow+shJSelfDot-2*dotJ[0]));
		
	}	

}


