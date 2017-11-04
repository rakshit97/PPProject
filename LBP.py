import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import cv2
import numpy

mod = SourceModule("""

#include <stdint.h>

__global__
void lbpval_data(uint8_t *gray, uint8_t *lbpvals, int numRows, int numCols)
{
    int row=threadIdx.y + blockIdx.y * blockDim.y;
	int col=threadIdx.x + blockIdx.x * blockDim.x;

    int bin[9];

    if (col>=numCols-1 || col<1 || row>=numRows-1 || row<1)
      return;

    for(int i=-1; i<=1; i++)
        for(int j=-1; j<=1; j++)
        {
            if(gray[(row+i)*numCols + (col+j)]<gray[row*numCols+col])
                bin[(i+1)*3 + (j+1)]=0;
            else
                bin[(i+1)*3 + (j+1)]=1;
        }

	lbpvals[row*numCols+col] = (bin[0]*128) + (bin[1]*64) + (bin[2]*32) + (bin[3]) + (bin[5]*16) + (bin[6]*2) + (bin[7]*4) + (bin[8]*8);
}
""")

kernel = mod.get_function("lbpval_data")

img = cv2.imread("image.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

rows, cols = img.shape
d_img = cuda.mem_alloc_like(img)
d_res = cuda.mem_alloc_like(img)
bins = numpy.zeros((3, 3), numpy.uint8)
d_bins = cuda.mem_alloc_like(bins)

numRows = numpy.int32(rows)
numCols = numpy.int32(cols)

cuda.memcpy_htod(d_img, img)
kernel(d_img, d_res, numRows, numCols, grid=(int(numpy.ceil(cols / 32.0)), int(numpy.ceil(rows / 32.0))), block=(32, 32, 1))

h_res = numpy.zeros((rows, cols), numpy.uint8)
cuda.memcpy_dtoh(h_res, d_res)

cv2.imshow('Final', h_res)
cv2.imwrite('LBP_RES.png', h_res)
cv2.waitKey(0)

