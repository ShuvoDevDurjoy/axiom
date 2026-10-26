import numpy as np
import cupy as cp

GPU_AVAILABLE = cp.cuda.is_available() and cp.cuda.runtime.getDeviceCount() > 0
DEVICE = 'cuda' if GPU_AVAILABLE else 'cpu'

#dtypes
float32 = np.float32
float64 = np.float64
float16 = np.float16
int32   = np.int32
int64   = np.int64
bool_   = np.bool_