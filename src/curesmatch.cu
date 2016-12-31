extern "C" {
   #include "lua.h"
   #include "lualib.h"
   #include "lauxlib.h"
}

#include "luaT.h"
#include "THC.h"

#include <stdio.h>
#include <assert.h>
#include <math_constants.h>
#include <stdint.h>
#include <unistd.h>
#include <png++/image.hpp>

#define TB 128

THCState* getCutorchState(lua_State* L)
{
   lua_getglobal(L, "cutorch");
   lua_getfield(L, -1, "getState");
   lua_call(L, 0, 1);
   THCState *state = (THCState*) lua_touserdata(L, -1);
   lua_pop(L, 2);
   return state;
}

void checkCudaError(lua_State *L) {
   cudaError_t status = cudaPeekAtLastError();
   if (status != cudaSuccess) {
      luaL_error(L, cudaGetErrorString(status));
   }
}

__global__ void outlier_detection(float *d0, float *d1, float *outlier, int size, int dim3, float *conf1, float *conf2, int disp_max, float t1, float t2)
{
   int id = blockIdx.x * blockDim.x + threadIdx.x;
   if (id < size) {
      int x = id % dim3;
      int d0i = d0[id];
      if (x - d0i < 0) {
         //assert(0);
         outlier[id] = 1;
      } else if ((abs(d0[id] - d1[id - d0i]) < 1.1) 
            || (conf1[id] > t1
               && (conf1[id] - conf2[id- d0i] > t2)
            )){
         outlier[id] = 0; /* match */
      } else {
         outlier[id] = 1; /* occlusion */
         for (int d = 0; d < disp_max; d++) {
            if (x - d >= 0 && abs(d - d1[id - d]) < 1.1) {
               outlier[id] = 2; /* mismatch */
               break;
            }
         }
      }
   }
}

int outlier_detection(lua_State *L)
{
   THCState *state = getCutorchState(L);
   THCudaTensor *d0 = (THCudaTensor*)luaT_checkudata(L, 1, "torch.CudaTensor");
   THCudaTensor *d1 = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
   THCudaTensor *outlier = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
   int disp_max = luaL_checkinteger(L, 4);
   THCudaTensor *conf1 = (THCudaTensor*)luaT_checkudata(L, 5, "torch.CudaTensor");
   THCudaTensor *conf2 = (THCudaTensor*)luaT_checkudata(L, 6, "torch.CudaTensor");
   float t1 = luaL_checknumber(L, 7);
   float t2 = luaL_checknumber(L, 8);

   outlier_detection<<<(THCudaTensor_nElement(state, d0) - 1) / TB + 1, TB>>>(
      THCudaTensor_data(state, d0),
      THCudaTensor_data(state, d1),
      THCudaTensor_data(state, outlier),
      THCudaTensor_nElement(state, d0),
      THCudaTensor_size(state, d0, 3),
      THCudaTensor_data(state, conf1),
      THCudaTensor_data(state, conf2),
      disp_max, t1, t2);
   checkCudaError(L);
   return 0;
}

__global__ void L2dist_(float *input_L, float *input_R, float *output_L, float *output_R, int size1_input, int size1, int size3, int size23)
{
   int id = blockIdx.x * blockDim.x + threadIdx.x;
   if (id < size23) {
      int dim3 = id % size3;
      assert(size1_input <= 512);
      float L_cache[512];
      for (int i = 0; i < size1_input; i++) {
         L_cache[i] = input_L[i * size23 + id];
      }

      for (int d = 0; d < size1; d++) {
         if (dim3 - d >= 0) {
            float sum = 0;
            float diff = 0;
            for (int i = 0; i < size1_input; i++) {
               diff = L_cache[i] - input_R[i * size23 + id - d];
               sum += diff*diff;
            }
            sum = sqrt(sum);
            output_L[d * size23 + id] = sum;
            output_R[d * size23 + id - d] = sum;
         }
      }
   }
}

int L2dist(lua_State *L)
{
   THCState *state = getCutorchState(L);
   THCudaTensor *input_L = (THCudaTensor*)luaT_checkudata(L, 1, "torch.CudaTensor");
   THCudaTensor *input_R = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
   THCudaTensor *output_L = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
   THCudaTensor *output_R = (THCudaTensor*)luaT_checkudata(L, 4, "torch.CudaTensor");
   int size23 = THCudaTensor_size(state, output_L, 2) * THCudaTensor_size(state, output_L, 3);
   L2dist_<<<(size23 - 1) / TB + 1, TB>>>(
      THCudaTensor_data(state, input_L),
      THCudaTensor_data(state, input_R),
      THCudaTensor_data(state, output_L),
      THCudaTensor_data(state, output_R),
      THCudaTensor_size(state, input_L, 1),
      THCudaTensor_size(state, output_L, 1),
      THCudaTensor_size(state, output_L, 3),
      size23);
   checkCudaError(L);
   return 0;
}

static const struct luaL_Reg funcs[] = {
   {"outlier_detection", outlier_detection},
   {"L2dist", L2dist},
   {NULL, NULL}
};

extern "C" int luaopen_libcuresmatch(lua_State *L) {
	srand(42);
	luaL_openlib(L, "curesmatch", funcs, 0);
   return 1;
}
