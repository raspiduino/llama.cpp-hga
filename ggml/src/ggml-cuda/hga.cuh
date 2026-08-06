#pragma once
#include "ggml.h"
#include "ggml-cuda/common.cuh"

void ggml_cuda_op_hga_summary(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_hga_route(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_hga_stitch(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_hga_mask(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_hga_gather(ggml_backend_cuda_context & ctx, ggml_tensor * dst);