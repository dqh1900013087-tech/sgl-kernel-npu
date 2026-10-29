cmake_minimum_required(VERSION 3.20)
project(sgl_qli_packed LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_HOME_PATH}")
if(NOT IS_DIRECTORY "${ASCEND_CANN_PACKAGE_PATH}")
  message(FATAL_ERROR "Source CANN set_env.sh before building")
endif()
set(ASCEND_COMPUTE_UNIT ascend950)
set(vendor_name sgl_qli_packed)
set(ENABLE_BINARY_PACKAGE ON)
set(ENABLE_SOURCE_PACKAGE ON)
find_package(ASC REQUIRED CONFIG PATHS "${ASCEND_CANN_PACKAGE_PATH}" "${ASCEND_CANN_PACKAGE_PATH}/lib64/cmake" "${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake" NO_DEFAULT_PATH)
set(qli "${CMAKE_CURRENT_SOURCE_DIR}/src/attention/quant_lightning_indexer")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/src/common/include"
 "${CMAKE_CURRENT_SOURCE_DIR}/src/third_party/opbase/include"
 "${CMAKE_CURRENT_SOURCE_DIR}/src/third_party/opbase/include/op_common")
link_directories("${ASCEND_CANN_PACKAGE_PATH}/lib64")
npu_op_package(sgl_qli_packed TYPE RUN CONFIG INSTALL_PATH "${CMAKE_CURRENT_BINARY_DIR}/package-stage")
set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/package-stage")
npu_op_code_gen(SRC "${CMAKE_CURRENT_SOURCE_DIR}/packed_def.cpp" PACKAGE sgl_qli_packed OUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/autogen")
file(GLOB api "${CMAKE_CURRENT_BINARY_DIR}/autogen/aclnn*.cpp")
npu_op_library(sgl_qli_opapi ACLNN ${api})
npu_op_library(sgl_qli_tiling TILING "${qli}/op_host/quant_lightning_indexer_tiling.cpp"
 "${qli}/op_host/quant_lightning_indexer_infershape.cpp"
 "${CMAKE_CURRENT_SOURCE_DIR}/src/third_party/opbase/src/op_common/log/log.cpp")
target_link_libraries(sgl_qli_tiling PRIVATE ascendalog error_manager)
npu_op_library(sgl_qli_proto GRAPH "${CMAKE_CURRENT_SOURCE_DIR}/packed_def.cpp" "${CMAKE_CURRENT_BINARY_DIR}/autogen/op_proto.cc")
npu_op_kernel_options(sgl_qli_kernels SglQuantLightningIndexerPacked OPTIONS
 --cce-auto-sync=off -Wno-deprecated-declarations
 -mllvm -cce-vf-remove-membar=false -mllvm -cce-aicore-hoist-movemask=false)
npu_op_kernel_sources(sgl_qli_kernels OP_TYPE SglQuantLightningIndexerPacked
 KERNEL_DIR "src/attention/quant_lightning_indexer/op_kernel" KERNEL_FILE "quant_lightning_indexer.cpp")
npu_op_kernel_library(sgl_qli_kernels SRC_BASE "${CMAKE_CURRENT_SOURCE_DIR}" TILING_LIBRARY sgl_qli_tiling)
npu_op_package_add(sgl_qli_packed LIBRARY sgl_qli_opapi sgl_qli_proto sgl_qli_tiling sgl_qli_kernels)
