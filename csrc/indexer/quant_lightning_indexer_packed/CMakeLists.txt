sgl_register_operator(
    ID quant_lightning_indexer_packed
    EXECUTION_UNIT AICORE
    CANN_OP_TYPE SglQuantLightningIndexerPacked
    TORCH_ADAPTER torch_adapter.cpp
    SUPPORTED_ARCHS 3510
    HOST_SOURCES op_host/quant_lightning_indexer_packed.cpp
    KERNEL_SOURCES op_kernel/quant_lightning_indexer.cpp
    KERNEL_COMPILE_OPTIONS
        --cce-auto-sync=off
        -Wno-deprecated-declarations
        -mllvm -cce-vf-remove-membar=false
        -mllvm -cce-aicore-hoist-movemask=false
)
