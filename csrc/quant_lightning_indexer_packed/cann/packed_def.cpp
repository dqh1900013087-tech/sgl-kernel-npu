// Packed QLI declaration. Implementation derived from ops-transformer; see LICENSE.ops-transformer.
#include "register/op_def_registry.h"
namespace ops {
class SglQuantLightningIndexerPacked : public OpDef {
public:
 explicit SglQuantLightningIndexerPacked(const char* name) : OpDef(name) {
  this->Input("query").ParamType(REQUIRED).DataType({ge::DT_FLOAT8_E4M3FN}).Format({ge::FORMAT_ND}).IgnoreContiguous();
  this->Input("key").ParamType(REQUIRED).DataType({ge::DT_UINT8}).Format({ge::FORMAT_ND}).IgnoreContiguous();
  this->Input("weights").ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND}).IgnoreContiguous();
  this->Input("query_dequant_scale").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).IgnoreContiguous();
  this->Input("key_dequant_scale").ParamType(OPTIONAL).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).IgnoreContiguous();
  this->Input("actual_seq_lengths_query").ParamType(OPTIONAL).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND}).IgnoreContiguous();
  this->Input("actual_seq_lengths_key").ParamType(OPTIONAL).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND}).IgnoreContiguous();
  this->Input("block_table").ParamType(OPTIONAL).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND}).IgnoreContiguous();
  this->Output("sparse_indices").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
  this->Attr("query_quant_mode").AttrType(REQUIRED).Int(0);
  this->Attr("key_quant_mode").AttrType(REQUIRED).Int(0);
  this->Attr("layout_query").AttrType(OPTIONAL).String("TND");
  this->Attr("layout_key").AttrType(OPTIONAL).String("PA_BSND");
  this->Attr("sparse_count").AttrType(OPTIONAL).Int(2048);
  this->Attr("sparse_mode").AttrType(OPTIONAL).Int(3);
  this->Attr("pre_tokens").AttrType(OPTIONAL).Int(9223372036854775807);
  this->Attr("next_tokens").AttrType(OPTIONAL).Int(9223372036854775807);
  this->Attr("key_stride0").AttrType(OPTIONAL).Int(-1);
  this->Attr("key_dequant_scale_stride0").AttrType(OPTIONAL).Int(-1);
  OpAICoreConfig config;
  config.DynamicCompileStaticFlag(true).DynamicFormatFlag(true).DynamicRankSupportFlag(true)
      .DynamicShapeSupportFlag(true).NeedCheckSupportFlag(false).PrecisionReduceFlag(true);
  this->AICore().AddConfig("ascend950", config);
 }
};
OP_ADD(SglQuantLightningIndexerPacked);
}
