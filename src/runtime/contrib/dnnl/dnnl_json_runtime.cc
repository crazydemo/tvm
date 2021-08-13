/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file src/runtime/contrib/dnnl/dnnl_json_runtime.cc
 * \brief A simple JSON runtime for DNNL.
 */

#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/registry.h>

#include <cstddef>
#include <string>
#include <vector>

#include "../json/json_node.h"
#include "../json/json_runtime.h"
#include "dnnl.hpp"
#include <map>

namespace tvm {
namespace runtime {
namespace contrib {

using namespace tvm::runtime;
using namespace tvm::runtime::json;

class DNNLJSONRuntime : public JSONRuntimeBase {
  using tag = dnnl::memory::format_tag;
  using dt = dnnl::memory::data_type;

 public:
  std::map<std::string, tag> layout_dict {
    {"NCHW16c", tag::nChw16c}, 
    {"OIHW16o16i", tag::OIhw16o16i},
    {"OIHW16i16o", tag::OIhw16i16o},
    {"OIHW16o", tag::Oihw16o},
    {"OHWI16o", tag::Ohwi16o},
    {"NCHW", tag::nchw},
    {"OIHW", tag::oihw},
    {"NCHW8c", tag::nChw8c}, 
    {"OIHW8o8i", tag::OIhw8o8i},
    {"OIHW8i8o", tag::OIhw8i8o},
    {"OHWI8o", tag::Ohwi8o},
    };

  DNNLJSONRuntime(const std::string& symbol_name, const std::string& graph_json,
                  const Array<String> const_names)
      : JSONRuntimeBase(symbol_name, graph_json, const_names) {}

  const char* type_key() const { return "dnnl_json"; }

  void Init(const Array<NDArray>& consts) override {
    BuildEngine();

    ICHECK_EQ(consts.size(), const_idx_.size())
        << "The number of input constants must match the number of required.";

    // Setup constants entries for weights.
    SetupConstants(consts);
  }

  void Run() override {
    // Fill in the input buffers.
    for (size_t i = 0; i < input_nodes_.size(); ++i) {
      auto eid = EntryID(input_nodes_[i], 0);
      // std::cout<<"write"<<std::endl;
      // std::cout<<eid<<" "<<std::endl;
      // TODO(@comaniac): Support other data lengths.
      size_t offset_in_bytes = entry_out_mem_[eid].second * 4;
      size_t buffer_size = GetDataSize(*data_entry_[eid]);
      write_to_dnnl_memory(data_entry_[eid]->data, entry_out_mem_[eid].first, buffer_size,
                           offset_in_bytes);
    }

    // Invoke the engine through intepreting the stream.
    for (size_t i = 0; i < net_.size(); ++i) {
      net_.at(i).execute(stream_, net_args_.at(i));
    }
    stream_.wait();

    // Read output buffers.
    for (size_t i = 0; i < outputs_.size(); ++i) {
      auto eid = EntryID(outputs_[i]);
      // std::cout<<"read"<<std::endl;
      // std::cout<<eid<<" "<<std::endl;
      size_t offset_in_bytes = entry_out_mem_[eid].second * 4;
      size_t buffer_size = GetDataSize(*data_entry_[eid]);
      read_from_dnnl_memory(data_entry_[eid]->data, entry_out_mem_[eid].first, buffer_size,
                            offset_in_bytes);
    }
  }

 private:
  // Build up the engine based on the input graph.
  void BuildEngine() {
    engine_ = dnnl::engine(dnnl::engine::kind::cpu, 0);
    stream_ = dnnl::stream(engine_);

    // Build subgraph engine.
    for (size_t nid = 0; nid < nodes_.size(); ++nid) {
      const auto& node = nodes_[nid];
      if (node.GetOpType() == "kernel") {
        ICHECK_EQ(node.GetOpType(), "kernel");
        auto op_name = node.GetOpName();
        if ("nn.conv2d" == op_name) {
          Conv2d(nid);
        } else if ("dnnl.conv2d_bias" == op_name) {
          Conv2d(nid, false, true);
        } else if ("dnnl.conv2d_relu" == op_name) {
          Conv2d(nid, true, false);
        } else if ("dnnl.conv2d_bias_relu" == op_name) {
          Conv2d(nid, true, true);
        } else if ("nn.dense" == op_name) {
          Dense(nid);
        } else if ("dnnl.dense_bias" == op_name) {
          Dense(nid, true);
        } else if ("nn.batch_norm" == op_name) {
          BatchNorm(nid);
        } else if ("nn.relu" == op_name) {
          Relu(nid);
        } else if ("add" == op_name) {
          Add(nid);
        } else if ("concatenate" == op_name) {
          Concat(nid);
        } else if ("nn.max_pool2d" == op_name) {
          MaxPool2d(nid);
        } else if ("nn.avg_pool2d" == op_name) {
          AvgPool2d(nid);
        } else {
          LOG(FATAL) << "Unsupported op: " << op_name;
        }
      }
    }
  }

  // Bind a JSON graph node entry to a DNNL memory.
  dnnl::memory BindDNNLMemory(const JSONGraphNodeEntry& entry, dnnl::memory::desc mem_desc,
                              size_t offset = 0) {
    auto eid = EntryID(entry);
    if (entry_out_mem_.count(eid) == 0) {
      return BindDNNLMemory(entry, dnnl::memory(mem_desc, engine_), offset);
    }
    return entry_out_mem_[eid].first;
  }

  // Bind a JSON graph node entry to a given DNNL memory.
  dnnl::memory BindDNNLMemory(const JSONGraphNodeEntry& entry, dnnl::memory mem,
                              size_t offset = 0) {
    auto eid = EntryID(entry);
    // Since the DNNL memory has been created before calling this function, we assume the entry
    // has not yet been bound to the other DNNL memory; otherwise it may have memory leak.
    ICHECK_EQ(entry_out_mem_.count(eid), 0);

    // TODO(@comanic): Support other data types (i.e., int8).
    auto data_node = nodes_[entry.id_];
    auto dltype = data_node.GetOpDataType()[entry.index_];
    ICHECK_EQ(dltype.bits, 32);

    entry_out_mem_[eid] = {mem, offset};
    return entry_out_mem_[eid].first;
  }

  void Conv2d(const size_t& nid, const bool has_relu = false, const bool has_bias = false) {
    auto node = nodes_[nid];

    // Setup attributes.
    auto data_entry = node.GetInputs()[0];
    auto weight_entry = node.GetInputs()[1];
    dnnl::memory::dims input_shape = nodes_[data_entry.id_].GetOpShape()[data_entry.index_];
    dnnl::memory::dims weight_shape = nodes_[weight_entry.id_].GetOpShape()[weight_entry.index_];
    std::vector<std::string> str_strides = node.GetAttr<std::vector<std::string>>("strides");
    std::vector<std::string> str_padding = node.GetAttr<std::vector<std::string>>("padding");
    dnnl::memory::dim groups = std::stoi(node.GetAttr<std::vector<std::string>>("groups")[0]);
    // std::cout<<node.GetAttr<std::vector<std::string>>("data_layout")[0]<<std::endl;
    // std::cout<<node.GetAttr<std::vector<std::string>>("kernel_layout")[0]<<std::endl;
    auto src_df = layout_dict[node.GetAttr<std::vector<std::string>>("data_layout")[0]];
    auto weight_df = layout_dict[node.GetAttr<std::vector<std::string>>("kernel_layout")[0]];
    auto dst_df = layout_dict[node.GetAttr<std::vector<std::string>>("out_layout")[0]];

    // if(node.GetAttr<std::vector<std::string>>("kernel_layout")[0]=="OHWI8o")
    // {dst_df = layout_dict["NCHW8c"];}
    
    // for (auto in: input_shape)
    // {
    //   std::cout<<in<<' ';
    // }

    // std::cout<<std::endl;

    // std::cout<<weight_shape.size()<<' '<<std::endl;

    // for (auto in: weight_shape)
    // {
    //   std::cout<<in<<' ';
    // }
    // std::cout<<std::endl;
    
    dnnl::memory::dim N = input_shape[0],       // batch size
        IC = input_shape[1],                    // input channels
        IH = input_shape[2],                    // input height
        IW = input_shape[2],                    // input width
        OC = weight_shape[0],                   // output channels
        KH = weight_shape[2],                   // weight height
        KW = weight_shape[3],                   // weight width
        PH_L = std::stoi(str_padding[1]),       // height padding: left
        PH_R = std::stoi(str_padding[3]),       // height padding: right
        PW_L = std::stoi(str_padding[0]),       // width padding: left
        PW_R = std::stoi(str_padding[2]),       // width padding: right
        SH = std::stoi(str_strides[0]),         // height-wise stride
        SW = std::stoi(str_strides[0]),         // weight-wise stride
        OH = (IH - KH + PH_L + PH_R) / SH + 1,  // output height
        OW = (IW - KW + PW_L + PW_R) / SW + 1;  // output width

    if(node.GetAttr<std::vector<std::string>>("data_layout")[0].size()>4)
      {
        IC = input_shape[1]*input_shape[4];                    // input channels
    }

    if(node.GetAttr<std::vector<std::string>>("kernel_layout")[0].size()>4)
      {
        OC = weight_shape[0]*weight_shape[weight_shape.size()-1];
        if (node.GetAttr<std::vector<std::string>>("kernel_layout")[0]=="OHWI8o")
        {
          OC = weight_shape[0]*weight_shape[weight_shape.size()-1];
          KH = weight_shape[1];
          KW = weight_shape[2];
          OH = (IH - KH + PH_L + PH_R) / SH + 1;
          OW = (IW - KW + PW_L + PW_R) / SW + 1;
          // std::cout<<"conv  OH "<<OH<<' '<<IH<<' '<<KH<<' '<<PH_L<<' '<<PH_R<<' '<<SH<<std::endl;
          // std::cout<<"conv  OW "<<OW<<' '<<IW<<' '<<KW<<' '<<PW_L<<' '<<PW_R<<' '<<SW<<std::endl;
        }
    }
    // std::cout<<"conv "<<OC<<" "<<OH<<" "<<OW<<std::endl;
// 
    // std::cout<<"conv "<<IC<<' '<<IH<<' '<<IW<<' '<<OC<<' '<<KH<<' '<<KW<<' '<<OH<<' '<<OW<<std::endl;
    // // std::cout<<input_shape.size()<<' '<<std::endl;
    // for (auto in: input_shape)
    // {
    //   std::cout<<in<<' ';
    // }

    // std::cout<<std::endl;

    // std::cout<<weight_shape.size()<<' '<<std::endl;

    // for (auto in: weight_shape)
    // {
    //   std::cout<<in<<' ';
    // }
    // std::cout<<std::endl;

    // Memory shapes.
    dnnl::memory::dims src_dims = {N, IC, IH, IW};
    dnnl::memory::dims weights_dims = {OC, IC, KH, KW};
    if (groups > 1) {
      weights_dims = {groups, 1, IC / groups, KH, KW};
    }
    dnnl::memory::dims bias_dims = {OC};
    dnnl::memory::dims dst_dims = {N, OC, OH, OW};
    dnnl::memory::dims strides_dims = {SH, SW};
    dnnl::memory::dims padding_dims_l = {PH_L, PW_L};
    dnnl::memory::dims padding_dims_r = {PH_R, PW_R};

    // Memory descriptions.
    auto conv_src_md = dnnl::memory::desc(src_dims, dt::f32, src_df);
    auto conv_weights_md = dnnl::memory::desc(weights_dims, dt::f32, weight_df);
    auto conv_bias_md = dnnl::memory::desc(bias_dims, dt::f32, tag::any);
    auto conv_dst_md = dnnl::memory::desc(dst_dims, dt::f32, dst_df);

    // std::cout<<N<<' '<<OC<<' '<<OH<<' '<<OW<<std::endl;

    // Covn2d description.
    auto conv_desc = dnnl::convolution_forward::desc(
        dnnl::prop_kind::forward_inference, dnnl::algorithm::convolution_direct, conv_src_md,
        conv_weights_md, conv_bias_md, conv_dst_md, strides_dims, padding_dims_l, padding_dims_r);

    // Enable ReLU
    dnnl::primitive_attr attr;
    if (has_relu) {
      dnnl::post_ops ops;
      ops.append_eltwise(1.f, dnnl::algorithm::eltwise_relu, 0.f, 0.f);
      attr.set_post_ops(ops);
    }

    auto conv2d_prim_desc = dnnl::convolution_forward::primitive_desc(conv_desc, attr, engine_);

    // Push to the network.
    auto conv = dnnl::convolution_forward(conv2d_prim_desc);
    net_.push_back(conv);

    // Data memory.
    // ICHECK_EQ(node.GetAttr<std::vector<std::string>>("data_layout")[0], "NCHW");
    auto conv2d_src_memory = BindDNNLMemory(data_entry, {src_dims, dt::f32, src_df});

    // Weight memory.
    // ICHECK_EQ(node.GetAttr<std::vector<std::string>>("kernel_layout")[0], "OIHW");
    auto conv2d_weights_memory = BindDNNLMemory(
        weight_entry, {weights_dims, dt::f32, (groups > 1) ? tag::goihw : weight_df});

    // Bias memory.
    auto conv2d_bias_memory = dnnl::memory({bias_dims, dt::f32, tag::x}, engine_);
    if (has_bias) {
      auto bias_entry = node.GetInputs()[2];
      BindDNNLMemory(bias_entry, conv2d_bias_memory);
    } else {
      float bias[OC] = {0};
      write_to_dnnl_memory(bias, conv2d_bias_memory, OC * sizeof(float));
    }

    // Output memory.
    JSONGraphNodeEntry out_entry(nid, 0);
    auto conv2d_dst_memory = BindDNNLMemory(out_entry, conv2d_prim_desc.dst_desc());

    // std::cout<<conv2d_dst_memory.get_data_handle()<<std::endl;
    // Bind memory buffers.
    net_args_.push_back({{DNNL_ARG_SRC, conv2d_src_memory},
                         {DNNL_ARG_WEIGHTS, conv2d_weights_memory},
                         {DNNL_ARG_BIAS, conv2d_bias_memory},
                         {DNNL_ARG_DST, conv2d_dst_memory}});
  }

  void Dense(const size_t& nid, const bool has_bias = false) {
    auto node = nodes_[nid];

    // Setup attributes.
    auto data_entry = node.GetInputs()[0];
    auto weight_entry = node.GetInputs()[1];
    dnnl::memory::dims input_shape = nodes_[data_entry.id_].GetOpShape()[data_entry.index_];
    dnnl::memory::dims weight_shape = nodes_[weight_entry.id_].GetOpShape()[weight_entry.index_];

    dnnl::memory::dim B = input_shape[0],  // batch size
        IC = input_shape[1],               // input channels
        OC = weight_shape[0];              // output channels

    
    // std::cout<<"dense"<<IC<<" "<<OC<<std::endl;

    // Memory shapes.
    dnnl::memory::dims data_dims = {B, IC};
    dnnl::memory::dims weight_dims = {OC, IC};
    dnnl::memory::dims bias_dims = {OC};
    dnnl::memory::dims out_dims = {B, OC};

    // Memory descriptions.
    auto data_md = dnnl::memory::desc({data_dims, dt::f32, tag::nc});
    auto weight_md = dnnl::memory::desc({weight_dims, dt::f32, tag::nc});
    auto bias_md = dnnl::memory::desc({bias_dims, dt::f32, tag::x});
    auto dst_md = dnnl::memory::desc({out_dims, dt::f32, tag::nc});

    // Dense description.
    auto dense_desc = dnnl::inner_product_forward::desc(dnnl::prop_kind::forward_inference, data_md,
                                                        weight_md, bias_md, dst_md);
    auto dense_prim_desc = dnnl::inner_product_forward::primitive_desc(dense_desc, engine_);

    auto dense = dnnl::inner_product_forward(dense_prim_desc);
    net_.push_back(dense);

    // Memories.
    auto data_memory = BindDNNLMemory(data_entry, data_md);
    auto weight_memory = BindDNNLMemory(weight_entry, weight_md);
    auto bias_memory = dnnl::memory(bias_md, engine_);
    if (has_bias) {
      auto bias_entry = node.GetInputs()[2];
      BindDNNLMemory(bias_entry, bias_memory);
    } else {
      float bias[OC] = {0};
      write_to_dnnl_memory(bias, bias_memory, OC * sizeof(float));
    }
    // float bias[OC] = {0};
    // write_to_dnnl_memory(bias, bias_memory, OC * sizeof(float));
    JSONGraphNodeEntry out_entry(nid, 0);
    auto dst_memory = BindDNNLMemory(out_entry, dense_prim_desc.dst_desc());

    net_args_.push_back({{DNNL_ARG_SRC, data_memory},
                         {DNNL_ARG_WEIGHTS, weight_memory},
                         {DNNL_ARG_BIAS, bias_memory},
                         {DNNL_ARG_DST, dst_memory}});
  }

  void BatchNorm(const size_t& nid) {
    auto node = nodes_[nid];

    auto data_entry = node.GetInputs()[0];
    auto gamma_entry = node.GetInputs()[1];
    auto beta_entry = node.GetInputs()[2];
    auto mean_entry = node.GetInputs()[3];
    auto variance_entry = node.GetInputs()[4];
    dnnl::memory::dims data_shape = nodes_[data_entry.id_].GetOpShape()[data_entry.index_];
    dnnl::memory::dim IC = data_shape[1];

    // std::cout<<"batchnorm raw data shape";
    // for(auto in : data_shape)
    // {
    //   std::cout<<in<<" ";
    // }
    // std::cout<<std::endl;

    auto data_format = tag::abcd;

    if(data_shape.size()>4)
    {IC = IC * data_shape[data_shape.size()-1];
    data_shape[1] = IC;
    dnnl::memory::dims new_data_shape{1,2,3,4};
    for(int i=0; i<data_shape.size()-1; i++)
    {new_data_shape[i] = data_shape[i];}
    data_shape = new_data_shape;
    data_format = tag::aBcd8b;
    }

    // std::cout<<"BN "<<IC<<std::endl;
    
    float epsilon = std::stof(node.GetAttr<std::vector<std::string>>("epsilon")[0]);
    // std::cout<<"batchnorm new shape";
    // for(auto in : data_shape)
    // {
    //   std::cout<<in<<" ";
    // }
    // std::cout<<std::endl;
    
    // Memory description.
    dnnl::memory::desc data_md = dnnl::memory::desc({data_shape, dt::f32, data_format});//GenDNNLMemDescByShape(data_shape, dt::f32);

    // BN description.
    auto bn_desc = dnnl::batch_normalization_forward::desc(
        dnnl::prop_kind::forward_inference, data_md, epsilon,
        dnnl::normalization_flags::use_global_stats | dnnl::normalization_flags::use_scale_shift);
    auto bn_prim_desc = dnnl::batch_normalization_forward::primitive_desc(bn_desc, engine_);
    auto bn = dnnl::batch_normalization_forward(bn_prim_desc);
    net_.push_back(bn);

    // Memories.
    auto data_memory = BindDNNLMemory(data_entry, data_md);
    JSONGraphNodeEntry out_entry(nid, 0);
    auto out_memory = BindDNNLMemory(out_entry, data_md);
    auto mean_memory = BindDNNLMemory(mean_entry, bn_prim_desc.mean_desc());
    auto variance_memory = BindDNNLMemory(variance_entry, bn_prim_desc.variance_desc());

    // In DNNL, weight is composed of gamma+beta, so we point them to the same DNNL memory but
    // assign an offset to beta data for runtime serialization.
    auto weight_memory = BindDNNLMemory(gamma_entry, bn_prim_desc.weights_desc(), 0);
    BindDNNLMemory(beta_entry, weight_memory, IC);

    net_args_.push_back({{DNNL_ARG_SRC, data_memory},
                         {DNNL_ARG_DST, out_memory},
                         {DNNL_ARG_SCALE_SHIFT, weight_memory},
                         {DNNL_ARG_MEAN, mean_memory},
                         {DNNL_ARG_VARIANCE, variance_memory}});
  }

  void Relu(const size_t& nid) {
    auto node = nodes_[nid];

    auto data_entry = node.GetInputs()[0];
    dnnl::memory::dims shape = nodes_[data_entry.id_].GetOpShape()[data_entry.index_];
    auto data_format = tag::abcd;
    // std::cout<<"Relu raw "; 
    // for (auto i : shape)
    // {
    //   std::cout<<i<<" ";
    // }
    // std::cout<<std::endl;

    if(shape.size()>4)
    {
      data_format = tag::aBcd8b;
      shape[1] = shape[1] * shape[shape.size()-1];
      dnnl::memory::dims new_data_shape{1,2,3,4};
      for(int i=0; i<new_data_shape.size(); i++)
      {new_data_shape[i] = shape[i];}
      shape = new_data_shape;
    }

    // std::cout<<"Relu new "; 
    // for (auto i : shape)
    // {
    //   std::cout<<i<<" ";
    // }
    // // std::cout<<std::endl;
    // std::cout<<std::endl;

    auto data_md = dnnl::memory::desc{{shape}, dt::f32, data_format};

    auto relu_desc = dnnl::eltwise_forward::desc(dnnl::prop_kind::forward_inference,
                                                 dnnl::algorithm::eltwise_relu, data_md, 0);
    auto relu_prim_desc = dnnl::eltwise_forward::primitive_desc(relu_desc, engine_);
    ICHECK(data_md == relu_prim_desc.dst_desc());

    auto relu = dnnl::eltwise_forward(relu_prim_desc);
    net_.push_back(relu);

    auto data_memory = BindDNNLMemory(data_entry, data_md);
    auto out_md = dnnl::memory::desc(shape, dt::f32, data_format);
    JSONGraphNodeEntry out_entry(nid, 0);
    auto out_memory = BindDNNLMemory(out_entry, out_md);

    net_args_.push_back({{DNNL_ARG_SRC, data_memory}, {DNNL_ARG_DST, out_memory}});
  }

  void Add(const size_t& nid) {
    auto node = nodes_[nid];

    // Memory and compute description.
    std::vector<dnnl::memory::dims> data_dims;
    std::vector<dnnl::memory::desc> data_mds;
    std::vector<dnnl::memory> data_memories;

    ICHECK_EQ(node.GetInputs().size(), 2U);
    for (auto entry : node.GetInputs()) {
      auto data_shape = nodes_[entry.id_].GetOpShape()[entry.index_];
      dnnl::memory::desc data_md = GenDNNLMemDescByShape(data_shape, dt::f32);

      data_dims.push_back(data_shape);
      data_mds.push_back(data_md);
      data_memories.push_back(BindDNNLMemory(entry, data_md));

    // std::cout<<"add "; 
    // for (auto i : data_shape)
    // {
    //   std::cout<<i<<" ";
    // }
    // std::cout<<std::endl;

    }
    
    
    ICHECK(data_dims[0] == data_dims[1]);
    auto out_md = data_mds[0];
    JSONGraphNodeEntry out_entry(nid, 0);
    auto out_memory = BindDNNLMemory(out_entry, out_md);

    auto add_desc =
        dnnl::binary::desc(dnnl::algorithm::binary_add, data_mds[0], data_mds[1], out_md);
    auto add_prim_desc = dnnl::binary::primitive_desc(add_desc, engine_);
    auto add = dnnl::binary(add_prim_desc);
    net_.push_back(add);

    net_args_.push_back({{DNNL_ARG_SRC_0, data_memories[0]},
                         {DNNL_ARG_SRC_1, data_memories[1]},
                         {DNNL_ARG_DST, out_memory}});
  }

void Concat(const size_t& nid) {
    auto node = nodes_[nid];

    std::vector<dnnl::memory::desc> data_mds;
    std::vector<dnnl::memory> data_memories;
    
    auto data_entry = node.GetInputs()[0];
    dnnl::memory::dims shape = nodes_[data_entry.id_].GetOpShape()[data_entry.index_];
    int axis = std::stoi(node.GetAttr<std::vector<std::string>>("axis")[0]); // axis
    
    if (axis < 0) {
        ICHECK_LE(0 - axis, shape.size());
        axis = shape.size() + axis;
    }
    else ICHECK_LT(axis, shape.size());
    
    for (auto entry : node.GetInputs()) {
      auto data_shape = nodes_[entry.id_].GetOpShape()[entry.index_];
      dnnl::memory::desc data_md = GenDNNLMemDescByShape(data_shape, dt::f32);

      data_mds.push_back(data_md);
      data_memories.push_back(BindDNNLMemory(entry, data_md));
    }
    
    auto concat_prim_desc = dnnl::concat::primitive_desc(axis, data_mds, engine_);

    auto concat = dnnl::concat(concat_prim_desc);
    net_.push_back(concat);
    
    auto out_md = concat_prim_desc.dst_desc();
    JSONGraphNodeEntry out_entry(nid, 0);
    auto out_memory = BindDNNLMemory(out_entry, out_md);

    std::unordered_map<int, dnnl::memory> args;
    for (int i = 0; i < (int)data_memories.size(); i++) {
        args.insert({DNNL_ARG_MULTIPLE_SRC + i, data_memories[i]});
    }
    args.insert({DNNL_ARG_DST, out_memory});
    net_args_.push_back(args);
  }
  
  void MaxPool2d(const size_t& nid) {
    auto node = nodes_[nid];

    // Setup attributes.
    auto data_entry = node.GetInputs()[0];
    dnnl::memory::dims input_shape = nodes_[data_entry.id_].GetOpShape()[data_entry.index_];
    dnnl::memory::dim pool_size0 = std::stoi(node.GetAttr<std::vector<std::string>>("pool_size")[0]); //notice
    dnnl::memory::dim pool_size1 = std::stoi(node.GetAttr<std::vector<std::string>>("pool_size")[1]);
    std::vector<std::string> str_strides = node.GetAttr<std::vector<std::string>>("strides");
    std::vector<std::string> str_padding = node.GetAttr<std::vector<std::string>>("padding");
    std::vector<std::string> ceil_mode = node.GetAttr<std::vector<std::string>>("ceil_mode");
    dnnl::memory::dim dilation0 = std::stoi(node.GetAttr<std::vector<std::string>>("dilation")[0]);
    dnnl::memory::dim dilation1 = std::stoi(node.GetAttr<std::vector<std::string>>("dilation")[1]);
    auto src_df = layout_dict[node.GetAttr<std::vector<std::string>>("layout")[0]];
    auto dst_df = src_df;
    dnnl::memory::dim N = input_shape[0],
      IC = input_shape[1],
      IH = input_shape[2],
      IW = input_shape[2],
      KH = pool_size0,
      KW = pool_size1,
      PH_L = std::stoi(str_padding[1]),
      PH_R = std::stoi(str_padding[3]),
      PW_L = std::stoi(str_padding[0]),
      PW_R = std::stoi(str_padding[2]),
      SH = std::stoi(str_strides[0]),
      SW = std::stoi(str_strides[1]),
      DH = dilation0,
      DW = dilation1,
      OH = (IH - KH + PH_L + PH_R) / SH + 1,
      OW = (IW - KW + PW_L + PW_R) / SW + 1;

    if(node.GetAttr<std::vector<std::string>>("layout")[0].size()>4)
      {
        IC = input_shape[1]*input_shape[4];                    // input channels
    }
    // Correctness:
    /* bool ceil_mode = node.GetAttr<bool>("ceil_mode");
    if(ceil_mode) {
      std::cout << "# Using ceiling mode;" << std::endl;
    }
    std::cout << "# Checking ceil_mode, poolsize, dilation: " << std::endl;*/

    // Memory shapes.
    dnnl::memory::dims src_dims = {N, IC, IH, IW};
    //dnnl::memory::dims kernel_dims = {OC, IC, KH, KW};
    dnnl::memory::dims kernel_dims = {KH, KW}; // modified
    dnnl::memory::dims dst_dims = {N, IC, OH, OW};
    dnnl::memory::dims strides_dims = {SH, SW};
    dnnl::memory::dims padding_dims_l = {PH_L, PW_L};
    dnnl::memory::dims padding_dims_r = {PH_R, PW_R};
    dnnl::memory::dims dilation = {DH, DW};

    // Memory descriptions.
    auto pool_src_md = dnnl::memory::desc(src_dims, dt::f32, src_df);
    auto pool_dst_md = dnnl::memory::desc(dst_dims, dt::f32, dst_df);

    // MaxPool2d description.
    // prop_kind, alg_kind, src_desc, dst_desc,
    // strides, kernel, padding_l, padding_r
    auto maxpool_desc = dnnl::pooling_forward::desc(
      dnnl::prop_kind::forward_inference, dnnl::algorithm::pooling_max,
      pool_src_md, pool_dst_md, strides_dims, kernel_dims,
      padding_dims_l, padding_dims_r
    );
    auto maxpool_prim_desc = dnnl::pooling_forward::primitive_desc(maxpool_desc, engine_);

    // Push to the network.
    auto pool = dnnl::pooling_forward(maxpool_prim_desc);
    net_.push_back(pool);

    // Memories.
    // ICHECK_EQ(node.GetAttr<std::vector<std::string>>("layout")[0], "NCHW");
    auto pool2d_src_memory = BindDNNLMemory(data_entry, pool_src_md);

    JSONGraphNodeEntry out_entry(nid, 0);
    ICHECK(pool_dst_md == maxpool_prim_desc.dst_desc());
    auto pool2d_dst_memory = BindDNNLMemory(out_entry, maxpool_prim_desc.dst_desc());

    // Bind memory buffers.
    net_args_.push_back({
      {DNNL_ARG_SRC, pool2d_src_memory},
      {DNNL_ARG_DST, pool2d_dst_memory}
    });
  }

  void AvgPool2d(const size_t& nid){
    auto node = nodes_[nid];

    // Setup attributes.
    auto data_entry = node.GetInputs()[0];
    dnnl::memory::dims input_shape = nodes_[data_entry.id_].GetOpShape()[data_entry.index_];
    dnnl::memory::dim pool_size0 = std::stoi(node.GetAttr<std::vector<std::string>>("pool_size")[0]); //notice
    dnnl::memory::dim pool_size1 = std::stoi(node.GetAttr<std::vector<std::string>>("pool_size")[1]);
    std::vector<std::string> str_strides = node.GetAttr<std::vector<std::string>>("strides");
    std::vector<std::string> str_padding = node.GetAttr<std::vector<std::string>>("padding");
    std::vector<std::string> ceil_mode = node.GetAttr<std::vector<std::string>>("ceil_mode");
    dnnl::memory::dim dilation0 = std::stoi(node.GetAttr<std::vector<std::string>>("dilation")[0]);
    dnnl::memory::dim dilation1 = std::stoi(node.GetAttr<std::vector<std::string>>("dilation")[1]);

    // Attributes related to AvgPool
    // bool count_include_pad = node.GetAttr<bool>("count_include_pad");
    // std::cout << "# count_include_pad or not?" << std::endl; // debug purpose
    // std::cout << node.GetAttr<std::vector<std::string>>("count_include_pad")[0] << std::endl;
    int int_countpad = std::stoi(node.GetAttr<std::vector<std::string>>("count_include_pad")[0]); //notice
    bool count_include_pad = int_countpad != 0 ? true : false;
    auto alg_kind = count_include_pad ?
      dnnl::algorithm::pooling_avg_include_padding :
      dnnl::algorithm::pooling_avg_exclude_padding ;

    dnnl::memory::dim N = input_shape[0],
      IC = input_shape[1],
      IH = input_shape[2],
      IW = input_shape[2],
      KH = pool_size0,
      KW = pool_size1,
      PH_L = std::stoi(str_padding[1]),
      PH_R = std::stoi(str_padding[3]),
      PW_L = std::stoi(str_padding[0]),
      PW_R = std::stoi(str_padding[2]),
      SH = std::stoi(str_strides[0]),
      SW = std::stoi(str_strides[1]),
      DH = dilation0,
      DW = dilation1,
      OH = (IH - KH + PH_L + PH_R) / SH + 1,
      OW = (IW - KW + PW_L + PW_R) / SW + 1;

    // Memory shapes.
    dnnl::memory::dims src_dims = {N, IC, IH, IW};
    dnnl::memory::dims kernel_dims = {KH, KW};
    dnnl::memory::dims dst_dims = {N, IC, OH, OW};
    dnnl::memory::dims strides_dims = {SH, SW};
    dnnl::memory::dims padding_dims_l = {PH_L, PW_L};
    dnnl::memory::dims padding_dims_r = {PH_R, PW_R};
    dnnl::memory::dims dilation = {DH, DW};

    // Memory descriptions.
    auto pool_src_md = dnnl::memory::desc(input_shape, dt::f32, tag::nchw);
    auto pool_dst_md = dnnl::memory::desc(dst_dims, dt::f32, tag::nchw); 

    // AvgPool2d description.
    auto avgpool_desc = dnnl::pooling_forward::desc(
      dnnl::prop_kind::forward_inference, alg_kind, pool_src_md,
      pool_dst_md, strides_dims, kernel_dims,
      padding_dims_l, padding_dims_r
    );
    auto avgpool_prim_desc = dnnl::pooling_forward::primitive_desc(avgpool_desc, engine_, true);//allow_enpty=true

    // Push to the network.
    auto pool = dnnl::pooling_forward(avgpool_prim_desc);
    net_.push_back(pool);

    // Memories
    ICHECK_EQ(node.GetAttr<std::vector<std::string>>("layout")[0], "NCHW"); // naming
    auto pool2d_src_memory = BindDNNLMemory(data_entry, pool_src_md);

    JSONGraphNodeEntry out_entry(nid, 0);
    auto pool2d_dst_memory = BindDNNLMemory(out_entry, avgpool_prim_desc.dst_desc());

    // Bind memory buffers.
    net_args_.push_back({
      {DNNL_ARG_SRC, pool2d_src_memory},
      {DNNL_ARG_DST, pool2d_dst_memory}
    });
  }

  // Read from DNNL memory (+offset) and write to the handle.
  inline void read_from_dnnl_memory(void* handle, const dnnl::memory& mem, size_t size,
                                    size_t offset = 0) {
    uint8_t* src = static_cast<uint8_t*>(mem.get_data_handle());
    // std::cout<<"Read from DNNL memory (+offset) and write to the handle."<<src<<std::endl;
    std::copy(src + offset, src + offset + size, static_cast<uint8_t*>(handle));
  }

  // Read from the handle and write to DNNL memory (+offset).
  inline void write_to_dnnl_memory(void* handle, const dnnl::memory& mem, size_t size,
                                   size_t offset = 0) {
    uint8_t* dst = static_cast<uint8_t*>(mem.get_data_handle());
    // std::cout<<"Read from the handle and write to DNNL memory (+offset)."<<dst<<std::endl;
    std::copy(reinterpret_cast<uint8_t*>(handle), reinterpret_cast<uint8_t*>(handle) + size,
              dst + offset);
  }

  // Generate DNNL memory description and infer the data layout by the given shape.
  inline dnnl::memory::desc GenDNNLMemDescByShape(const dnnl::memory::dims& shape, dt dtype) {
    dnnl::memory::desc data_md;
    switch (shape.size()) {
      case 1:
        data_md = dnnl::memory::desc({shape, dtype, tag::a});
        break;
      case 2:
        data_md = dnnl::memory::desc({shape, dtype, tag::ab});
        break;
      case 3:
        data_md = dnnl::memory::desc({shape, dtype, tag::abc});
        break;
      case 4:
        data_md = dnnl::memory::desc({shape, dtype, tag::abcd});
        break;
      case 5:
        data_md = dnnl::memory::desc({shape, dtype, tag::abcde});
        break;
      default:
        LOG(FATAL) << "Unsupported data shape dimension: " << shape.size();
        break;
    }
    return data_md;
  }

  /* The dnnl engine. */
  dnnl::engine engine_;
  /* The dnnl stream. */
  dnnl::stream stream_;
  /* The network layers that are represented in dnnl primitives. */
  std::vector<dnnl::primitive> net_;
  /* The memory that is consumed by arguments. */
  std::vector<std::unordered_map<int, dnnl::memory>> net_args_;
  /* The entry ID to its corresponding output memory. */
  std::unordered_map<uint32_t, std::pair<dnnl::memory, size_t>> entry_out_mem_;
};

runtime::Module DNNLJSONRuntimeCreate(String symbol_name, String graph_json,
                                      const Array<String>& const_names) {
  auto n = make_object<DNNLJSONRuntime>(symbol_name, graph_json, const_names);
  return runtime::Module(n);
}

TVM_REGISTER_GLOBAL("runtime.DNNLJSONRuntimeCreate").set_body_typed(DNNLJSONRuntimeCreate);

TVM_REGISTER_GLOBAL("runtime.module.loadbinary_dnnl_json")
    .set_body_typed(JSONRuntimeBase::LoadFromBinary<DNNLJSONRuntime>);

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm
