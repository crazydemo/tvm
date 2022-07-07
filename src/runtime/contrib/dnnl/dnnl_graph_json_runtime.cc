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
#include <regex>
#include <string>
#include <vector>

#include "../json/json_node.h"
#include "../json/json_runtime.h"
#include "oneapi/dnnl/dnnl_graph.hpp"
#include "dnnl.hpp"
#include "dnnl_graph_utils.h"

namespace tvm {
namespace runtime {
namespace contrib {

using namespace tvm::runtime;
using namespace tvm::runtime::json;
using namespace dnnl::graph;
using std::regex_replace; using std::regex;

class DNNLGraphJSONRuntime : public JSONRuntimeBase {
  using data_type = logical_tensor::data_type;
  using layout_type = logical_tensor::layout_type;

 public:
  DNNLGraphJSONRuntime(const std::string& symbol_name, const std::string& graph_json,
                  const Array<String> const_names)
      : JSONRuntimeBase(symbol_name, graph_json, const_names) {}

  const char* type_key() const { return "dnnl_graph_json"; }

  void Init(const Array<NDArray>& consts) override {
    BuildEngine();

    ICHECK_EQ(consts.size(), const_idx_.size())
        << "The number of input constants must match the number of required.";

    // Setup constants entries for weights.
    SetupConstants(consts);
  }

  void Run() override {
    /// create tensors for the execution
    // Fill in the input buffers.
    for (size_t i = 0; i < input_nodes_.size(); ++i) {
      auto eid = EntryID(input_nodes_[i], 0);
      entry_out_mem_[eid].set_data_handle(data_entry_[eid]->data);
    }
    for (size_t i = 0; i < outputs_.size(); ++i) {
      auto eid = EntryID(outputs_[i]);
      entry_out_mem_[eid].set_data_handle(data_entry_[eid]->data);
    }
    
    std::cout<<"dnnl graph executing ..."<<std::endl;
    /// execute the compile partition
    for (size_t i = 0; i < cps_.size(); ++i) {
      cps_.at(i).execute(strm_, input_args_.at(i), output_args_.at(i));
    }
  }

 private:
  
  const std::map<std::string, op::kind> elt_name2kind{
      {"nn.relu", op::kind::ReLU},
      {"tanh", op::kind::Tanh},
      {"sigmoid", op::kind::Sigmoid},
  };

  const std::map<std::string, op::kind> binary_name2kind{
      {"add", op::kind::Add},
  };

  // Build up the engine based on the input graph.
  void BuildEngine() {
    /// create a new engine and stream
    eng_ = {engine::kind::cpu, 0};
    strm_ = {eng_};
    graph g(engine::kind::cpu);
    // Build subgraph engine.
    for (size_t nid = 0; nid < nodes_.size(); ++nid) {
      const auto& node = nodes_[nid];
      if (node.GetOpType() == "kernel") {
        ICHECK_EQ(node.GetOpType(), "kernel");
        auto op_name = node.GetOpName();
        if (op_name == "nn.conv2d") {
          Convolution(nid, g);
        } else if (elt_name2kind.count(op_name)) {
          Eltwise(nid, g);
        } else if (binary_name2kind.count(op_name)) {
          Binary(nid, g);
        } else if (op_name == "nn.bias_add") {
          BiasAdd(nid, g);
        } else {
          LOG(FATAL) << "Unsupported op: " << op_name;
        }
      }
    }
    Compile(g);
  }

  void Convolution(const size_t& nid, graph& g) {
    auto node = nodes_[nid];

    // Setup attributes.
    auto src_desc = GetInput(nid, 0);
    auto wgh_desc = GetInput(nid, 1);
    auto dst_desc = GetOutput(nid, 0);
    auto strides = GetNodeAttr<std::vector<int64_t>>(node, "strides");
    auto dilates = GetNodeAttr<std::vector<int64_t>>(node, "dilation");
    auto padding = GetNodeAttr<std::vector<int64_t>>(node, "padding");
    std::vector<int64_t> padding_l(padding.begin(), padding.begin() + padding.size() / 2);
    std::vector<int64_t> padding_r(padding.begin() + padding.size() / 2, padding.end());
    auto groups = GetNodeAttr<int>(node, "groups");
    std::string data_layout = node.GetAttr<std::vector<std::string>>("data_layout")[0];
    std::string kernel_layout = node.GetAttr<std::vector<std::string>>("kernel_layout")[0];
    // todo @crazydemo check the validaty of layout string
    std::string data_format = regex_replace(data_layout, regex("(D?)(H?)W"), "X");
    std::string filter_format = regex_replace(kernel_layout, regex("(D?)(H?)W"), "X");
    /// create op conv
    std::cout<<"conv" + std::to_string(nid)<<std::endl;
    op conv {nid, op::kind::Convolution, {src_desc, wgh_desc},
            {dst_desc}, "conv" + std::to_string(nid)};
    conv.set_attr<std::vector<int64_t>>("strides", strides);
    conv.set_attr<std::vector<int64_t>>("pads_begin", padding_l);
    conv.set_attr<std::vector<int64_t>>("pads_end", padding_r);
    conv.set_attr<std::vector<int64_t>>("dilations", dilates);
    conv.set_attr<std::string>("data_format", data_format);  
    conv.set_attr<std::string>("filter_format", filter_format);
    conv.set_attr<int64_t>("groups", groups);

    /// add the ops to the graph
    g.add_op(conv);
  }

  void Eltwise(const size_t& nid, graph& g) {
    auto node = nodes_[nid];
    auto op_name = node.GetOpName();
    auto op_kind = elt_name2kind.at(op_name);

    // Setup attributes.
    auto src_desc = GetInput(nid, 0);
    auto dst_desc = GetOutput(nid, 0);

    /// create op conv
    std::cout<<op_name + std::to_string(nid)<<std::endl;
    op elt {nid, op_kind, {src_desc}, {dst_desc}, op_name + std::to_string(nid)};
    /// add the ops to the graph
    g.add_op(elt);
  }

  void Binary(const size_t& nid, graph& g) {
    auto node = nodes_[nid];
    auto op_name = node.GetOpName();
    auto op_kind = binary_name2kind.at(op_name);

    // Setup attributes.
    auto src1_desc = GetInput(nid, 0);
    auto src2_desc = GetInput(nid, 1);
    auto dst_desc = GetOutput(nid, 0);

    /// create op conv
    std::cout<<op_name + std::to_string(nid)<<std::endl;
    op elt {nid, op_kind, {src1_desc, src2_desc}, {dst_desc}, op_name + std::to_string(nid)};
    /// add the ops to the graph
    g.add_op(elt);
  }

  void BiasAdd(const size_t& nid, graph& g) {
    auto node = nodes_[nid];
    auto op_name = node.GetOpName();

    // Setup attributes.
    auto src1_desc = GetInput(nid, 0);
    auto src2_desc = GetInput(nid, 1);
    auto dst_desc = GetOutput(nid, 0);

    /// create op conv
    std::cout<<op_name + std::to_string(nid)<<std::endl;
    op elt {nid, op::kind::BiasAdd, {src1_desc, src2_desc}, {dst_desc}, op_name + std::to_string(nid)};
    /// add the ops to the graph
    g.add_op(elt);
  }

  void Compile(graph& g) {
    // Partition the graph.
    std::cout<<"dnnl graph compiling ..."<<std::endl;
    auto partitions = g.get_partitions();
    if (partitions.size() < 1)
        throw std::runtime_error(
                "cpu_simple_pattern_f32: incorrect partition number");
    
    /// compile the partition. the shape and layout of output logical tensor
    /// will be inferred during the compilation.
    for (int pi = 0; pi < partitions.size(); ++pi) {
      std::vector<logical_tensor> inputs = partitions[pi].get_in_ports();
      std::vector<logical_tensor> outputs = partitions[pi].get_out_ports();
      auto cp = partitions[pi].compile(inputs, outputs, eng_);

      std::vector<tensor> in_trs, out_trs;
      for (int i = 0; i < inputs.size(); ++i) {
        tensor tr = CreateTensor(inputs[i]);
        in_trs.push_back(tr);
      }
      for (int i = 0; i < outputs.size(); ++i) {
        tensor tr = CreateTensor(outputs[i]);
        out_trs.push_back(tr);
      }

      cps_.push_back(cp);
      input_args_.push_back(in_trs);
      output_args_.push_back(out_trs);
    }
  }

  template <typename T, std::enable_if_t<std::is_integral<T>::value, int> = 0>
  T AttrConvert(std::vector<std::string> val) {
    ICHECK_EQ(val.size(), 1);
    return std::stol(val[0]);
  }

  template <typename T, std::enable_if_t<std::is_floating_point<T>::value, int> = 0>
  T AttrConvert(std::vector<std::string> val) {
    ICHECK_EQ(val.size(), 1);
    return std::stof(val[0]);
  }

  template <typename T, std::enable_if_t<std::is_same<T, std::string>::value, int> = 0>
  T AttrConvert(std::vector<std::string> val) {
    ICHECK_EQ(val.size(), 1);
    return val[0];
  }

  template <typename T,
            std::enable_if_t<std::is_same<T, std::vector<typename T::value_type>>::value, int> = 0>
  T AttrConvert(std::vector<std::string> val) {
    T res;
    for (const auto& el : val) res.push_back(AttrConvert<typename T::value_type>({el}));
    return res;
  }

  /*!
   * \brief Helper to extract node attribute with ability to specify default value and result type.
   */
  template <typename T>
  const T GetNodeAttr(const json::JSONGraphNode& node, std::string name,
                      std::vector<std::string> def = {}) {
    auto attr = node.HasAttr(name) ? node.GetAttr<std::vector<std::string>>(name) : def;
    return AttrConvert<T>(attr);
  }

  logical_tensor GetInput(const size_t& nid, const int idx, layout_type lt = layout_type::strided) {
    if (idx == -1) return {};  // -1 reserved value for empty input.

    const JSONGraphNode& node = nodes_[nid];

    ICHECK_LT(idx, node.GetInputs().size());
    auto data_entry = node.GetInputs()[idx];

    auto shape = nodes_[data_entry.id_].GetOpShape()[data_entry.index_];
    auto dtype = nodes_[data_entry.id_].GetOpDataType()[data_entry.index_];
    auto eid = node_row_ptr_[data_entry.id_] + data_entry.index_;
    // todo @crazydemo alter dtype.
    // auto dgraph_dtype = dtype_dl2dgraph(dtype);
    logical_tensor res {eid, data_type::f32, shape, lt};
    return res;
  }

  logical_tensor GetOutput(const size_t& nid, const int idx, layout_type lt = layout_type::strided) {
    if (idx == -1) return {};  // -1 reserved value for empty input.

    const JSONGraphNode& node = nodes_[nid];

    ICHECK_LT(idx, node.GetNumOutput());
    auto shape = node.GetOpShape()[idx];
    auto dtype = node.GetOpDataType()[idx];
    auto eid = node_row_ptr_[nid] + static_cast<uint32_t>(idx);

    ICHECK(data_entry_[eid] == nullptr);
    // auto dgraph_dtype = dtype_dl2dgraph(dtype);
    logical_tensor res {eid, data_type::f32, shape, lt};
    return res;
  }

  tensor CreateTensor(logical_tensor desc) {
    auto eid = desc.get_id();
    tensor res {desc, eng_, {}};
    entry_out_mem_[eid] = res;
    return res;
  }

  /* The dnnl engine. */
  engine eng_;
  /* The dnnl stream. */
  stream strm_;

  /* The network layers that are represented in dnnl primitives. */
  /* The memory that is consumed by arguments. */
  std::vector<std::unordered_map<int, std::vector<logical_tensor>>> partition_args_;
  std::vector<std::vector<tensor>> input_args_;
  std::vector<std::vector<tensor>> output_args_;
  /* The entry ID to its corresponding output memory. */
  std::unordered_map<uint32_t, tensor> entry_out_mem_;
  std::vector<compiled_partition> cps_;
};

runtime::Module DNNLGraphJSONRuntimeCreate(String symbol_name, String graph_json,
                                      const Array<String>& const_names) {
  auto n = make_object<DNNLGraphJSONRuntime>(symbol_name, graph_json, const_names);
  return runtime::Module(n);
}

TVM_REGISTER_GLOBAL("runtime.DNNLGraphJSONRuntimeCreate").set_body_typed(DNNLGraphJSONRuntimeCreate);

TVM_REGISTER_GLOBAL("runtime.module.loadbinary_dnnl_json")
    .set_body_typed(JSONRuntimeBase::LoadFromBinary<DNNLGraphJSONRuntime>);

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm