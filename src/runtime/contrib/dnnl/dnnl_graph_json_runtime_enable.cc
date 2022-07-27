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
#include "dnnl.hpp"
#include "oneapi/dnnl/dnnl_graph.hpp"

namespace tvm {
namespace runtime {
namespace contrib {

using namespace tvm::runtime;
using namespace tvm::runtime::json;
using namespace dnnl::graph;
using std::regex;
using std::regex_replace;

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
    // for (size_t i = 0; i < input_nodes_.size(); ++i) {
    //   auto eid = EntryID(input_nodes_[i], 0);
    //   entry_out_mem_[eid].set_data_handle(data_entry_[eid]->data);
    // }
    // for (size_t i = 0; i < outputs_.size(); ++i) {
    //   auto eid = EntryID(outputs_[i]);
    //   entry_out_mem_[eid].set_data_handle(data_entry_[eid]->data);
    // }
    // Fill in the input buffers.
    for (size_t i = 0; i < input_nodes_.size(); ++i) {
      auto eid = EntryID(input_nodes_[i], 0);
      SetBuffer(eid, entry_out_strided_desc_[eid].get_mem_size());
    }
    for (size_t i = 0; i < outputs_.size(); ++i) {
      auto eid = EntryID(outputs_[i]);
      SetBuffer(eid, entry_out_strided_desc_[eid].get_mem_size());
    }

    std::cout << "dnnl graph executing ..." << std::endl;
    /// execute the compile partition
    for (size_t i = 0; i < cps_.size(); ++i) {
      cps_.at(i).execute(strm_, input_args_[i], output_args_[i]);
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

  std::map<std::string, std::string> op_map{
      {"bias", "nn.bias_add"},
      {"relu", "nn.relu"},
      {"tanh", "tanh"},
      {"sigmoid", "sigmoid"},
      {"nn.deconv2d", "nn.conv2d_transpose"},
      {"nn.deconv3d", "nn.conv3d_transpose"},
  };

  std::vector<std::string> ParsingOpList(const std::string& pattern_name,
                                         std::string interval = "_") {
    ICHECK_NE(pattern_name, "");
    std::vector<std::string> op_list;
    size_t pos = 0, start = 0;

    while ((pos = pattern_name.find(interval, start)) != std::string::npos) {
      std::string op_name = pattern_name.substr(start, pos - start);
      std::cout << "parsing list op_name: " << op_name << std::endl;
      if (op_name.find("dnnl") != std::string::npos) {
        op_name.replace(op_name.find("dnnl"), 4, "nn");
        if (op_name.find("deconv") != std::string::npos) {
          op_name = op_map[op_name];
        }
      } else {
        op_name = op_map[op_name];
      }
      if (pos > start) op_list.push_back(op_name);
      start = pos + interval.size();
    }
    if (pattern_name.size() > start) {
      op_list.push_back(op_map[pattern_name.substr(start)]);
    }
    return op_list;
  }

  // Build up the engine based on the input graph.
  void BuildEngine() {
    /// create a new engine and stream
    eng_ = {engine::kind::cpu, 0};
    strm_ = {eng_};
    graph g(engine::kind::cpu);
    size_t nid;
    graph_pat_idx_ = 0;
    graph_op_idx_ = 0;
    // Build subgraph engine.
    for (nid = 0; nid < nodes_.size(); ++nid) {
      const auto& node = nodes_[nid];
      if (node.GetOpType() == "kernel") {
        ICHECK_EQ(node.GetOpType(), "kernel");
        ConstructPartition(nid, g);
      }
    }
    // op end{nid + 1, op::kind::End, {tmp_end_}, {}, "end" + std::to_string(nid + 1)};
    Compile(g);
  }

  void ConstructPartition(const size_t& nid, graph& g) {
    pat_op_idx_ = graph_pat_idx_;
    const auto& node = nodes_[nid];
    auto pat_name = node.GetOpName();
    std::vector<std::string> op_names = ParsingOpList(pat_name);
    for (auto op_name : op_names) {
      std::cout << "op_name: " << op_name << std::endl;
      if (op_name == "nn.conv2d") {
        Convolution(nid, g);
      } else if (op_name == "nn.bias_add") {
        BiasAdd(nid, g);
      } else {
        LOG(FATAL) << "Unsupported op: " << op_name;
      }
      ++pat_op_idx_;
    }
  }

  void SetBuffer(uint32_t eid, size_t mem_bytes) {
    tensor mem = entry_out_mem_[eid];
    size_t offset = 0;
    // const size_t offset = entry_out_mem_[eid].second;
    const DLTensor* tensor = data_entry_[eid];
    if (tensor == nullptr || tensor->data == nullptr) {
      LOG(FATAL) << "empty data entry" << std::endl;
    }
    size_t tensor_bytes = GetDataSize(*tensor);
    void* buffer = tensor->data;
    if (mem_bytes == tensor_bytes) {
      // std::cout<<"mem_bytes: "<<mem_bytes<<std::endl;
      // std::cout<<"tensor_bytes: "<<tensor_bytes<<std::endl;
      if (offset == 0) {
        mem.set_data_handle(buffer);
      } else {
        LOG(FATAL) << "faild to fill a tensor with " << mem_bytes << " bytes using " << tensor_bytes
                   << " bytes buffer and offset=" << offset << std::endl;
      }
    } else if (mem_bytes > tensor_bytes) {
      LOG(FATAL) << "faild to fill a tensor with " << mem_bytes << " bytes using " << tensor_bytes
                 << " bytes buffer and offset=" << offset << std::endl;
    }
  }

  void Convolution(const size_t& nid, graph& g) {
    auto node = nodes_[nid];

    // Setup attributes.
    auto src_desc = GetInput(nid, pat_op_idx_);
    auto wgh_desc = GetInput(nid, pat_op_idx_ + 1);
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
    std::cout << "conv" + std::to_string(nid) << std::endl;
    op conv{
        nid, op::kind::Convolution, {src_desc, wgh_desc}, {dst_desc}, "conv" + std::to_string(nid)};
    conv.set_attr<std::vector<int64_t>>("strides", strides);
    conv.set_attr<std::vector<int64_t>>("pads_begin", padding_l);
    conv.set_attr<std::vector<int64_t>>("pads_end", padding_r);
    conv.set_attr<std::vector<int64_t>>("dilations", dilates);
    conv.set_attr<std::string>("data_format", data_format);
    conv.set_attr<std::string>("filter_format", filter_format);
    conv.set_attr<int64_t>("groups", groups);

    /// add the ops to the graph
    g.add_op(conv);

    tmp_end_ = dst_desc;
  }

  void BiasAdd(const size_t& nid, graph& g) {
    auto node = nodes_[nid];
    auto op_name = node.GetOpName();

    // Setup attributes.
    auto src1_desc = GetInput(nid, 0);
    auto src2_desc = GetInput(nid, 1);
    auto dst_desc = GetOutput(nid, 0);

    /// create op conv
    std::cout << op_name + std::to_string(nid) << std::endl;
    op elt{
        nid, op::kind::BiasAdd, {src1_desc, src2_desc}, {dst_desc}, op_name + std::to_string(nid)};
    /// add the ops to the graph
    g.add_op(elt);

    tmp_end_ = dst_desc;
  }

  void Compile(graph& g) {
    // Partition the graph.
    std::cout << "dnnl graph compiling ..." << std::endl;
    auto partitions = g.get_partitions();
    if (partitions.size() < 1)
      throw std::runtime_error("cpu_simple_pattern_f32: incorrect partition number");
    std::cout << "Number of returned partitions: " << partitions.size() << "\n";
    for (size_t i = 0; i < partitions.size(); ++i) {
      std::cout << "Partition[" << partitions[i].get_id()
                << "]'s supporting status: " << (partitions[i].is_supported() ? "true" : "false")
                << "\n";
    }

    /// compile the partition. the shape and layout of output logical tensor
    /// will be inferred during the compilation.
    for (int pi = 0; pi < partitions.size(); ++pi) {
      auto p = partitions[pi];
      if (p.is_supported()) {
        std::cout << "\nPartition[" << p.get_id() << "] is being processed.\n";
        std::vector<logical_tensor> inputs = p.get_in_ports();
        std::vector<logical_tensor> outputs = p.get_out_ports();
        std::vector<logical_tensor> inputs_;
        std::vector<logical_tensor> outputs_;
        for (auto it : inputs) {
          inputs_.push_back(entry_out_strided_desc_[it.get_id()]);
        }
        for (auto ot : outputs) {
          outputs_.push_back(entry_out_strided_desc_[ot.get_id()]);
        }
        auto cp = p.compile(inputs_, outputs_, eng_);

        std::vector<tensor> in_trs, out_trs;
        for (int i = 0; i < inputs.size(); ++i) {
          tensor tr = CreateTensor(inputs_[i]);
          in_trs.push_back(tr);
        }
        for (int i = 0; i < outputs.size(); ++i) {
          tensor tr = CreateTensor(outputs_[i]);
          out_trs.push_back(tr);
        }

        cps_.push_back(cp);
        input_args_.insert(std::make_pair(pi, in_trs));
        output_args_.insert(std::make_pair(pi, out_trs));
      } else {
        std::vector<size_t> unsupported_op_ids = p.get_ops();
        std::cout << "unsupported_op_ids: ";
        for (auto i : unsupported_op_ids) {
          std::cout << i << ", ";
        }
        std::cout << std::endl;
      }
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

  logical_tensor GetInput(const size_t& nid, const int idx, layout_type lt = layout_type::undef) {
    if (idx == -1) return {};  // -1 reserved value for empty input.

    const JSONGraphNode& node = nodes_[nid];

    ICHECK_LT(idx, node.GetInputs().size());
    auto data_entry = node.GetInputs()[idx];

    auto shape = nodes_[data_entry.id_].GetOpShape()[data_entry.index_];
    auto dtype = nodes_[data_entry.id_].GetOpDataType()[data_entry.index_];
    auto eid = node_row_ptr_[data_entry.id_] + data_entry.index_;
    if (entry_out_undef_desc_.count(eid)) return entry_out_undef_desc_[eid];
    // todo @crazydemo alter dtype.
    // auto dgraph_dtype = dtype_dl2dgraph(dtype);
    logical_tensor undef_res{eid, data_type::f32, shape, lt};
    logical_tensor strided_res{eid, data_type::f32, shape, layout_type::strided};
    entry_out_undef_desc_[eid] = undef_res;
    entry_out_strided_desc_[eid] = strided_res;
    return undef_res;
  }

  std::vector<logical_tensor> GetInputs(const size_t& nid, layout_type lt = layout_type::undef) {
    const JSONGraphNode& node = nodes_[nid];

    ICHECK_GT(node.GetInputs().size(), 0);
    std::vector<logical_tensor> inputs;
    for (auto data_entry : node.GetInputs()) {
      auto shape = nodes_[data_entry.id_].GetOpShape()[data_entry.index_];
      auto dtype = nodes_[data_entry.id_].GetOpDataType()[data_entry.index_];
      auto eid = node_row_ptr_[data_entry.id_] + data_entry.index_;
      std::cout << "input: " << eid << std::endl;
      if (entry_out_undef_desc_.count(eid)) inputs.push_back(entry_out_undef_desc_[eid]);
      // todo @crazydemo alter dtype.
      // auto dgraph_dtype = dtype_dl2dgraph(dtype);
      logical_tensor undef_res{eid, data_type::f32, shape, lt};
      logical_tensor strided_res{eid, data_type::f32, shape, layout_type::strided};
      entry_out_undef_desc_[eid] = undef_res;
      entry_out_strided_desc_[eid] = strided_res;
      inputs.push_back(undef_res);
    }
    return inputs;
  }

  logical_tensor GetOutput(const size_t& nid, const int idx, layout_type lt = layout_type::undef) {
    if (idx == -1) return {};  // -1 reserved value for empty input.

    const JSONGraphNode& node = nodes_[nid];

    ICHECK_LT(idx, node.GetNumOutput());
    auto shape = node.GetOpShape()[idx];
    auto dtype = node.GetOpDataType()[idx];
    auto eid = node_row_ptr_[nid] + static_cast<uint32_t>(idx);
    if (entry_out_undef_desc_.count(eid)) return entry_out_undef_desc_[eid];

    ICHECK(data_entry_[eid] == nullptr);
    // auto dgraph_dtype = dtype_dl2dgraph(dtype);
    logical_tensor undef_res{eid, data_type::f32, shape, lt};
    logical_tensor strided_res{eid, data_type::f32, shape, layout_type::strided};
    entry_out_undef_desc_[eid] = undef_res;
    entry_out_strided_desc_[eid] = strided_res;
    return undef_res;
  }

  tensor CreateTensor(logical_tensor desc) {
    auto eid = desc.get_id();
    tensor res{desc, eng_, {}};
    entry_out_mem_[eid] = res;
    return res;
  }

  /* The dnnl engine. */
  engine eng_;
  /* The dnnl stream. */
  stream strm_;

  unsigned int graph_pat_idx_;
  unsigned int graph_op_idx_;
  unsigned int pat_op_idx_;

  // /* The network layers that are represented in dnnl primitives. */
  // /* The memory that is consumed by arguments. */
  // std::vector<std::unordered_map<int, std::vector<logical_tensor>>> partition_args_;
  // std::unordered_map<int, std::vector<tensor>> input_args_;
  // std::unordered_map<int, std::vector<tensor>> output_args_;
  // /* The entry ID to its corresponding output memory. */
  // std::unordered_map<uint32_t, tensor> entry_out_mem_;
  // std::unordered_map<uint32_t, logical_tensor> entry_out_undef_desc_;
  // std::unordered_map<uint32_t, logical_tensor> entry_out_strided_desc_;
  // std::vector<compiled_partition> cps_;
  // logical_tensor tmp_end_;
};

runtime::Module DNNLGraphJSONRuntimeCreate(String symbol_name, String graph_json,
                                           const Array<String>& const_names) {
  auto n = make_object<DNNLGraphJSONRuntime>(symbol_name, graph_json, const_names);
  return runtime::Module(n);
}

TVM_REGISTER_GLOBAL("runtime.DNNLGraphJSONRuntimeCreate")
    .set_body_typed(DNNLGraphJSONRuntimeCreate);

TVM_REGISTER_GLOBAL("runtime.module.loadbinary_dnnl_json")
    .set_body_typed(JSONRuntimeBase::LoadFromBinary<DNNLGraphJSONRuntime>);

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm