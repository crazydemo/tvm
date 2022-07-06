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
#include "dnnl_utils.h"

namespace tvm {
namespace runtime {
namespace contrib {

using namespace tvm::runtime;
using namespace tvm::runtime::json;
using namespace dnnl::graph;
using std::regex_replace; using std::regex;

class DNNLGraphJSONRuntime : public JSONRuntimeBase {
  using tag = dnnl::memory::format_tag;
  using dt = dnnl::memory::data_type;
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
    
    /// execute the compile partition
    for (size_t i = 0; i < cps_.size(); ++i) {
      cps_.at(i).execute(strm_, input_args_.at(i), output_args_.at(i));
    }

  }

 private:
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
        } else {
          LOG(FATAL) << "Unsupported op: " << op_name;
        }
      }
    }
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
    op conv {nid, op::kind::Convolution, {src_desc, wgh_desc},
            {dst_desc}, "conv"};
    conv.set_attr<std::vector<int64_t>>("strides", strides);
    conv.set_attr<std::vector<int64_t>>("pads_begin", padding_l);
    conv.set_attr<std::vector<int64_t>>("pads_end", padding_r);
    conv.set_attr<std::vector<int64_t>>("dilations", dilates);
    conv.set_attr<std::string>("data_format", data_format);  
    conv.set_attr<std::string>("filter_format", filter_format);
    conv.set_attr<int64_t>("groups", groups);

    /// add the ops to the graph
    g.add_op(conv);
    // Partition the graph.
    auto partitions = g.get_partitions();
    if (partitions.size() != 1)
        throw std::runtime_error(
                "cpu_simple_pattern_f32: incorrect partition number");
    auto partition = partitions[0];
    /// compile the partition. the shape and layout of output logical tensor
    /// will be inferred during the compilation.
    std::vector<logical_tensor> inputs = partition.get_in_ports();
    std::vector<logical_tensor> outputs = partition.get_out_ports();
    auto cp = partition.compile(inputs, outputs, eng_);

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

  logical_tensor GetInput(const size_t& nid, const int idx) {
    if (idx == -1) return {};  // -1 reserved value for empty input.

    const JSONGraphNode& node = nodes_[nid];

    ICHECK_LT(idx, node.GetInputs().size());
    auto data_entry = node.GetInputs()[idx];

    auto shape = nodes_[data_entry.id_].GetOpShape()[data_entry.index_];
    auto dtype = nodes_[data_entry.id_].GetOpDataType()[data_entry.index_];
    auto eid = node_row_ptr_[data_entry.id_] + data_entry.index_;
    // todo @crazydemo alter dtype.
    logical_tensor res {eid, data_type::f32, shape, layout_type::strided};
    return res;
  }

  logical_tensor GetOutput(const size_t& nid, const int idx) {
    if (idx == -1) return {};  // -1 reserved value for empty input.

    const JSONGraphNode& node = nodes_[nid];

    ICHECK_LT(idx, node.GetNumOutput());
    auto shape = node.GetOpShape()[idx];
    auto dtype = node.GetOpDataType()[idx];
    auto eid = node_row_ptr_[nid] + static_cast<uint32_t>(idx);

    ICHECK(data_entry_[eid] == nullptr);
    logical_tensor res {eid, data_type::f32, shape, layout_type::strided};
    return res;
  }

  tensor CreateTensor(logical_tensor desc) {
    auto eid = desc.get_id();
    // auto const_dl_tensor = data_entry_[eid];

    tensor res {desc, eng_, {}};
    // }
    entry_out_mem_[eid] = res;
    return res;
  }

  // Read from DNNL memory (+offset) and write to the handle.
  inline void read_from_dnnl_memory(void* handle, const dnnl::memory& mem, size_t size,
                                    size_t offset = 0) {
    uint8_t* src = static_cast<uint8_t*>(mem.get_data_handle());
    std::copy(src + offset, src + offset + size, static_cast<uint8_t*>(handle));
  }

  // Read from the handle and write to DNNL memory (+offset).
  inline void write_to_dnnl_memory(void* handle, const dnnl::memory& mem, size_t size,
                                   size_t offset = 0) {
    uint8_t* dst = static_cast<uint8_t*>(mem.get_data_handle());
    std::copy(reinterpret_cast<uint8_t*>(handle), reinterpret_cast<uint8_t*>(handle) + size,
              dst + offset);
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
  // std::unordered_map<uint32_t, std::pair<dnnl::memory, size_t>> entry_out_mem_;
  /* The dnnl engine. */
  engine eng_;
  /* The dnnl stream. */
  stream strm_;
  /* The network layers that are represented in dnnl primitives. */
  // graph g_ = engine::kind::cpu;
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