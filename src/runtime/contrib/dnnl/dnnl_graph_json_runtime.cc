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

#include <algorithm>
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
    std::cout << "=====================Run=========================" << std::endl;
    for (size_t i = 0; i < input_nodes_.size(); ++i) {
      auto eid = EntryID(input_nodes_[i], 0);
      std::cout << "inputs: " << eid << std::endl;
      SetBuffer(eid, eid2desc_[eid].get_mem_size());
    }
    for (size_t i = 0; i < outputs_.size(); ++i) {
      auto eid = EntryID(outputs_[i]);
      std::cout << "outputs: " << eid << std::endl;
      SetBuffer(eid, eid2desc_[eid].get_mem_size());
    }

    std::cout << "dnnl graph executing ..." << std::endl;
    /// execute the compile partition
    for (size_t i = 0; i < cps_.size(); ++i) {
      std::cout << "executing " << i << std::endl;
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
    graph_op_idx_ = 0;
    max_eid_ = *max_element(node_row_ptr_.begin(), node_row_ptr_.end()) + 10;
    tmp_idx_ = 0;
    std::cout << "max_eid_: " << max_eid_ << std::endl;
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
    const auto& node = nodes_[nid];
    auto pat_name = node.GetOpName();
    std::vector<std::string> op_names = ParsingOpList(pat_name);
    std::vector<logical_tensor> pat_inputs = {};
    GetPatInputs(nid, pat_inputs);

    for (size_t i = 0; i < op_names.size(); ++i) {
      auto op_name = op_names[i];
      std::vector<logical_tensor> op_inputs;
      logical_tensor op_output;
      bool is_first = i == 0;
      bool is_last = i == op_names.size() - 1;
      std::cout << "op_name: " << op_name << std::endl;
      if (op_name == "nn.conv2d") {
        GetOpInputs(op_inputs, pat_inputs, 2, is_first);
        GetOutput(nid, op_output, is_last);
        Convolution(nid, op_inputs, op_output, g);
      } else if (op_name == "nn.bias_add") {
        GetOpInputs(op_inputs, pat_inputs, 2, is_first);
        GetOutput(nid, op_output, is_last);
        op bias_add{graph_op_idx_,
                    op::kind::BiasAdd,
                    op_inputs,
                    {op_output},
                    "bias_add" + std::to_string(graph_op_idx_)};
        std::string data_layout = node.GetAttr<std::vector<std::string>>("data_layout")[0];
        std::string data_format = regex_replace(data_layout, regex("(D?)(H?)W"), "X");
        bias_add.set_attr<std::string>("data_format", data_format);
        g.add_op(bias_add);
      } else if (op_name == "nn.relu") {
        GetOpInputs(op_inputs, pat_inputs, 1, is_first);
        GetOutput(nid, op_output, is_last);
        std::cout << "relu input: " << op_inputs.size() << std::endl;
        op relu{graph_op_idx_,
                    op::kind::ReLU,
                    op_inputs,
                    {op_output},
                    "relu" + std::to_string(graph_op_idx_)};
        g.add_op(relu);
      } else {
        LOG(FATAL) << "Unsupported op: " << op_name;
      }
      tmp_end_ = op_output;
      ++graph_op_idx_;
    }
  }

  void SetBuffer(uint32_t eid, size_t mem_bytes) {
    tensor mem = eid2tr_[eid];
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

  void Convolution(const size_t& nid, std::vector<logical_tensor>& inputs, logical_tensor& output,
                   graph& g) {
    auto node = nodes_[nid];

    // Setup attributes.
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
    std::cout << "conv" + std::to_string(graph_op_idx_) << std::endl;
    op conv{graph_op_idx_,
            op::kind::Convolution,
            inputs,
            {output},
            "conv" + std::to_string(graph_op_idx_)};
    conv.set_attr<std::vector<int64_t>>("strides", strides);
    conv.set_attr<std::vector<int64_t>>("pads_begin", padding_l);
    conv.set_attr<std::vector<int64_t>>("pads_end", padding_r);
    conv.set_attr<std::vector<int64_t>>("dilations", dilates);
    conv.set_attr<std::string>("data_format", data_format);
    conv.set_attr<std::string>("filter_format", filter_format);
    conv.set_attr<int64_t>("groups", groups);

    std::cout << "strides: ";
    for (auto i : strides) {
      std::cout << i << " ";
    }
    std::cout << std::endl;

    std::cout << "padding_l: ";
    for (auto i : padding_l) {
      std::cout << i << " ";
    }
    std::cout << std::endl;

    std::cout << "padding_r: ";
    for (auto i : padding_r) {
      std::cout << i << " ";
    }
    std::cout << std::endl;

    std::cout << "dilates: ";
    for (auto i : dilates) {
      std::cout << i << " ";
    }
    std::cout << std::endl;

    std::cout << "data_format: " << data_format << std::endl;
    std::cout << "filter_format: " << filter_format << std::endl;
    std::cout << "groups: " << groups << std::endl;

    /// add the ops to the graph
    g.add_op(conv);
  }

  void Compile(graph& g) {
    // Partition the graph.
    std::cout << "dnnl graph compiling ..." << std::endl;
    auto partitions = g.get_partitions();
    if (partitions.size() < 1)
      throw std::runtime_error("cpu_simple_pattern_f32: incorrect partition number");
    std::cout << "Number of returned partitions: " << partitions.size() << "\n";

    // for (const auto& partition : partitions) {
    for (int pi = 0; pi < partitions.size(); ++pi) {
      auto partition = partitions[pi];
      if (partition.is_supported()) {
        std::vector<logical_tensor> inputs = partition.get_in_ports();
        std::vector<logical_tensor> outputs = partition.get_out_ports();

        // update input logical tensors with concrete shape and plain layout
        for (size_t idx = 0; idx < inputs.size(); ++idx) {
          auto ori_lt = inputs[idx];
          std::cout << "input " << ori_lt.get_id() << ": ";
          for (auto s : eid2shape_[ori_lt.get_id()]) {
            std::cout << s << ", ";
          }
          std::cout << std::endl;
          inputs[idx] = logical_tensor{ori_lt.get_id(), ori_lt.get_data_type(),
                                       eid2shape_[ori_lt.get_id()], layout_type::strided};
          eid2desc_[ori_lt.get_id()] = inputs[idx];
        }

        // update output logical tensors with unknown shape and plain layout
        for (size_t idx = 0; idx < outputs.size(); ++idx) {
          auto ori_lt = outputs[idx];
          std::cout << "output " << ori_lt.get_id() << ": ";
          for (auto s : eid2shape_[ori_lt.get_id()]) {
            std::cout << s << ", ";
          }
          std::cout << std::endl;
          outputs[idx] = logical_tensor{ori_lt.get_id(), ori_lt.get_data_type(),
                                        eid2shape_[ori_lt.get_id()], layout_type::strided};
          eid2desc_[ori_lt.get_id()] = outputs[idx];
        }

        /// Compile the partition to generate compiled partition with the
        /// input and output logical tensors.
        //[Compile partition]
        compiled_partition cp = partition.compile(inputs, outputs, eng_);
        cps_.push_back(cp);
        //[Compile partition]

        // // update output logical tensors with inferred shape
        // for (size_t idx = 0; idx < outputs.size(); ++idx) {
        //   size_t id = outputs[idx].get_id();
        //   outputs[idx] = cp.query_logical_tensor(id);
        //   eid2desc_[idx] = outputs[idx];
        // }

        std::vector<tensor> in_trs, out_trs;
        in_trs.reserve(inputs.size());
        out_trs.reserve(outputs.size());
        for (int i = 0; i < inputs.size(); ++i) {
          size_t eid = inputs[i].get_id();
          if (eid2tr_.count(eid)) {
            in_trs.push_back(eid2tr_[eid]);
            continue;
          }
          auto mem_size = eid2desc_[eid].get_mem_size();
          data_buffers_.push_back({});
          data_buffers_.back().reset(malloc(mem_size), cpu_deletor{});
          tensor tr{inputs[i], eng_, data_buffers_.back().get()};
          in_trs.push_back(tr);
          eid2tr_[eid] = tr;
        }
        for (int i = 0; i < outputs.size(); ++i) {
          size_t eid = outputs[i].get_id();
          if (eid2tr_.count(eid)) {
            out_trs.push_back(eid2tr_[eid]);
            continue;
          }
          auto mem_size = eid2desc_[eid].get_mem_size();
          data_buffers_.push_back({});
          data_buffers_.back().reset(malloc(mem_size), cpu_deletor{});
          tensor tr{outputs[i], eng_, data_buffers_.back().get()};
          out_trs.push_back(tr);
          eid2tr_[eid] = tr;
        }

        input_args_.insert(std::make_pair(pi, in_trs));
        output_args_.insert(std::make_pair(pi, out_trs));
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
                std::enable_if_t<std::is_same<T, std::vector<typename T::value_type>>::value, int> =
                    0>
      T AttrConvert(std::vector<std::string> val) {
        T res;
        for (const auto& el : val) res.push_back(AttrConvert<typename T::value_type>({el}));
        return res;
      }

      /*!
       * \brief Helper to extract node attribute with ability to specify default value and result
       * type.
       */
      template <typename T>
      const T GetNodeAttr(const json::JSONGraphNode& node, std::string name,
                          std::vector<std::string> def = {}) {
        auto attr = node.HasAttr(name) ? node.GetAttr<std::vector<std::string>>(name) : def;
        return AttrConvert<T>(attr);
      }

      struct cpu_deletor {
        cpu_deletor() = default;
        void operator()(void* ptr) {
          if (ptr) free(ptr);
        }
      };

      void GetPatInputs(const size_t& nid, std::vector<logical_tensor>& pat_inputs) {
        const JSONGraphNode& node = nodes_[nid];

        ICHECK_GT(node.GetInputs().size(), 0);
        for (auto data_entry : node.GetInputs()) {
          std::vector<int64_t> shape = nodes_[data_entry.id_].GetOpShape()[data_entry.index_];
          uint32_t eid = node_row_ptr_[data_entry.id_] + data_entry.index_;
          std::cout << "input: " << eid << std::endl;
          eid2shape_.insert(std::make_pair(eid, shape));
          if (eid2desc_.count(eid)) pat_inputs.push_back(eid2desc_[eid]);
          else {
            logical_tensor desc{eid, data_type::f32, layout_type::undef};
            eid2desc_[eid] = desc;
            pat_inputs.push_back(desc);
          }
        }
      }

      void GetOutput(const size_t& nid, logical_tensor& desc, bool is_last) {
        const JSONGraphNode& node = nodes_[nid];
        if (!is_last) {
          desc = logical_tensor{max_eid_ + tmp_idx_, data_type::f32, layout_type::undef};
          eid2desc_.insert(std::make_pair(max_eid_ + tmp_idx_, desc));
          std::cout << "tmp_idx: " << max_eid_ + tmp_idx_ << std::endl;
          ++tmp_idx_;
          return;
        };  // -1 reserved value for empty input.

        ICHECK_EQ(node.GetNumOutput(), 1);
        auto shape = node.GetOpShape()[0];
        auto eid = node_row_ptr_[nid] + static_cast<uint32_t>(0);
        std::cout << "output eid: " << eid << std::endl;
        eid2shape_.insert(std::make_pair(eid, shape));
        if (eid2desc_.count(eid)) desc = eid2desc_[eid];

        ICHECK(data_entry_[eid] == nullptr);
        desc = logical_tensor{eid, data_type::f32, layout_type::undef};
        eid2desc_[eid] = desc;
      }

      void GetOpInputs(std::vector<logical_tensor> & inputs,
                       std::vector<logical_tensor> & pat_inputs, const size_t& in_num,
                       bool is_first) {
        std::cout << "is_first:" << is_first << std::endl;
        std::cout << "op input: "
                  << ": ";
        std::cout << inputs.size() << ", ";
        std::cout << std::endl;
        std::cout << "pat_inputs: "
                  << ": ";
        for (auto s : pat_inputs) {
          std::cout << s.get_id() << ", ";
        }
        std::cout << std::endl;
        if (!is_first) inputs.push_back(tmp_end_);
        int add_num = in_num - inputs.size();
        inputs.insert(inputs.end(), pat_inputs.begin(), pat_inputs.begin() + add_num);
        pat_inputs.erase(pat_inputs.begin(), pat_inputs.begin() + add_num);
        std::cout << "op input: "<< ": ";
        for (auto s : inputs) {
          std::cout << s.get_id() << ", ";
        }
        std::cout << std::endl;
        std::cout << "pat_inputs: "
                  << ": ";
        for (auto s : pat_inputs) {
          std::cout << s.get_id() << ", ";
        }
        std::cout << std::endl;
      }

      /* The dnnl engine. */
      engine eng_;
      /* The dnnl stream. */
      stream strm_;

      unsigned int graph_op_idx_;
      int max_eid_;
      int tmp_idx_;
      logical_tensor tmp_end_;

      std::vector<std::shared_ptr<void>> data_buffers_;

      std::unordered_map<uint32_t, std::vector<int64_t>> eid2shape_;
      std::unordered_map<uint32_t, logical_tensor> eid2desc_;
      std::unordered_map<uint32_t, tensor> eid2tr_;
      std::unordered_map<size_t, std::vector<logical_tensor>> nid2inputs_;
      std::unordered_map<size_t, logical_tensor> nid2output_;

      std::vector<compiled_partition> cps_;
      std::unordered_map<int, std::vector<tensor>> input_args_;
      std::unordered_map<int, std::vector<tensor>> output_args_;

      // /* The network layers that are represented in dnnl primitives. */
      // /* The memory that is consumed by arguments. */
      // std::vector<std::unordered_map<int, std::vector<logical_tensor>>> partition_args_;
      
      // /* The entry ID to its corresponding output memory. */
      // std::unordered_map<uint32_t, tensor> entry_out_mem_;
      // std::unordered_map<uint32_t, logical_tensor> entry_out_undef_desc_;
      // std::unordered_map<uint32_t, logical_tensor> entry_out_strided_desc_;
      
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