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
 * \file src/tvm/relay/dataflow_matcher.cc
 * \brief The dataflow pattern matcher for Relay.
 */

#include <tvm/relay/analysis.h>
#include <tvm/relay/partition_dnnl_graph.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/relay/attrs/nn.h>

#include <stack>
#include <regex>

#include "indexed_graph.h"
#include "oneapi/dnnl/dnnl_graph.hpp"
#include "dnnl.hpp"

namespace tvm {
namespace relay {

using namespace dnnl::graph;
using std::regex_replace;
using std::regex;
/*!
 * \brief PatternPartitioner replaces expressions that match a pattern with function call that
 * perform the same computation but allow for further analysis and lowering.
 *
 * The class uses PatternGrouper to support the dominator pattern.
 */
class PatternPartitioner : protected MixedModeMutator {
 using data_type = logical_tensor::data_type;
 using layout_type = logical_tensor::layout_type;
 public:
  Expr Partition(const Expr& pre) {
    std::unique_ptr<IndexedGraph<Expr>> expr_graph = CreateIndexedGraph(pre);
    std::cout<<"expr_graph: "<<expr_graph->ToString()<<std::endl;
    std::cout<<expr_graph->size()<<std::endl;

    graph dnnl_graph(engine::kind::cpu);
    for (PostDfsIndex index = 0; index < expr_graph->size() - 1; ++index) { // overlook the function pattern
      auto node = expr_graph->index_to_node(index);
      // std::cout<<index<<", "<<node->ref()<<std::endl;
      if (node->inputs_.size() == 0) continue;

      const CallNode* call = node->ref().as<CallNode>();

      // Get input descriptor.
      std::vector<logical_tensor> inputs;
      std::vector<int64_t> dnnl_shape;
      data_type dnnl_dtype = data_type::undef;
      for (PostDfsIndex i = 1; i < node->inputs_.size(); ++i) {
        Expr arg = call->args[i-1];
        PostDfsIndex idx = node->inputs_[i]->index_;
        dnnl_shape = GetShape(arg->checked_type());
        dnnl_dtype = GetDtype(arg->checked_type());
        logical_tensor input {idx, dnnl_dtype, dnnl_shape, layout_type::undef};
        inputs.push_back(input);
      }
      
      // Get output descriptor.
      PostDfsIndex next_call_idx = node->outputs_[0]->index_;
      auto next_node = expr_graph->index_to_node(next_call_idx);
      logical_tensor output {index, dnnl_dtype, dnnl_shape.size(), layout_type::undef};
      if (auto next_call = next_node->ref().as<CallNode>()) {
        Expr arg = next_call->args[0];
        dnnl_shape = GetShape(arg->checked_type());
        dnnl_dtype = GetDtype(arg->checked_type());
        output = {index, dnnl_dtype, dnnl_shape, layout_type::undef};
      }
      
      //Get OP.
      PostDfsIndex op_idx = node->inputs_[0]->index_;
      if (IsOp(call, "nn.conv2d")) {
        Conv2d(call, op_idx, dnnl_graph, inputs, output);
      } else if (IsOp(call, "nn.bias_add")) {
        BiasAdd(call, op_idx, dnnl_graph, inputs, output);
      } else {
        LOG_FATAL << "Unsupported DNNL Graph Op";
      }
    }

    // Get partitions.
    std::cout<<"dnnl graph compiling ..."<<std::endl;
    auto partitions = dnnl_graph.get_partitions();
    if (partitions.size() < 1)
        throw std::runtime_error(
                "cpu_simple_pattern_f32: incorrect partition number");
    std::cout << "Number of returned partitions: " << partitions.size() << "\n";
    for (size_t i = 0; i < partitions.size(); ++i) {
        std::cout << "Partition[" << partitions[i].get_id()
                  << "]'s supporting status: "
                  << (partitions[i].is_supported() ? "true" : "false") << "\n";
    }
    return pre;
  }

 protected:
  
  void Conv2d(const CallNode* call, PostDfsIndex idx, graph& g, std::vector<logical_tensor> inputs, logical_tensor output) {
    const auto* conv2d_attr = call->attrs.as<Conv2DAttrs>();
    ICHECK(conv2d_attr);

    int groups = conv2d_attr->groups;
    std::vector<int64_t> strides = ShapeToInt(conv2d_attr->strides);
    std::vector<int64_t> padding = ShapeToInt(conv2d_attr->padding);
    std::vector<int64_t> padding_l(padding.begin(), padding.begin() + padding.size() / 2);
    std::vector<int64_t> padding_r(padding.begin() + padding.size() / 2, padding.end());
    std::vector<int64_t> dilation = ShapeToInt(conv2d_attr->dilation);
    std::string data_layout = conv2d_attr->data_layout;
    std::string kernel_layout = conv2d_attr->kernel_layout;
    std::string data_format = regex_replace(data_layout, regex("(D?)(H?)W"), "X");
    std::string filter_format = regex_replace(kernel_layout, regex("(D?)(H?)W"), "X");

    op conv {idx, op::kind::Convolution, inputs,
            {output}, "conv" + std::to_string(idx)};
    conv.set_attr<std::vector<int64_t>>("strides", strides);
    conv.set_attr<std::vector<int64_t>>("pads_begin", padding_l);
    conv.set_attr<std::vector<int64_t>>("pads_end", padding_r);
    conv.set_attr<std::vector<int64_t>>("dilations", dilation);
    conv.set_attr<std::string>("data_format", data_format);  
    conv.set_attr<std::string>("filter_format", filter_format);
    conv.set_attr<int64_t>("groups", conv2d_attr->groups);

    /// add the ops to the graph
    g.add_op(conv);
  }

  void BiasAdd(const CallNode* call, PostDfsIndex idx, graph& g, std::vector<logical_tensor> inputs, logical_tensor output) {
    op bias_add {idx, op::kind::BiasAdd, inputs, {output}, "bias_add" + std::to_string(idx)};
    /// add the ops to the graph
    g.add_op(bias_add);
  }

  /*!
  * \brief Extract the shape from a Relay tensor type.
  * \param type The provided type.
  * \return The extracted shape in a list.
  */
  inline std::vector<int64_t> GetShape(const Type& type) {
    const auto* ttype = type.as<TensorTypeNode>();
    ICHECK(ttype) << "Expect TensorTypeNode";
    std::vector<int64_t> shape;
    for (size_t i = 0; i < ttype->shape.size(); ++i) {
      auto* val = ttype->shape[i].as<IntImmNode>();
      ICHECK(val);
      shape.push_back(val->value);
    }
    return shape;
  }

  inline std::vector<int64_t> ShapeToInt(Array<IndexExpr> raw_shape) {
    std::vector<int64_t> shape;
    for (size_t i = 0; i < raw_shape.size(); ++i) {
      auto* val = raw_shape[i].as<IntImmNode>();
      ICHECK(val);
      shape.push_back(val->value);
    }
    return shape;
  }

  data_type GetDtype(const Type& type) {
    const auto* ttype = type.as<TensorTypeNode>();
    ICHECK(ttype) << "Expect TensorTypeNode";
    auto dltype = ttype->dtype;
    data_type dgraph_type = data_type::undef;
    if (dltype.code() == DataType::TypeCode::kFloat) {
      if (dltype.bits() == 16) {
        dgraph_type = data_type::f16;
      } else if (dltype.bits() == 32) {
        dgraph_type = data_type::f32;
      }
    } else if (dltype.code() == DataType::TypeCode::kBFloat && dltype.bits() == 16) {
      dgraph_type = data_type::bf16;
    } else if (dltype.code() == DataType::TypeCode::kInt) {
      if (dltype.bits() == 8) {
        dgraph_type = data_type::s8;
      } else if (dltype.bits() == 32) {
        dgraph_type = data_type::s32;
      }
    } else if (dltype.code() == DataType::TypeCode::kUInt && dltype.bits() == 8) {
      dgraph_type = data_type::u8;
    }
    if (dgraph_type == data_type::undef) {
      LOG_ERROR << "unsupported datatype: code=" << dltype.code() << ", bits=" << dltype.bits();
    }
    return dgraph_type;
  }
  
  /*!
  * \brief Check if a call has the provided name.
  * \param call A Relay call node.
  * \param op_name The name of the expected call.
  * \return true if the call's name is equivalent to the given name. Otherwise,
  * false.
  */
  inline bool IsOp(const CallNode* call, const std::string& op_name) {
    const auto* op_node = call->op.as<OpNode>();
    ICHECK(op_node) << "Expects a single op.";
    Op op = GetRef<Op>(op_node);
    return op == Op::Get(op_name);
  }
};

Expr PartitionDNNLPattern(Expr expr) {
  return PatternPartitioner().Partition(expr);
}

TVM_REGISTER_GLOBAL("relay.dataflow_pattern.partition_dnnl_pattern")
    .set_body_typed([](Expr expr) { return PartitionDNNLPattern(expr); });

}  // namespace relay
}  // namespace tvm
