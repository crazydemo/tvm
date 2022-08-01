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
class DNNLPatternPartitioner : protected MixedModeMutator {
 using data_type = logical_tensor::data_type;
 using layout_type = logical_tensor::layout_type;

 std::map<std::string, std::string> op_map{
     {"nn.bias_add", "bias"},
     {"nn.conv2d", "conv2d"},
     {"add", "add"},
     {"nn.relu", "relu"},
 };

public:
 std::vector<std::pair<runtime::String, DFPattern>> Partition(const Expr& pre) {
   std::unique_ptr<IndexedGraph<Expr>> expr_graph = CreateIndexedGraph(pre);
   std::cout << "expr_graph: " << expr_graph->ToString() << std::endl;
   std::cout << expr_graph->size() << std::endl;

   graph dnnl_graph(engine::kind::cpu);
   for (PostDfsIndex index = 0; index < expr_graph->size() - 1;
        ++index) {  // overlook the function pattern
     auto node = expr_graph->index_to_node(index);
     // std::cout<<index<<", "<<node->ref()<<std::endl;
     if (node->inputs_.size() == 0) continue;

     const CallNode* call = node->ref().as<CallNode>();

     // Get input descriptor.
     std::vector<logical_tensor> inputs;
     std::vector<int64_t> dnnl_shape;
     data_type dnnl_dtype = data_type::undef;
     for (PostDfsIndex i = 1; i < node->inputs_.size(); ++i) {
       Expr arg = call->args[i - 1];
       PostDfsIndex idx = node->inputs_[i]->index_;
       dnnl_shape = GetShape(arg->checked_type());
       dnnl_dtype = GetDtype(arg->checked_type());
       logical_tensor input{idx, dnnl_dtype, dnnl_shape, layout_type::undef};
       inputs.push_back(input);
     }

     // Get output descriptor.
     PostDfsIndex next_call_idx = node->outputs_[0]->index_;
     auto next_node = expr_graph->index_to_node(next_call_idx);
     logical_tensor output{index, dnnl_dtype, dnnl_shape.size(), layout_type::undef};
     if (auto next_call = next_node->ref().as<CallNode>()) {
       Expr arg = next_call->args[0];
       dnnl_shape = GetShape(arg->checked_type());
       dnnl_dtype = GetDtype(arg->checked_type());
       output = {index, dnnl_dtype, dnnl_shape, layout_type::undef};
     }

     // Get OP.
     PostDfsIndex op_idx = index;
     if (IsOp(call, "nn.conv2d")) {
       Conv2d(call, op_idx, dnnl_graph, inputs, output);
    //  } else if (IsOp(call, "nn.dense")) {
    //    op dense{
    //        op_idx, op::kind::MatMul, inputs, {output}, "dense" + std::to_string(op_idx)};
    //    dnnl_graph.add_op(dense);
    //  } else if (IsOp(call, "nn.max_pool2d")) {
    //    Maxpool2d(call, op_idx, dnnl_graph, inputs, output);
    //  } else if (IsOp(call, "nn.global_avg_pool2d")) {
    //    GlobalAvgPool2d(call, op_idx, dnnl_graph, inputs, output);
     } else if (IsOp(call, "nn.bias_add")) {
       op bias_add{
           op_idx, op::kind::BiasAdd, inputs, {output}, "bias_add" + std::to_string(op_idx)};
       dnnl_graph.add_op(bias_add);
     } else if (IsOp(call, "nn.relu")) {
       op relu{
           op_idx, op::kind::ReLU, inputs, {output}, "relu" + std::to_string(op_idx)};
       dnnl_graph.add_op(relu);
     } else if (IsOp(call, "add")) {
       op add{op_idx, op::kind::Add, inputs, {output}, "add" + std::to_string(op_idx)};
       dnnl_graph.add_op(add);
     } else {
       op wildcard{
           op_idx, op::kind::Wildcard, inputs, {output}, "wildcard" + std::to_string(op_idx)};
       dnnl_graph.add_op(wildcard);
     }
   }

   // Get partitions.
   std::cout << "dnnl graph compiling ..." << std::endl;
   std::vector<std::pair<runtime::String, DFPattern>> pat_vec;
   auto partitions = dnnl_graph.get_partitions();
   if (partitions.size() < 1)
     LOG_WARNING << "cannot be partitioned by DNNL Graph, will then be build with native codegen";

   /*** Convert DNNL_Graph partition into TVM pattern ***/
   for (size_t i = 0; i < partitions.size(); ++i) {
     std::cout << "Partition[" << partitions[i].get_id()
               << "]'s supporting status: " << (partitions[i].is_supported() ? "true" : "false")
               << "\n";
     auto p = partitions[i];
     if (p.get_ops_num() == 1) {
       size_t op_idx = p.get_ops()[0];
       auto op_node = expr_graph->index_to_node(op_idx);
       auto op = op_node->ref().as<CallNode>()->op;
       auto op_name = op.as<OpNode>()->name;
       if (!op_map.count(op_name)) {
         std::cout << op_name << "will run on native codegen" <<std::endl;
         continue;
       }
     }
     std::unordered_map<int, DFPattern> pat_map;
     for (auto i_desc : p.get_in_ports()) {
       auto pi_idx = i_desc.get_id();
       auto pi_pat = IsWildcard();
       pat_map.insert(std::make_pair(pi_idx, pi_pat));
     }

     ICHECK_GE(p.get_ops().size(), 1);
     DFPattern pat;
     runtime::String pat_name = "dnnl.";
     for (auto op_idx : p.get_ops()) {
       std::cout << "op_idx:" << op_idx << std::endl;
       auto op_node = expr_graph->index_to_node(op_idx);
       ICHECK(op_node->ref().as<CallNode>());
       ICHECK_GE(op_node->inputs_.size(), 2);
       std::vector<DFPattern> input_pats;
       for (size_t j = 1; j < op_node->inputs_.size(); ++j) {
         auto pi_idx = op_node->inputs_[j]->index_;
         ICHECK(pat_map.count(pi_idx));
         input_pats.push_back(pat_map[pi_idx]);
       }
       auto op = op_node->ref().as<CallNode>()->op;
       auto op_name = op.as<OpNode>()->name;
       pat_name = pat_name + op_map[op_name] + "_";
       pat = relay::IsOp(op_name)(input_pats);
       pat_map.insert(std::make_pair(op_idx, pat));
     }
     std::cout << "pat: " << pat << std::endl;
     pat_vec.push_back(std::make_pair(pat_name, pat));
   }
   return pat_vec;
  }

 protected:
  
  void Conv2d(const CallNode* call, PostDfsIndex idx, graph& g, std::vector<logical_tensor> inputs, logical_tensor output) {
    const auto* conv2d_attr = call->attrs.as<Conv2DAttrs>();
    ICHECK(conv2d_attr);

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

  void Maxpool2d(const CallNode* call, PostDfsIndex idx, graph& g, std::vector<logical_tensor> inputs,
              logical_tensor output) {
    const auto* attr = call->attrs.as<MaxPool2DAttrs>();
    ICHECK(attr);
    std::vector<int64_t> pool_size = ShapeToInt(attr->pool_size);
    std::vector<int64_t> strides = ShapeToInt(attr->strides);
    std::vector<int64_t> padding = ShapeToInt(attr->padding);
    std::vector<int64_t> padding_l(padding.begin(), padding.begin() + padding.size() / 2);
    std::vector<int64_t> padding_r(padding.begin() + padding.size() / 2, padding.end());
    std::vector<int64_t> dilation = ShapeToInt(attr->dilation);
    std::string data_layout = attr->layout;
    std::string data_format = regex_replace(data_layout, regex("(D?)(H?)W"), "X");
    std::string rounding_type = attr->ceil_mode ? "ceil" : "floor";

    op maxpool{idx, op::kind::MaxPool, inputs, {output}, "maxpool" + std::to_string(idx)};
    maxpool.set_attr<std::vector<int64_t>>("kernel", pool_size);
    maxpool.set_attr<std::vector<int64_t>>("strides", strides);
    maxpool.set_attr<std::vector<int64_t>>("pads_begin", padding_l);
    maxpool.set_attr<std::vector<int64_t>>("pads_end", padding_r);
    maxpool.set_attr<std::vector<int64_t>>("dilations", dilation);
    maxpool.set_attr<std::string>("data_format", data_format);
    maxpool.set_attr<std::string>("rounding_type", rounding_type);

    /// add the ops to the graph
    g.add_op(maxpool);
  }

  void GlobalAvgPool2d(const CallNode* call, PostDfsIndex idx, graph& g,
                 std::vector<logical_tensor> inputs, logical_tensor output) {
    const auto* attr = call->attrs.as<GlobalPool2DAttrs>();
    ICHECK(attr);

    std::vector<int64_t> pool_size;
    for (size_t i = 2; i <inputs[0].get_dims().size(); ++i) {
      pool_size.push_back(inputs[0].get_dims()[i]);
      std::cout << "dim: " << inputs[0].get_dims()[i] << std::endl;
    }
    std::string data_layout = attr->layout;
    std::string data_format = regex_replace(data_layout, regex("(D?)(H?)W"), "X");
    std::vector<int64_t> strides = std::vector<int64_t>(inputs[0].get_dims().size() - 2, 1);
    std::vector<int64_t> padding = std::vector<int64_t>((inputs[0].get_dims().size() - 2) * 2, 0);
    std::vector<int64_t> padding_l(padding.begin(), padding.begin() + padding.size() / 2);
    std::vector<int64_t> padding_r(padding.begin() + padding.size() / 2, padding.end());
    std::vector<int64_t> dilation = std::vector<int64_t>(inputs[0].get_dims().size() - 2, 1);

    op pool{idx, op::kind::AvgPool, inputs, {output}, "global_avgpool" + std::to_string(idx)};
    pool.set_attr<std::vector<int64_t>>("kernel", pool_size);
    pool.set_attr<std::vector<int64_t>>("strides", strides);
    pool.set_attr<std::vector<int64_t>>("pads_begin", padding_l);
    pool.set_attr<std::vector<int64_t>>("pads_end", padding_r);
    pool.set_attr<std::vector<int64_t>>("dilations", dilation);
    pool.set_attr<std::string>("data_format", data_format);

    /// add the ops to the graph
    g.add_op(pool);
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

std::vector<std::pair<runtime::String, DFPattern>> PartitionDNNLPattern(Expr expr) {
  return DNNLPatternPartitioner().Partition(expr);
}

}  // namespace relay
}  // namespace tvm
