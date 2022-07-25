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
 * \file src/relay/transforms/merge_composite.cc
 * \brief Merges expressions matching patterns into functions marked
 * as 'composite'. This is primarily intended to be used alongside the
 * external codegen infrastructure to support the case where multiple
 * Relay operators map to a single external operator.
 */

#include <tvm/relay/analysis.h>
#include <tvm/relay/partition_dnnl_graph.h>
// #include <tvm/relay/dataflow_matcher.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/transform.h>
#include <tvm/te/operation.h>

namespace tvm {
namespace relay {
namespace partition_via_dnnl_graph {

Function InferType(const Function& expr, const IRModule& m) {
  IRModule mod(m);
  mod->Update(mod->GetGlobalVar("main"), expr);
  mod = transform::InferType()(mod);
  return Downcast<Function>(mod->Lookup("main"));
}

Expr MergeDNNLGraph(const Function& func, const IRModule& m) {
  Function merged_func = func;
  // merge the patterns one-by-one in order
  std::vector<std::pair<std::string, DFPattern>> dnnl_partition = PartitionDNNLPattern(merged_func);
  for (size_t i = 0; i < dnnl_partition.size(); i++) {
    std::cout<<dnnl_partition[i].first<<std::endl;
    std::cout<<dnnl_partition[i].second<<std::endl;
    Map<String, ObjectRef> attrs;
    // attrs.Set("Composite", runtime::String(dnnl_partition[i].first));
    // merged_func = Downcast<Function>(PartitionPattern(dnnl_partition[i].second, merged_func, attrs, {}, true));
    std::cout<<merged_func<<std::endl;
    merged_func = InferType(merged_func, m);
  }
  // attrs.Set("Composite", dnnl_partition[0].first);
  // merged_func = InferType(merged_func, m);
  return std::move(merged_func);
}

}  // namespace merge_composite

namespace transform {

Pass MergeDNNLGraph() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        return Downcast<Function>(
            relay::partition_via_dnnl_graph::MergeDNNLGraph(f, m));
      };
  auto func_pass = CreateFunctionPass(pass_func, 0, "MergeDNNLGraph", {});
  return func_pass;
}

TVM_REGISTER_GLOBAL("relay._transform.MergeDNNLGraph").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = MergeDNNLGraph();
});

}  // namespace transform

}  // namespace relay
}  // namespace tvm
