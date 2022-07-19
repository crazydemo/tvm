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

#include <stack>

namespace tvm {
namespace relay {

/*!
 * \brief PatternPartitioner replaces expressions that match a pattern with function call that
 * perform the same computation but allow for further analysis and lowering.
 *
 * The class uses PatternGrouper to support the dominator pattern.
 */
class PatternPartitioner : protected MixedModeMutator {
 public:
  Expr Partition(const Expr& pre) {
    std::cout<<"pre: "<<pre<<std::endl;
    return pre;
  }

 protected:
  
};

Expr PartitionDNNLPattern(Expr expr) {
  return PatternPartitioner().Partition(expr);
}

TVM_REGISTER_GLOBAL("relay.dataflow_pattern.partition_dnnl_pattern")
    .set_body_typed([](Expr expr) { return PartitionDNNLPattern(expr); });

}  // namespace relay
}  // namespace tvm
