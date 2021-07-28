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
 * \file alter_op_layout.cc
 * \brief Alternate the layouts of operators or replace primitive operators with
          other expressions. This pass can be used for computing convolution in
          custom layouts or other general weight pre-transformation.
 */
#include <tvm/relay/analysis.h>
#include <tvm/relay/attrs/transform.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/transform.h>
#include <tvm/te/operation.h>

#include <functional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pattern_utils.h"
#include "transform_layout.h"

namespace tvm {
namespace relay {

namespace alter_func_layout {

/*!
 * \brief Container to instantiate a Node for alter op layouts.
 */
class AlterFuncTransformMemorizerNode : public TransformMemorizerNode {
 public:
  static constexpr const char* _type_key = "relay.alter_func_layout.AlterFuncTransformMemorizerNode";
};

/*!
 * \brief Container that provides the transformation function for alter layout..
 */
class AlterFuncTransformMemorizer : public TransformMemorizer {
 public:
  AlterFuncTransformMemorizer() {}
  explicit AlterFuncTransformMemorizer(ObjectPtr<Object> n) : TransformMemorizer(n) {}

  AlterFuncTransformMemorizerNode* operator->() {

    std::cout<<"AlterFuncLayout: class AlterTransformMemorizer : public TransformMemorizer"<<std::endl;
    return static_cast<AlterFuncTransformMemorizerNode*>(get_mutable());
  }

  /*!
   * \brief Defines the call transformation for AlterOpLayout pass. The new layouts are defined by
   * used for different targets using a packed func.
   * \param ref_call The original call.
   * \param new_attrs Updated attributes consistent with new layouts.
   * \param new_args The traversed/recursed args to the call.
   * \return The new Call after calling the packed func.
   */
  Call CallWithNewLayouts(const Call& ref_call, Attrs new_attrs,
                          const std::vector<Expr>& new_args) override {
    static auto falter_layout = Op::GetAttrMap<FTVMAlterFuncLayout>("FTVMAlterFuncLayout");
    Op op = Downcast<Op>(ref_call->op);

    std::cout<<"AlterFuncLayout: Call CallWithNewLayouts(const Call& ref_call"<<std::endl;

    Expr new_e;
    bool modified = false;
    if (falter_layout.count(op)) {
      tvm::Array<tvm::te::Tensor> tinfos;
      for (auto expr : ref_call->args) {
        auto ttype = expr->type_as<TensorTypeNode>();
        tinfos.push_back(tvm::te::placeholder(ttype->shape, ttype->dtype));
      }
      // TODO(@kevinthesun, @icemelon9): This won't work if inputs/outputs are dynamic shapes.
      //   Probably we need to disable the AlterOpLayout when compiling dynamic models.
      Expr altered_value = falter_layout[op](new_attrs, new_args, tinfos, ref_call->checked_type());
      if (altered_value.defined()) {
        new_e = altered_value;
        modified = true;
      }
    }
    if (!modified) {
      new_e = Call(ref_call->op, new_args, new_attrs);
    }

    const CallNode* new_call = new_e.as<CallNode>();
    ICHECK(new_call) << "Can only replace the original operator with another call node";
    return GetRef<Call>(new_call);
  }

  using TransformMemorizer::CallWithNewLayouts;
  using ContainerType = AlterFuncTransformMemorizerNode;
};

/*!
 * Limitations:
 * 1. The altered op should have the same number of arguments as the previous one.
 * 2. Do not support nested tuple arguments.
 */
Expr AlterFuncLayout(const Expr& expr) {
  // TODO(@icemelon9): need to rerun type inference after applying an alter op.
  AlterFuncTransformMemorizer alterMemorizer(make_object<AlterFuncTransformMemorizer>());
  auto fcontext = [&](const Call& call) -> ObjectRef { return alterMemorizer; };
  std::cout<<"Expr AlterFuncLayout(const Expr& expr)"<<std::endl;
  return ForwardRewrite(expr, LayoutRewriter<AlterFuncTransformMemorizer>, fcontext);
}

}  // namespace alter_op_layout

namespace transform {

Pass AlterFuncLayout() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        return Downcast<Function>(relay::alter_func_layout::AlterFuncLayout(f));
      };
  std::cout<<"Pass AlterFuncLayout()"<<std::endl;
  return CreateFunctionPass(pass_func, 3, "AlterFuncLayout", {});
}

TVM_REGISTER_GLOBAL("relay._transform.AlterFuncLayout").set_body_typed(AlterFuncLayout);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
