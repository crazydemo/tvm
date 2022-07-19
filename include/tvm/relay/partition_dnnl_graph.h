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
 * \file tvm/relay/dataflow_matcher.h
 * \brief A pattern matcher for matching dataflow properties.
 */
#ifndef TVM_RELAY_DATAFLOW_MATCHER_H_
#define TVM_RELAY_DATAFLOW_MATCHER_H_

#include <tvm/relay/dataflow_pattern.h>
#include <tvm/relay/dataflow_pattern_functor.h>

#include <string>
#include <unordered_map>
#include <utility>

namespace tvm {
namespace relay {

class DFPatternCallback;
/*!
 * \brief Base type of all dataflow pattern callbacks.
 * \sa DFPatternCallback
 */
class DFPatternCallbackNode : public Object {
 public:
  /*! \brief Pattern this callback matches */
  DFPattern pattern;
  /*! \brief Function to call when finding a matched expression */
  PackedFunc function;
  /*! \brief Require InferType to be run before the callback */
  bool require_type;
  /*! \brief Run the callback only once */
  bool rewrite_once;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("pattern", &pattern);
    v->Visit("require_type", &require_type);
    v->Visit("rewrite_once", &rewrite_once);
  }

  static constexpr const char* _type_key = "DFPatternCallbackNode";
  TVM_DECLARE_BASE_OBJECT_INFO(DFPatternCallbackNode, Object);
};

/*!
 * \brief Managed reference to dataflow pattern callbacks.
 * \sa DFPatternCallbackNode
 */
class DFPatternCallback : public ObjectRef {
 public:
  TVM_DLL DFPatternCallback(DFPattern pattern, PackedFunc callback, bool require_type,
                            bool rewrite_once = false);
  TVM_DEFINE_OBJECT_REF_METHODS(DFPatternCallback, ObjectRef, DFPatternCallbackNode);
};

/*!
 * \brief Partition all matches of a DFPattern inside an Expr into separate Function calls
 *
 * \param pattern The pattern to match
 * \param expr The expression to patition
 * \param attrs A set of parameter names and values to apply to the partitioned function
 * \param check A callback function for checking more complicated properties of the matched
 * expressions, returns true if the match is accepted and false otherwise
 *
 * \return Return the paritioned Expr.
 */
Expr PartitionDNNLPattern(Expr expr);

}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_DATAFLOW_MATCHER_H_
