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
 * \file src/relay/backend/contrib/dnnl/query_layout.cc
 * \brief layout auto-query func.
 */

#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/relay/type.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>

#include <fstream>
#include <numeric>
#include <regex>
#include <sstream>

#include "../../utils.h"
#include "dnnl.hpp"

using dim_t = dnnl_dim_t;
using dims_t = dnnl_dims_t;

namespace tvm {
namespace relay {
namespace contrib {

template <typename T, typename U>
inline void array_set(T* arr, const U& val, size_t size) {
  for (size_t i = 0; i < size; ++i) arr[i] = static_cast<T>(val);
}

template <typename T>
inline void array_copy(T* dst, const T* src, size_t size) {
  for (size_t i = 0; i < size; ++i) dst[i] = src[i];
}

template <typename T>
inline void swap(T& t1, T& t2) {
  T tmp(t1);
  t1 = t2;
  t2 = tmp;
}

template <typename T, typename U, typename F>
inline void simultaneous_sort(T* vals, T* vals_2nd_level, U* keys, size_t size, F comparator) {
  if (size == 0) return;

  for (size_t i = 0; i < size - 1; ++i) {
    bool swapped = false;

    for (size_t j = 0; j < size - i - 1; j++) {
      auto res = comparator(vals[j], vals[j + 1]);
      if (res == 0) res = comparator(vals_2nd_level[j], vals_2nd_level[j + 1]);

      if (res > 0) {
        swap(vals[j], vals[j + 1]);
        swap(vals_2nd_level[j], vals_2nd_level[j + 1]);
        swap(keys[j], keys[j + 1]);
        swapped = true;
      }
    }

    if (swapped == false) break;
  }
}

void compute_blocks(dims_t blocks, const dnnl::memory::desc* md) {
  using format_kind_t = dnnl_format_kind_t;
  const format_kind_t blocked = dnnl_blocked;
  if (!(md->data.format_kind == blocked)) {
    array_set(blocks, 0, md->data.ndims);
    return;
  }
  array_set(blocks, 1, md->data.ndims);
  const auto& bd = md->data.format_desc.blocking;
  for (int iblk = 0; iblk < bd.inner_nblks; ++iblk)
    blocks[bd.inner_idxs[iblk]] *= bd.inner_blks[iblk];
}

inline bool has_runtime_strides(const dnnl::memory::desc* md) {
  using format_kind_t = dnnl_format_kind_t;
  const format_kind_t blocked = dnnl_blocked;
  if (!(md->data.format_kind == blocked)) return false;
  for (int d = 0; d < md->data.ndims; ++d)
    if (md->data.format_desc.blocking.strides[d] == DNNL_RUNTIME_DIM_VAL) return true;
  return false;
}

std::string md2fmt_tag_str(const dnnl::memory::desc* md) {
  const auto& blk = md->data.format_desc.blocking;

  dims_t blocks = {0};
  compute_blocks(blocks, md);

  char dim_chars[DNNL_MAX_NDIMS + 1];

  dims_t ou_blocks = {0};
  array_copy(ou_blocks, md->data.padded_dims, md->data.ndims);

  bool plain = true;
  for (int d = 0; d < md->data.ndims; ++d) {
    dim_chars[d] = (blocks[d] == 1 ? 'a' : 'A') + static_cast<char>(d);
    if (blocks[d] != 1) plain = false;
    ou_blocks[d] /= blocks[d];
  }

  // Can't report meaningful tag for runtime dimensions.
  if (has_runtime_strides(md)) return "*";

  dims_t strides;
  array_copy(strides, blk.strides, md->data.ndims);

  simultaneous_sort(strides, ou_blocks, dim_chars, md->data.ndims,
                    [](dim_t a, dim_t b) { return b - a; });

  dim_chars[md->data.ndims] = '\0';

  std::string s(dim_chars);

  if (!plain) {
    for (int iblk = 0; iblk < blk.inner_nblks; ++iblk) {
      char c = ('a' + static_cast<char>(blk.inner_idxs[iblk]));
      s += (std::to_string(blk.inner_blks[iblk]) + c);
    }
  }
  return s;
}

dnnl::memory::dims str2dims(std::string str_shape, int input_size) {
  std::string str_reg = "(\\d*)";
  for (int i = 0; i < input_size - 1; i++) {
    str_reg.append(",(\\d*)");
  }
  std::regex rex(str_reg);
  std::smatch m;
  dnnl::memory::dims out_dims;
  if (std::regex_search(str_shape, m, rex)) {
    std::transform(m.begin() + 1, m.end(), std::back_inserter(out_dims),
                   [](const std::string& str) { return std::stoi(str); });
  } else {
    LOG(FATAL) << "Unsupported shape for querying optimal dnnl layout: " << str_shape;
  }
  return out_dims;
}

std::string get_optimal_layout_for_conv(int input_size, std::string weight_shape,
                                        std::string out_shape, std::string paddings,
                                        std::string strides, std::string dilates, std::string G) {
  dnnl::engine eng(dnnl::engine::kind::cpu, 0);
  dnnl::stream s(eng);
  using tag = dnnl::memory::format_tag;
  using dt = dnnl::memory::data_type;

  dnnl::memory::dim groups = std::stoi(G);
  dnnl::memory::dims weight_dims_ = str2dims(weight_shape, input_size);
  dnnl::memory::dims weight_dims = weight_dims_;
  if (groups > 1) {
    if (weight_dims_.size() == 5) {
      weight_dims = {weight_dims_[0] * weight_dims_[1], weight_dims_[2], weight_dims_[3],
                     weight_dims_[4]};
    } else {
      weight_dims[1] = weight_dims[1] * groups;
    }
  }
  dnnl::memory::dims out_dims = str2dims(out_shape, input_size);
  dnnl::memory::dims padding_dims = str2dims(paddings, 2 * (input_size - 2));
  dnnl::memory::dims padding_dims_l(padding_dims.begin(),
                                    padding_dims.begin() + padding_dims.size() / 2);
  dnnl::memory::dims padding_dims_r(padding_dims.end() - padding_dims.size() / 2,
                                    padding_dims.end());
  dnnl::memory::dims strides_dims = str2dims(strides, input_size - 2);
  dnnl::memory::dims dilates_dims = str2dims(dilates, input_size - 2);

  dnnl::memory::dims input_dims = out_dims;
  input_dims[1] = weight_dims[1];
  for (int i = 2; i < input_size; i++) {
    dnnl::memory::dim K = weight_dims[i];
    dnnl::memory::dim S = strides_dims[i - 2];
    dnnl::memory::dim D = dilates_dims[i - 2] - 1;
    dnnl::memory::dim PL = padding_dims_l[i - 2];
    dnnl::memory::dim PR = padding_dims_r[i - 2];
    dnnl::memory::dim DK = 1 + (K - 1) * (D + 1);
    dilates_dims[i - 2] = D;
    input_dims[i] = out_dims[i] * S - PL - PR + DK - 1;
  }

  dnnl::memory::dims conv_src_dims = input_dims;
  dnnl::memory::dims conv_weights_dims = weight_dims;
  if (groups > 1) {
    conv_weights_dims = {groups, out_dims[1] / groups, input_dims[1] / groups};
    conv_weights_dims.insert(conv_weights_dims.end(), weight_dims.begin() + 2, weight_dims.end());
  }
  dnnl::memory::dims conv_dst_dims = out_dims;
  dnnl::memory::dims conv_strides = strides_dims;
  dnnl::memory::dims conv_dilates = dilates_dims;
  dnnl::memory::dims conv_padding_l = padding_dims_l;
  dnnl::memory::dims conv_padding_r = padding_dims_r;

  auto conv_src_md = dnnl::memory::desc({conv_src_dims}, dt::f32, tag::any);
  auto conv_weights_md = dnnl::memory::desc({conv_weights_dims}, dt::f32, tag::any);
  auto conv_dst_md = dnnl::memory::desc({conv_dst_dims}, dt::f32, tag::any);

  auto conv_desc = dnnl::convolution_forward::desc(
      dnnl::prop_kind::forward_inference, dnnl::algorithm::convolution_direct, conv_src_md,
      conv_weights_md, conv_dst_md, conv_strides, conv_dilates, conv_padding_l, conv_padding_r);

  auto conv_prim_desc = dnnl::convolution_forward::primitive_desc(conv_desc, eng);

  auto src_format = conv_prim_desc.src_desc();
  auto weights_format = conv_prim_desc.weights_desc();
  auto dst_format = conv_prim_desc.dst_desc();
  std::string src_df, weight_df, dst_df;

  src_df = md2fmt_tag_str(&src_format);
  weight_df = md2fmt_tag_str(&weights_format);
  dst_df = md2fmt_tag_str(&dst_format);
  std::string res = src_df + "," + weight_df + "," + dst_df;
  return res;
}

std::string get_optimal_layout_for_conv_transpose(int input_size, std::string weight_shape,
                                                  std::string out_shape, std::string paddings,
                                                  std::string output_paddings, std::string strides,
                                                  std::string dilates, std::string G) {
  dnnl::engine eng(dnnl::engine::kind::cpu, 0);
  dnnl::stream s(eng);
  using tag = dnnl::memory::format_tag;
  using dt = dnnl::memory::data_type;

  dnnl::memory::dim groups = std::stoi(G);
  dnnl::memory::dims weight_dims_ = str2dims(weight_shape, input_size);
  dnnl::memory::dims weight_dims = weight_dims_;
  if (groups > 1) {
    if (weight_dims_.size() == 5) {
      weight_dims = {weight_dims_[0] * weight_dims_[1], weight_dims_[2], weight_dims_[3],
                     weight_dims_[4]};
    } else {
      weight_dims[1] = weight_dims[1] * groups;
    }
  }
  dnnl::memory::dims out_dims = str2dims(out_shape, input_size);
  dnnl::memory::dims padding_dims = str2dims(paddings, 2 * (input_size - 2));
  dnnl::memory::dims padding_dims_l(padding_dims.begin(),
                                    padding_dims.begin() + padding_dims.size() / 2);
  dnnl::memory::dims padding_dims_r(padding_dims.end() - padding_dims.size() / 2,
                                    padding_dims.end());
  dnnl::memory::dims output_padding_dims = str2dims(output_paddings, input_size - 2);
  dnnl::memory::dims strides_dims = str2dims(strides, input_size - 2);
  dnnl::memory::dims dilates_dims = str2dims(dilates, input_size - 2);

  dnnl::memory::dims input_dims = out_dims;
  if (out_dims[1] == weight_dims[0]) {
    input_dims[1] = weight_dims[1];
  } else {
    input_dims[1] = weight_dims[0];
    std::swap(weight_dims[0], weight_dims[1]);
  }
  for (int i = 2; i < input_size; i++) {
    dnnl::memory::dim K = weight_dims[i];
    dnnl::memory::dim S = strides_dims[i - 2];
    dnnl::memory::dim D = dilates_dims[i - 2] - 1;
    dnnl::memory::dim PL = padding_dims_l[i - 2];
    dnnl::memory::dim PR = padding_dims_r[i - 2];
    dnnl::memory::dim OP = output_padding_dims[i - 2];
    dnnl::memory::dim DK = 1 + (K - 1) * (D + 1);
    dilates_dims[i - 2] = D;
    input_dims[i] = (out_dims[i] - DK + PL + PR - OP) / S + 1;
  }

  dnnl::memory::dims deconv_src_dims = input_dims;
  dnnl::memory::dims deconv_weights_dims = weight_dims;
  if (groups > 1) {
    deconv_weights_dims = {groups, out_dims[1] / groups, input_dims[1] / groups};
    deconv_weights_dims.insert(deconv_weights_dims.end(), weight_dims.begin() + 2,
                               weight_dims.end());
  }
  dnnl::memory::dims deconv_dst_dims = out_dims;
  dnnl::memory::dims deconv_strides = strides_dims;
  dnnl::memory::dims deconv_dilates = dilates_dims;
  dnnl::memory::dims deconv_padding_l = padding_dims_l;
  dnnl::memory::dims deconv_padding_r = padding_dims_r;

  auto deconv_src_md = dnnl::memory::desc({deconv_src_dims}, dt::f32, tag::any);
  auto deconv_weights_md = dnnl::memory::desc({deconv_weights_dims}, dt::f32, tag::any);
  auto deconv_dst_md = dnnl::memory::desc({deconv_dst_dims}, dt::f32, tag::any);

  auto deconv_desc = dnnl::deconvolution_forward::desc(
      dnnl::prop_kind::forward_inference, dnnl::algorithm::deconvolution_direct, deconv_src_md,
      deconv_weights_md, deconv_dst_md, deconv_strides, deconv_dilates, deconv_padding_l,
      deconv_padding_r);

  auto deconv_prim_desc = dnnl::deconvolution_forward::primitive_desc(deconv_desc, eng);

  auto src_format = deconv_prim_desc.src_desc();
  auto weights_format = deconv_prim_desc.weights_desc();
  auto dst_format = deconv_prim_desc.dst_desc();
  std::string src_df, weight_df, dst_df;

  src_df = md2fmt_tag_str(&src_format);
  weight_df = md2fmt_tag_str(&weights_format);
  dst_df = md2fmt_tag_str(&dst_format);
  std::string res = src_df + "," + weight_df + "," + dst_df;
  return res;
}

TVM_REGISTER_GLOBAL("relay.ir.get_optimal_layout_for_conv")
    .set_body([](TVMArgs args, TVMRetValue* rv) {
      *rv = get_optimal_layout_for_conv(args[0], args[1], args[2], args[3], args[4], args[5],
                                        args[6]);
    });

TVM_REGISTER_GLOBAL("relay.ir.get_optimal_layout_for_conv_transpose")
    .set_body([](TVMArgs args, TVMRetValue* rv) {
      *rv = get_optimal_layout_for_conv_transpose(args[0], args[1], args[2], args[3], args[4],
                                                  args[5], args[6], args[7]);
    });

}  // namespace contrib
}  // namespace relay
}  // namespace tvm
