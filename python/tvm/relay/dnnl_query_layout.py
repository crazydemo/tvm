# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=no-else-return, invalid-name, unused-import
"""The layout auto-query func for dnnl."""

from . import _ffi_api
from tvm import relay

def AutoQuery(N,IC,KH,KW,OC,SH,SW,PH_L,PH_R,PW_L,PW_R,OH,OW,G):
    """Get absolute value of the input element-wise.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    return _ffi_api.AutoQuery(N,IC,KH,KW,OC,SH,SW,PH_L,PH_R,PW_L,PW_R,OH,OW,G)  # type: ignore


data_dic = {"a":"N",
            "b":"C",
            "c":"H",
            "d":"W",}

weight_dic = {"a":"O",
            "b":"I",
            "c":"H",
            "d":"W",
            "e":"G"}

def transfer_to_dnnl_layout(level=0):
    # set level = 114 to enable dnnl optimal layout query
    if level == 0:
        return
    @relay.op.register_alter_op_layout("nn.conv2d", level=level)
    def alter_conv2d(attrs, inputs, tinfos, out_type):
        data, weight = inputs
        def get_shape(tensor):
            if 'Var' in str(type(tensor)):
                return tensor.type_annotation.concrete_shape
            elif 'Constant' in str(type(tensor)):
                return tensor.data.shape
            elif 'TensorType' in str(type(tensor)):
                return tensor.concrete_shape
            else:
                if "pad" in tensor.op.name:
                    return tensor.type_args[0].concrete_shape
                return (-1, -1, -1, -1)

        N, IC, IH, IW = get_shape(data)
        OC, IC, KH, KW = get_shape(weight)
        N, OC, OH, OW = get_shape(out_type)
        PH_L, PW_L, PH_R, PW_R = attrs.get_int_tuple("padding")
        SH, SW = attrs.get_int_tuple("strides")
        dilation = attrs.get_int_tuple("dilation")
        G = int(attrs.groups)
        new_attrs = dict(attrs)

        if G>1: # for mobilenet
            IC = IC * G
            new_attrs['data_layout'] = "NCHW"
            new_attrs['kernel_layout'] = "OIHW"
            new_attrs['out_layout'] = "NCHW"
            return relay.nn.conv2d(data, weight, **new_attrs        )

        res = relay.dnnl_query_layout.AutoQuery(N,IC,KH,KW,OC,SH,SW,PH_L,PH_R,PW_L,PW_R,OH,OW,G)

        src_df, weight_df, dst_df = res.split(',')

        def trans_data(input_data, is_weight=False):
            dic = data_dic
            res = input_data
            if is_weight:
                dic = weight_dic
                    
            for key, value in dic.items():
                if key.upper() in input_data:
                    res = res.replace(key.upper(), value, 1)
                    res = res.replace(key, value.lower(), 1)
                else:
                    res = res.replace(key, value, 1)
            return res

        new_attrs['data_layout'] = trans_data(src_df, is_weight=False)
        new_attrs['kernel_layout'] = trans_data(weight_df, is_weight=True)
        new_attrs['out_layout'] = trans_data(dst_df, is_weight=False)

        return relay.nn.conv2d(data, weight, **new_attrs)
