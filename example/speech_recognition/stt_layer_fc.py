import mxnet as mx

from stt_layer_batchnorm import batchnorm


def fc(net,
       num_hidden,
       act_type,
       weight=None,
       bias=None,
       no_bias=False):
    # when weight and bias doesn't have specific name
    if weight is None and bias is None:
        net = mx.sym.FullyConnected(data=net, num_hidden=num_hidden, no_bias=no_bias)
    # when weight doesn't have specific name but bias has
    elif weight is None and bias is not None:
        if no_bias:
            net = mx.sym.FullyConnected(data=net, num_hidden=num_hidden, no_bias=no_bias)
        else:
            net = mx.sym.FullyConnected(data=net, num_hidden=num_hidden, bias=bias, no_bias=no_bias)
    # when bias doesn't have specific name but weight has
    elif weight is not None and bias is None:
        net = mx.sym.FullyConnected(data=net, num_hidden=num_hidden, weight=weight, no_bias=no_bias)
    # when weight and bias specific name
    else:
        if no_bias:
            net = mx.sym.FullyConnected(data=net, num_hidden=num_hidden, weight=weight, no_bias=no_bias)
        else:
            net = mx.sym.FullyConnected(data=net, num_hidden=num_hidden, weight=weight, bias=bias, no_bias=no_bias)
    # activation
    net = mx.sym.Activation(data=net, act_type=act_type)
    return net


def sequence_fc(net,
                seq_len,
                num_layer,
                prefix,
                num_hidden_list=[],
                act_type_list=[],
                is_batchnorm=False
                ):
    if num_layer == len(num_hidden_list) == len(act_type_list):
        if num_layer > 0:
            weight_list = []
            bias_list = []

            for layer_index in range(num_layer):
                weight_list.append(mx.sym.Variable(name='%s_sequence_fc%d_weight' % (prefix, layer_index)))
                # if you use batchnorm bias do not have any effect
                if not is_batchnorm:
                    bias_list.append(mx.sym.Variable(name='%s_sequence_fc%d_bias' % (prefix, layer_index)))
            # batch normalization parameters
            gamma_list = []
            beta_list = []
            if is_batchnorm:
                for layer_index in range(num_layer):
                    gamma_list.append(mx.sym.Variable(name='%s_sequence_fc%d_gamma' % (prefix, layer_index)))
                    beta_list.append(mx.sym.Variable(name='%s_sequence_fc%d_beta' % (prefix, layer_index)))
            # batch normalization parameters ends
            if type(net) is mx.symbol.Symbol:
                net = mx.sym.SliceChannel(data=net, num_outputs=seq_len, axis=1, squeeze_axis=1)
            elif type(net) is list:
                for net_index, one_net in enumerate(net):
                    if type(one_net) is not mx.symbol.Symbol:
                        raise Exception('%d th elements of the net should be mx.symbol.Symbol' % net_index)
            else:
                raise Exception('type of net should be whether mx.symbol.Symbol or list of mx.symbol.Symbol')
            hidden_all = []
            for seq_index in range(seq_len):
                for layer_index in range(num_layer):
                    hidden = net[seq_index]
                    if is_batchnorm:
                        hidden = fc(net=hidden,
                                    num_hidden=num_hidden_list[layer_index],
                                    act_type=act_type_list[layer_index],
                                    weight=weight_list[layer_index],
                                    no_bias=is_batchnorm
                                    )
                        # last layer doesn't have batchnorm
                        if layer_index < num_layer - 1:
                            hidden = batchnorm(net=hidden,
                                               gamma=gamma_list[layer_index],
                                               beta=beta_list[layer_index])
                    else:
                        hidden = fc(net=hidden,
                                    num_hidden=num_hidden_list[layer_index],
                                    act_type=act_type_list[layer_index],
                                    weight=weight_list[layer_index],
                                    bias=bias_list[layer_index]
                                    )
                hidden_all.append(hidden)
            net = hidden_all
        return net
    else:
        raise Exception("length doesn't met - num_layer:",
                        num_layer, ",len(num_hidden_list):",
                        len(num_hidden_list),
                        ",len(act_type_list):",
                        len(act_type_list)
                        )
