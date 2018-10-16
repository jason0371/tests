import mxnet as mx
from model_squeezenet import SqueezeNet
import joblib
import numpy as np

varkeys_mxnet = ['squeezenet0_conv0_weight',
'squeezenet0_conv0_bias',


'squeezenet0_conv2_weight',
'squeezenet0_conv2_bias',
'squeezenet0_conv3_weight',
'squeezenet0_conv3_bias',
'squeezenet0_conv1_weight',
'squeezenet0_conv1_bias',


'squeezenet0_conv5_weight',
'squeezenet0_conv5_bias',
'squeezenet0_conv6_weight',
'squeezenet0_conv6_bias',
'squeezenet0_conv4_weight',
'squeezenet0_conv4_bias',

'squeezenet0_conv8_weight',
'squeezenet0_conv8_bias',
'squeezenet0_conv9_weight',
'squeezenet0_conv9_bias',
'squeezenet0_conv7_weight',
'squeezenet0_conv7_bias',

'squeezenet0_conv11_weight',
'squeezenet0_conv11_bias',
'squeezenet0_conv12_weight',
'squeezenet0_conv12_bias',
'squeezenet0_conv10_weight',
'squeezenet0_conv10_bias',

'squeezenet0_conv14_weight',
'squeezenet0_conv14_bias',
'squeezenet0_conv15_weight',
'squeezenet0_conv15_bias',
'squeezenet0_conv13_weight',
'squeezenet0_conv13_bias',

'squeezenet0_conv17_weight',
'squeezenet0_conv17_bias',
'squeezenet0_conv18_weight',
'squeezenet0_conv18_bias',
'squeezenet0_conv16_weight',
'squeezenet0_conv16_bias',

'squeezenet0_conv20_weight',
'squeezenet0_conv20_bias',
'squeezenet0_conv21_weight',
'squeezenet0_conv21_bias',
'squeezenet0_conv19_weight',
'squeezenet0_conv19_bias',

'squeezenet0_conv23_weight',
'squeezenet0_conv23_bias',
'squeezenet0_conv24_weight',
'squeezenet0_conv24_bias',
'squeezenet0_conv22_weight',
'squeezenet0_conv22_bias',

'squeezenet0_conv25_weight',
'squeezenet0_conv25_bias']

varkeys = ['fire2/expand1x1', 'fire5/squeeze1x1', 'fire8/squeeze1x1', 'fire3/expand3x3', 'fire7/expand1x1',
           'fire5/expand3x3', 'fire4/expand1x1', 'fire6/expand1x1', 'fire6/expand3x3', 'fire3/expand1x1',
           'fire10/squeeze1x1', 'fire10/expand1x1', 'fire9/expand1x1', 'conv1', 'fire11/expand3x3', 'fire6/squeeze1x1',
           'fire7/squeeze1x1', 'fire9/expand3x3', 'conv12', 'fire8/expand1x1', 'fire11/expand1x1', 'fire11/squeeze1x1',
           'fire2/expand3x3', 'fire2/squeeze1x1', 'fire4/squeeze1x1', 'fire7/expand3x3', 'fire5/expand1x1',
           'fire10/expand3x3', 'fire3/squeeze1x1', 'fire8/expand3x3', 'fire9/squeeze1x1', 'fire4/expand3x3']


c_weight = joblib.load('/home/siiva/RUDY/SIIVA/GoalCam/squeezeDet/data/SqueezeNet/squeezenet_v1.1.pkl')

model = SqueezeNet(pretrained=True)
model.load_params('tinyimagenet_squeezenet_pretrained.params')


x = mx.sym.var('data')

y = model(x)
print(y)

model.hybridize()

arg_names = set(y.list_arguments())
aux_names = set(y.list_auxiliary_states())
arg_dict = {}

mxnet_names = []
for name, param in model.collect_params().items():
    print(name, param)
    mxnet_names.append(name)
    if name in arg_names:

        arg_dict['arg:%s' % name] = param._reduce()
    else:
        assert name in aux_names
        arg_dict['aux:%s' % name] = param._reduce()

varkeys1 = []
print(c_weight.keys())
for key_ in sorted(c_weight.keys()):
    #print(key_, np.array(c_weight[key_]).shape)
    if key_ in varkeys:
        print(key_)
        varkeys1.append(key_)
        # print(np.array(c_weight[key_][0]).shape)
        # print(np.array(c_weight[key_][1]).shape)

print(varkeys1)
print(mxnet_names)

varkeys_caffe = varkeys1

new_varkeys_caffe = {}
idx = 0
for varkey_mxnet in (varkeys_mxnet):

    varkey_mxnet_w = varkey_mxnet.split('_')
    if varkey_mxnet_w[1] == 'conv25':
        break
    if varkey_mxnet_w[2] == 'weight':
        print(varkey_mxnet)
        print(varkeys_caffe[idx])
        print(np.array(c_weight[varkeys_caffe[idx]][0]).shape)
        print(np.array(arg_dict['arg:%s' % varkey_mxnet].asnumpy()).shape)

        if varkeys_caffe[idx] in new_varkeys_caffe:
            new_varkeys_caffe[varkeys_caffe[idx]].append(arg_dict['arg:%s' % varkey_mxnet].asnumpy())
        else:
            new_varkeys_caffe[varkeys_caffe[idx]] = [arg_dict['arg:%s' % varkey_mxnet].asnumpy()]

    if varkey_mxnet_w[2] == 'bias':
        print(varkey_mxnet)
        print(varkeys_caffe[idx])
        print(np.array(c_weight[varkeys_caffe[idx]][1]).shape)
        print(np.array(arg_dict['arg:%s' % varkey_mxnet].asnumpy()).shape)

        if varkeys_caffe[idx] in new_varkeys_caffe:
            new_varkeys_caffe[varkeys_caffe[idx]].append(arg_dict['arg:%s' % varkey_mxnet].asnumpy())
        else:
            new_varkeys_caffe[varkeys_caffe[idx]] = [arg_dict['arg:%s' % varkey_mxnet].asnumpy()]
        idx += 1

        #assert (np.array(c_weight[varkeys_caffe[idx]][0]).shape) == np.array(arg_dict['arg:%s' % varkey_mxnet]).shape

joblib.dump(new_varkeys_caffe, '/home/siiva/RUDY/SIIVA/GoalCam/squeezeDet/data/SqueezeNet/half_squeezenet_v1.1.pkl')

c_weight = joblib.load('/home/siiva/RUDY/SIIVA/GoalCam/squeezeDet/data/SqueezeNet/half_squeezenet_v1.1.pkl')
print(c_weight)