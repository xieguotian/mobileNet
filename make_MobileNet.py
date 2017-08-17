from __future__ import print_function
from caffe import layers as L, params as P, to_proto
from caffe.proto import caffe_pb2
import caffe

def hierarchical_conv(bottom,output_num,pad,stride,kernel_size,with_relu=True):
    if kernel_size>1:
        # if stride==2:
        #     concat = L.Concat(bottom,bottom,axis=1)
        #     #bottom=concat
        # else:
        #     concat = bottom
        conv1 = L.ConvolutionDepthwise(bottom,
                             param=[dict(lr_mult=1, decay_mult=0)], convolution_param=dict(kernel_size=kernel_size,num_output=output_num, pad=pad, stride=stride,
                                                                                           bias_term=False, weight_filler=dict(type='msra'),
                                                                                           bias_filler=dict(type='constant'))
                        )

        bn_act = L.BatchNormTorch(conv1, in_place=False,
                                  param=[dict(lr_mult=0, decay_mult=0),
                                         dict(lr_mult=0, decay_mult=0),
                                         dict(lr_mult=0, decay_mult=0),
                                         dict(lr_mult=1, decay_mult=0),
                                         dict(lr_mult=1, decay_mult=0)],
                                  scale_param=dict(bias_term=True, filler=dict(value=1), bias_filler=dict(value=0)))

        relu_act = L.ReLU(bn_act, in_place=True)

    for i in range(1):
        if i==0 and kernel_size==1:
            stride = stride
            relu_act = bottom
        else:
            stride = 1

        engine=4
        conv1 = L.Convolution(relu_act, kernel_size=1,
                                param=[dict(lr_mult=1, decay_mult=1)],
                                num_output=output_num, pad=0, stride=stride, bias_term=False,#  group=output_num / 2,
                                weight_filler=dict(type='msra'), bias_filler=dict(type='constant'),engine=engine)

        bn_act = L.BatchNormTorch(conv1, in_place=False,
                                param=[dict(lr_mult=0, decay_mult=0),
                                    dict(lr_mult=0, decay_mult=0),
                                    dict(lr_mult=0, decay_mult=0),
                                    dict(lr_mult=1, decay_mult=0),
                                    dict(lr_mult=1, decay_mult=0)],
                                scale_param=dict(bias_term=True, filler=dict(value=1), bias_filler=dict(value=0)))
        if with_relu:
            relu_act = L.ReLU(bn_act, in_place=True)
        else:
            relu_act = bn_act
    return relu_act

def hierarchical_basic_block(bottom,output_num,input_num,pad=1,stride=1,kernel_size=3):
    if stride == 2:
        concat = L.Concat(bottom, bottom, axis=1)
    else:
        concat = bottom
    conv1 = hierarchical_conv(concat,output_num,pad,stride,kernel_size)
    conv2 = hierarchical_conv(conv1,output_num,pad,1,kernel_size,with_relu=False)

    if output_num==input_num and stride==1:
        output = L.Eltwise(conv2,bottom)
    else:
        up_conv = hierarchical_conv(concat,output_num,0,stride,1,with_relu=False)
        output = L.Eltwise(conv2,up_conv)
    relu_act = L.ReLU(output, in_place=True)
    return relu_act
def dw_conv_bn_relu(bottom,output_num,pad,stride,kernel_size):
    conv1 = L.ConvolutionDepthwise(bottom,
                                   param=[dict(lr_mult=1, decay_mult=0)],
                                   convolution_param=dict(kernel_size=kernel_size, num_output=output_num, pad=pad,
                                                          stride=stride,
                                                          bias_term=False, weight_filler=dict(type='msra'),
                                                          bias_filler=dict(type='constant'))
                                   )

    bn_act = L.BatchNormTorch(conv1, in_place=False,
                              param=[dict(lr_mult=0, decay_mult=0),
                                     dict(lr_mult=0, decay_mult=0),
                                     dict(lr_mult=0, decay_mult=0),
                                     dict(lr_mult=1, decay_mult=0),
                                     dict(lr_mult=1, decay_mult=0)],
                              scale_param=dict(bias_term=True, filler=dict(value=1), bias_filler=dict(value=0)))

    relu_act = L.ReLU(bn_act, in_place=True)
    return relu_act

def conv_block_bn_relu(bottom,output_num,pad,stride,kernel_size):
    conv1 = L.Convolution(bottom, kernel_size=kernel_size,
                         param=[dict(lr_mult=1, decay_mult=1)],
                    num_output=output_num, pad=pad, stride=stride, bias_term=False, weight_filler=dict(type='msra'), bias_filler=dict(type='constant'))

    bn_act = L.BatchNormTorch(conv1, in_place=False,
                              param=[dict(lr_mult=0, decay_mult=0),
                                     dict(lr_mult=0, decay_mult=0),
                                     dict(lr_mult=0, decay_mult=0),
                                     dict(lr_mult=1, decay_mult=0),
                                     dict(lr_mult=1, decay_mult=0)],
                              scale_param=dict(bias_term=True, filler=dict(value=1), bias_filler=dict(value=0)))

    relu_act = L.ReLU(bn_act, in_place=True)
    return relu_act


def conv_block_bn(bottom,output_num,pad,stride,kernel_size):
    conv1 = L.Convolution(bottom, kernel_size=kernel_size,
                         param=[dict(lr_mult=1, decay_mult=1)],
                    num_output=output_num, pad=pad, stride=stride, bias_term=False, weight_filler=dict(type='msra'), bias_filler=dict(type='constant'))

    bn_act = L.BatchNormTorch(conv1, in_place=False,
                              param=[dict(lr_mult=0, decay_mult=0),
                                     dict(lr_mult=0, decay_mult=0),
                                     dict(lr_mult=0, decay_mult=0),
                                     dict(lr_mult=1, decay_mult=0),
                                     dict(lr_mult=1, decay_mult=0)],
                              scale_param=dict(bias_term=True, filler=dict(value=1), bias_filler=dict(value=0)))
    return bn_act

def basic_block(bottom,output_num,input_num,pad=1,stride=1,kernel_size=3):
    conv1 = conv_block_bn_relu(bottom,output_num,pad,stride,kernel_size)
    conv2 = conv_block_bn(conv1,output_num,pad,1,kernel_size)

    if output_num==input_num and stride==1:
        output = L.Eltwise(conv2,bottom)
    else:
        up_conv = conv_block_bn(bottom,output_num,0,stride,1)
        output = L.Eltwise(conv2,up_conv)
    relu_act = L.ReLU(output, in_place=True)
    return relu_act

def bottle_nect_block(bottom,output_num,input_num,pad=1,stride=1,kernel_size=3):
    conv1_1 = conv_block_bn_relu(bottom,output_num / 4,0,1,1)
    conv1_2 = conv_block_bn_relu(conv1_1,output_num / 4,pad,stride,kernel_size)
    conv1_3 = conv_block_bn(conv1_2,output_num,0,1,1)

    if output_num==input_num and stride==1:
        output = L.Eltwise(conv1_3,bottom)
    else:
        up_conv = conv_block_bn(bottom,output_num,0,stride,1)
        output = L.Eltwise(conv1_3,up_conv)
    relu_act = L.ReLU(output, in_place=True)
    return relu_act


def makeResNet(bottom):
    model = conv_block_bn_relu(bottom,32,1,2,3) #stride = 2
    model = dw_conv_bn_relu(model,32,1,1,3)
    model = conv_block_bn_relu(model, 64, 0, 1, 1)
    model = dw_conv_bn_relu(model, 64, 1, 2, 3) #stride = 2
    model = conv_block_bn_relu(model, 128, 0, 1, 1)
    model = dw_conv_bn_relu(model, 128, 1, 1, 3)
    model = conv_block_bn_relu(model, 128, 0, 1, 1)
    model = dw_conv_bn_relu(model, 128, 1, 2, 3) #stride = 2
    model = conv_block_bn_relu(model, 256, 0, 1, 1)
    model = dw_conv_bn_relu(model, 256, 1, 1, 3)
    model = conv_block_bn_relu(model, 256, 0, 1, 1)
    model = dw_conv_bn_relu(model, 256, 1, 2, 3) #stride = 2
    model = conv_block_bn_relu(model, 512, 0, 1, 1)

    for i in range(5):
        model = dw_conv_bn_relu(model, 512, 1, 1, 3)
        model = conv_block_bn_relu(model, 512, 0, 1, 1)

    model = dw_conv_bn_relu(model, 512, 1, 2, 3) #stride = 2
    model = conv_block_bn_relu(model, 1024, 0, 1, 1)

    model = dw_conv_bn_relu(model, 1024, 1, 1, 3) #stride = 2
    model = conv_block_bn_relu(model, 1024, 0, 1, 1)

    model = L.Pooling(model, pool=P.Pooling.AVE, kernel_size=7, stride=1)  # global_pooling=True)

    model = L.InnerProduct(model, num_output=1000, bias_term=True, weight_filler=dict(type='msra'),
                           bias_filler=dict(type='constant'),
                           param=[dict(lr_mult=1, decay_mult=1),
                                  dict(lr_mult=1, decay_mult=0)])
    return model



def bin_net(data_file,batch_size=64,depth=[4,4,4,4],first_output=64,out_dim=[64,128,256,512],bottle_nect=False):
    data, label = L.Data(source=data_file, backend=P.Data.LMDB, batch_size=batch_size, ntop=2,
                         shuffle=True,key_files="D:/users/v-guoxie/data/train_org_key.txt",
                         transform_param=dict(mirror=True,
                                              force_color=True,
                                              crop_size=224,
                                              mean_value=[104.007,116.669,122.679],
                                              multi_scale_param=dict(
                                                  is_multi_scale= True,
                                                  min_length= 256,
                                                  max_length= 480)))

    model = makeResNet(data)

    loss = L.SoftmaxWithLoss(model, label)
    accuracy = L.Accuracy(model, label)

    return to_proto(loss,accuracy)
                    # all_out[8],all_out[9],
                    # all_out[10],all_out[11],
                    # all_out[12],all_out[13],
                    # all_out[14],all_out[15])


def make_net():

    # with open('DesNet121.prototxt', 'w') as f:
    #     #change the path to your data. If it's not lmdb format, also change first line of densenet() function
    #     print(str(densenet('/home/zl499/caffe/examples/cifar10/cifar10_train_lmdb', batch_size=64)), file=f)
    #with open('ResNet34_share_bin.prototxt', 'w') as f:
    #with open('ResNet34_nl1x1_3.prototxt', 'w') as f:
    with open('MobileNet.prototxt', 'w') as f:
    #with open('ResNet50.prototxt', 'w') as f:
        #change the path to your data. If it's not lmdb format, also change first line of densenet() function
        #print(str(densenet('/home/zl499/caffe/examples/cifar10/cifar10_train_lmdb', batch_size=64,depth=[6,12,36,24], growth_rate=48,first_output=96)), file=f)
        #print(str(bin_net('D:/users/v-guoxie/data/Imagenet_org_train_lmdb')), file=f)
        print(str(bin_net('D:/users/v-guoxie/data/Imagenet_org_train_lmdb')), file=f)
        #print(str(bin_net('D:/users/v-guoxie/data/Imagenet_org_train_lmdb',bottle_nect=True,depth=[9,12,18,9],out_dim=[256,512,1024,2048])), file=f)
            #bin_net('/home/zl499/caffe/examples/cifar10/cifar10_train_lmdb', depth=[4, 4, 4, 4])), file=f)

    # with open('test_densenet.prototxt', 'w') as f:
    #     print(str(densenet('/home/zl499/caffe/examples/cifar10/cifar10_test_lmdb', batch_size=50)), file=f)

def make_solver():
    s = caffe_pb2.SolverParameter()
    s.random_seed = 0xCAFFE

    s.train_net = 'train_densenet.prototxt'
    s.test_net.append('test_densenet.prototxt')
    s.test_interval = 800
    s.test_iter.append(200)

    s.max_iter = 230000
    s.type = 'Nesterov'
    s.display = 1

    s.base_lr = 0.1
    s.momentum = 0.9
    s.weight_decay = 1e-4

    s.lr_policy='multistep'
    s.gamma = 0.1
    s.stepvalue.append(int(0.5 * s.max_iter))
    s.stepvalue.append(int(0.75 * s.max_iter))
    s.solver_mode = caffe_pb2.SolverParameter.GPU

    solver_path = 'solver.prototxt'
    with open(solver_path, 'w') as f:
        f.write(str(s))

if __name__ == '__main__':

    make_net()
    #make_solver()










