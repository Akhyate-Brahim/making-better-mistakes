import torch.cuda
import torch.nn as nn
from torchvision import models

import torch
import torch.nn as nn

class PlainNet(nn.Module):
    def __init__(self, output_dim, filters=[64, 64, 'ap', 128, 128, 128, 'ap', 256, 256, 256, 'ap', 512, 'gap', 'fc512'],
                 activation='relu', final_activation=None, input_shape=(None, None, 3), pool_size=(2, 2)):
        super(PlainNet, self).__init__()
        
        layers = []
        self.flatten_indices = []
        in_channels = input_shape[2]
        for i, f in enumerate(filters):
            if f == 'mp':
                layers.append(nn.MaxPool2d(kernel_size=pool_size))
            elif f == 'ap':
                layers.append(nn.AvgPool2d(kernel_size=pool_size))
            elif f == 'gap':
                layers.append(nn.AdaptiveAvgPool2d(1))
                self.flatten_indices.append(len(layers))
            elif isinstance(f, str) and f.startswith('fc'):
                self.flatten_indices.append(len(layers))
                layers.append(nn.Linear(in_channels, int(f[2:])))
                layers.append(nn.BatchNorm1d(int(f[2:])))
                in_channels = int(f[2:])
            else:
                layers.append(nn.Conv2d(in_channels, f, kernel_size=3, padding=1))
                layers.append(nn.BatchNorm2d(f))
                layers.append(nn.ReLU(inplace=True))
                in_channels = f
        
        self.flatten_indices.append(len(layers))
        layers.append(nn.Linear(in_channels, output_dim))
        if final_activation == 'softmax':
            layers.append(nn.Softmax(dim=1))
        
        self.layers = nn.ModuleList(layers)
    
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if i in self.flatten_indices:
                x = x.view(x.size(0), -1)
            x = layer(x)
        return x

def init_model_on_gpu(gpus_per_node, opts):
    arch_dict = models.__dict__
    pretrained = False if not hasattr(opts, "pretrained") else opts.pretrained
    distributed = False if not hasattr(opts, "distributed") else opts.distributed
    print("=> using model '{}', pretrained={}".format(opts.arch, pretrained))

    if opts.arch == "resnet18":
        model = arch_dict[opts.arch](pretrained=pretrained)
        feature_dim = 512
    elif opts.arch == "resnet50":
        model = arch_dict[opts.arch](pretrained=pretrained)
        feature_dim = 2048
    elif opts.arch == "mobilenet_v2":
        model = arch_dict[opts.arch](pretrained=pretrained)
        feature_dim = 1280
    elif opts.arch == "plainnet11":
        model = PlainNet(output_dim=opts.num_classes, final_activation='softmax')
    else:
        raise ValueError("Unknown architecture ", opts.arch)

    if opts.devise or opts.barzdenzler:
        if opts.pretrained or opts.pretrained_folder:
            for param in model.parameters():
                if opts.train_backbone_after == 0:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        if opts.use_2fc:
            if opts.use_fc_batchnorm:
                model.classifier = nn.Sequential(
                    nn.Linear(feature_dim, opts.fc_inner_dim),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm1d(opts.fc_inner_dim),
                    nn.Linear(opts.fc_inner_dim, opts.embedding_size),
                )
            else:
                model.classifier = nn.Sequential(
                    nn.Linear(feature_dim, opts.fc_inner_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(opts.fc_inner_dim, opts.embedding_size),
                )
        else:
            if opts.use_fc_batchnorm:
                model.classifier = nn.Sequential(
                    nn.BatchNorm1d(feature_dim),
                    nn.Linear(feature_dim, opts.embedding_size),
                )
            else:
                model.classifier = nn.Linear(feature_dim, opts.embedding_size)
    else:
        if opts.arch != "plainnet11":
            model.classifier = nn.Sequential(
                nn.Dropout(opts.dropout),
                nn.Linear(feature_dim, opts.num_classes),
            )

    if distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if opts.gpu is not None:
            torch.cuda.set_device(opts.gpu)
            model.cuda(opts.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            opts.batch_size = int(opts.batch_size / gpus_per_node)
            opts.workers = int(opts.workers / gpus_per_node)
            model = nn.parallel.DistributedDataParallel(model, device_ids=[opts.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = nn.parallel.DistributedDataParallel(model)
    elif opts.gpu is not None:
        torch.cuda.set_device(opts.gpu)
        model = model.cuda(opts.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = nn.DataParallel(model).cuda()

    return model
