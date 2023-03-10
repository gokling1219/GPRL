import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np
import math
import argparse
import h5py
import time
from torchvision import transforms
from PIL import Image

#torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = True
NAME = "KD_CROSS_MULTI_IP"


parser = argparse.ArgumentParser(description="One Shot Visual Recognition")
#parser.add_argument("-f","--feature_dim",type = int, default = 512)              # 最后一个池化层输出的维度
#parser.add_argument("-r","--relation_dim",type = int, default = 128)               # 第一个全连接层维度
parser.add_argument("-w","--n_way",type = int, default = 16)                      # way
parser.add_argument("-s","--n_shot",type = int, default = 2)       # support set per class
parser.add_argument("-b","--n_query",type = int, default = 3)       # query set per class
parser.add_argument("-e","--episode",type = int, default= 1000)
#-----------------------------------------------------------------------------------#
#parser.add_argument("-t","--test_episode", type = int, default = 600)
parser.add_argument("-l","--learning_rate", type = float, default = 0.0001)
parser.add_argument("-g","--gpu",type=int, default=0)
args = parser.parse_args()


# Hyper Parameters
#FEATURE_DIM = args.feature_dim
#RELATION_DIM = args.relation_dim
n_way = args.n_way
n_shot = args.n_shot
n_query = args.n_query
EPISODE = args.episode
#-----------------------------------------------------------------------------------#
#TEST_EPISODE = args.test_episode
LEARNING_RATE = args.learning_rate
GPU = args.gpu

n_examples = 5  # 训练数据集中每类200个样本
channel_hsi = 200
im_width, im_height, depth = 28, 28, 100 # 输入的cube为固定值

num_fea = 128
num_fea_2 = num_fea*2
num_fea_3 = num_fea_2*2
num_fea_4 = num_fea_3*2

class ChannelTransformation(nn.Module):
    """docstring for ClassName"""

    # Conv3d(in_depth, out_depth, kernel_size, stride=1, padding=0)
    # nn.Conv2d(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True))

    def __init__(self):
        super(ChannelTransformation, self).__init__()

        self.layer = nn.Sequential(
                        nn.Conv2d(channel_hsi, depth, kernel_size=1, padding=0),
                        nn.BatchNorm2d(depth),
                        nn.ReLU())

    def forward(self,x):

        out = self.layer(x)

        return out # 64

class CNNEncoder(nn.Module):
    """docstring for ClassName"""

    # Conv3d(in_depth, out_depth, kernel_size, stride=1, padding=0)
    # nn.Conv2d(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True))

    def __init__(self):
        super(CNNEncoder, self).__init__()

        self.layer1 = nn.Sequential(
                        nn.Conv2d(depth, num_fea, kernel_size=1, padding=0),
                        nn.BatchNorm2d(num_fea),
                        nn.ReLU())

        # self.res1 = nn.Sequential(
        #                 nn.Conv2d(num_fea, num_fea, kernel_size=3, padding=1),
        #                 nn.BatchNorm2d(num_fea),
        #                 nn.ReLU(),
        #                 nn.Conv2d(num_fea, num_fea, kernel_size=3, padding=1),
        #                 nn.BatchNorm2d(num_fea),
        #                 nn.ReLU())

        self.layer2 = nn.Sequential(
            nn.Conv2d(num_fea, num_fea_2, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_fea_2),
            nn.ReLU())

        self.res2 = nn.Sequential(
                        nn.Conv2d(num_fea_2, num_fea_2, kernel_size=3, padding=1),
                        nn.BatchNorm2d(num_fea_2),
                        nn.ReLU(),
                        nn.Conv2d(num_fea_2, num_fea_2, kernel_size=3, padding=1),
                        nn.BatchNorm2d(num_fea_2),
                        nn.ReLU())

        self.layer3 = nn.Sequential(
            nn.Conv2d(num_fea_2, num_fea_3, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_fea_3),
            nn.ReLU())

        # self.res3 = nn.Sequential(
        #                 nn.Conv2d(num_fea_3,num_fea_3,kernel_size=3,padding=1),
        #                 nn.BatchNorm2d(num_fea_3),
        #                 nn.ReLU(),
        #                 nn.Conv2d(num_fea_3, num_fea_3, kernel_size=3, padding=1),
        #                 nn.BatchNorm2d(num_fea_3),
        #                 nn.ReLU())

        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2), padding=1)

        self.layer4 = nn.Sequential(
            nn.Conv2d(num_fea_3, num_fea_4, kernel_size=3, padding=0),
            nn.BatchNorm2d(num_fea_4),
            nn.ReLU())

        self.layer5 = nn.Sequential(
            nn.Conv2d(num_fea_4, num_fea_4, kernel_size=3, padding=0),
            nn.BatchNorm2d(num_fea_4),
            nn.ReLU())

    def forward(self,x):

        out1 = self.layer1(x)
        out1 = self.maxpool(out1)
        # out1 = self.res1(out) + out
        # out1 = self.maxpool(out1)
        # print(out1.shape)

        out2 = self.layer2(out1)
        out2 = self.res2(out2) + out2
        out2 = self.maxpool(out2)
        # print(out2.shape)

        out3 = self.layer3(out2)
        # out4 = self.res3(out3) + out3
        out4 = self.maxpool(out3)
        # print(out4.shape)

        out5 = self.layer4(out4)
        out5 = self.layer5(out5)
        # print(out5.shape)


        #out = out.view(out.size(0),-1)
        #print(list(out5.size())) # [100, 128, 1, 1]
        return out5 # 64

class RelationNetwork(nn.Module):
    """docstring for RelationNetwork"""
    def __init__(self):
        super(RelationNetwork, self).__init__()

        self.layer1 = nn.Sequential(
                        nn.Conv2d(num_fea_4*2, 512, kernel_size=1, padding=0),
                        nn.BatchNorm2d(512),
                        nn.ReLU())
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(p = 0.5)                                                                              # 测试的时候需要修改....？？？

    def forward(self,x): # [7600, 128, 2, 2]
        out = self.layer1(x)
        #print(list(out.size()))
        #print(list(out.size())) # [6000, 128, 2, 2]
        out = out.view(out.size(0),-1) # flatten
        #print(list(out.size())) # [6000, 512]
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = F.sigmoid(self.fc2(out))
        #print("ssss", list(out.size())) # [6000, 1]
        return out

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())


def train():

    T = 6

    np.random.seed(123456789)

    channel_transformation = ChannelTransformation()
    feature_encoder = CNNEncoder()
    relation_network = RelationNetwork()

    feature_encoder.load_state_dict(torch.load(str("./model/KD_CROSS_CT_feature_encoder_10000epoch_128conv_lr0.0001_m4_ct_3kd_t" + str(T) + ".pkl"), map_location='cuda:0'))
    print("load feature encoder success")

    # relation_network.load_state_dict(torch.load(str("./model/kd_single_relationnet/KD_CROSS_relation_network_10000epoch_128conv_lr0.0001.pkl"), map_location='cuda:0'))
    # print("load relation network success")
    relation_network.apply(weights_init)
    channel_transformation.apply(weights_init)


    feature_encoder.cuda(GPU)
    relation_network.cuda(GPU)
    channel_transformation.cuda(GPU)

    channel_transformation.train()
    feature_encoder.train()
    relation_network.train()

    channel_transformation_optim = torch.optim.Adam(channel_transformation.parameters(), lr=LEARNING_RATE)
    channel_transformation_scheduler = StepLR(optimizer=channel_transformation_optim, step_size=5000, gamma=0.5)
    feature_encoder_optim = torch.optim.Adam(feature_encoder.parameters(), lr=LEARNING_RATE)
    feature_encoder_scheduler = StepLR(optimizer=feature_encoder_optim, step_size=5000, gamma=0.5)
    # 每过step_size次,更新一次学习率;每经过100000次，学习率折半
    relation_network_optim = torch.optim.Adam(relation_network.parameters(), lr=LEARNING_RATE)
    relation_network_scheduler = StepLR(relation_network_optim, step_size=5000, gamma=0.5)



    # 训练数据集
    f = h5py.File(r'./data/IP_28_28_200_support5.h5', 'r')
    train_dataset = f['data_s'][:]
    # print(train_dataset.shape) # (2000, 20, 28, 28, 3)
    f.close()

    train_dataset = train_dataset.reshape(-1, n_examples, 28, 28, channel_hsi)  # 划分成了78类，每类200个样本
    train_dataset = train_dataset.transpose((0, 1, 4, 2, 3))
    print(train_dataset.shape) # (55, 200, 100, 28, 28)
    n_train_classes = train_dataset.shape[0]

    accuracy_ = []
    loss_ = []
    a = time.time()
    for episode in range(1, EPISODE+1):

        feature_encoder_scheduler.step(episode)
        relation_network_scheduler.step(episode)
        channel_transformation_scheduler.step(episode)

        # start:每一个episode的采样过程##################################################
        # start:每一个episode的采样过程##################################################
        epi_classes = np.random.permutation(n_train_classes)[
                      :n_way]  # 在48个数里面随机抽取前20个 48为类别数量 随机抽取20个类别，例如15 69 23 ....
        support = np.zeros([n_way, n_shot, channel_hsi, im_height, im_width], dtype=np.float32)  # n_shot = 1
        query = np.zeros([n_way, n_query, channel_hsi, im_height, im_width], dtype=np.float32)  # n_query= 19
        # (N,C_in,H_in,W_in)

        for i, epi_cls in enumerate(epi_classes):
            selected = np.random.permutation(n_examples)[:n_shot + n_query]  # 支撑集合
            support[i] = train_dataset[epi_cls, selected[:n_shot]]
            query[i] = train_dataset[epi_cls, selected[n_shot:]]

        support = support.reshape(n_way * n_shot, channel_hsi, im_height, im_width)
        query = query.reshape(n_way * n_query, channel_hsi, im_height, im_width)
        support_tensor = torch.from_numpy(support).type(torch.FloatTensor)
        #print("support_tensor.shape", support_tensor.shape) # (20, 3, 28, 28)
        query_tensor = torch.from_numpy(query).type(torch.FloatTensor)
        #print("query_tensor.shape", query_tensor.shape)


        labels = np.tile(np.arange(n_way)[:, np.newaxis], (1, n_query)).astype(np.uint8).reshape(-1)
        label_tensor = torch.LongTensor(labels)
        # end:每一个episode的采样过程####################################################################################
        # end:每一个episode的采样过程####################################################################################

        # calculate features
        support_tensor = channel_transformation(Variable(support_tensor).cuda(GPU))
        query_tensor = channel_transformation(Variable(query_tensor).cuda(GPU))

        sample_features = feature_encoder(Variable(support_tensor).cuda(GPU))  # 数量*通道*高度*宽度
        # print("list(sample_features.size())", list(sample_features.size()) ) # [20, 256, 1, 1]
        sample_features = sample_features.view(n_way, n_shot, list(sample_features.size())[-3])  # view函数改变shape:

        #sample_features = torch.sum(sample_features, 1).squeeze(1)  # 同类样本作和
        sample_features = torch.mean(sample_features, 1).squeeze(1)  # 同类样本取平均

        # print("list(sample_features.size())", list(sample_features.size()) ) # [20, 256]
        batch_features = feature_encoder(Variable(query_tensor).cuda(GPU))
        # print( "list(batch_features.size())", list(batch_features.size())) # [180, 256, 1, 1]
        batch_features = batch_features.view(list(batch_features.size())[0], list(batch_features.size())[1])
        # print("list(batch_features.size())", list(batch_features.size()))  # [180, 256]

        # calculate relations
        sample_features_ext = sample_features.unsqueeze(0).repeat(n_query * n_way, 1, 1)  # # repeat函数沿着指定的维度重复tensor
        # print("list(sample_features_ext.size())", list(sample_features_ext.size())) # [180, 20, 256]
        batch_features_ext = batch_features.unsqueeze(0).repeat(n_way, 1, 1)
        batch_features_ext = torch.transpose(batch_features_ext, 0, 1)
        # print("list(batch_features_ext.size())", list(batch_features_ext.size())) # [180, 20, 256]

        relation_pairs = torch.cat((sample_features_ext, batch_features_ext), 2)
        # print("list(relation_pairs.size())", list(relation_pairs.size())) # [180, 20, 512]
        relation_pairs = relation_pairs.view(-1, list(relation_pairs.size())[-1], 1, 1)
        # print("list(relation_pairs.size())", list(relation_pairs.size())) # [3600, 512, 1, 1]

        relations = relation_network(relation_pairs)
        # print("list(relations.size())", list(relations.size())) # [3600, 1]
        relations = relations.view(-1, n_way)
        # print("list(relations.size())", list(relations.size())) # [180, 20]

        mse = nn.MSELoss().cuda(GPU)
        one_hot_labels = Variable(
            torch.zeros(n_query * n_way, n_way).scatter_(dim=1, index=label_tensor.view(-1, 1), value=1).cuda(GPU))
        # scatter中1表示按照行顺序进行填充，labels_tensor.view(-1,1)为索引，1为填充数字
        loss = mse(relations, one_hot_labels)

        # training
        # 把模型中参数的梯度设为0
        feature_encoder.zero_grad()
        relation_network.zero_grad()
        channel_transformation.zero_grad()

        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm(feature_encoder.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm(relation_network.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm(channel_transformation.parameters(), 0.5)

        # 进行单次优化，参数更新
        feature_encoder_optim.step()
        relation_network_optim.step()
        channel_transformation_optim.step()


        if episode == 1 or episode % 100 == 0:

            print("episode:",episode,"loss",loss)
            #################调试#################
            _, predict_label = torch.max(relations.data, 1)
            predict_label = predict_label.cpu().numpy().tolist()
            #print(predict_label)
            #print(labels)
            rewards = [1 if predict_label[j] == labels[j] else 0 for j in range(labels.shape[0])]
            # print(rewards)
            total_rewards = np.sum(rewards)
            # print(total_rewards)

            accuracy = total_rewards*100.0 / labels.shape[0]
            print("accuracy:", accuracy)
            accuracy_.append(accuracy)
            loss_.append(loss.item())


    torch.save(channel_transformation.state_dict(),
               str('./model/' + NAME + '_channel_transformation_' + str(n_way) + 'way_' + str(n_shot) + 'shot_' + str(episode)
                   + 'epoch_' + str(num_fea) + 'conv_lr' + str(LEARNING_RATE) + '_m4_ct_3kd_t' + str(T) + '.pkl'))
    torch.save(feature_encoder.state_dict(),
                       str('./model/' + NAME + '_feature_encoder_' + str(n_way) + 'way_' + str(n_shot) + 'shot_' + str(episode)
                           + 'epoch_' + str(num_fea) + 'conv_lr' + str(LEARNING_RATE) + '_m4_ct_3kd_t' + str(T) + '.pkl'))
    torch.save(relation_network.state_dict(),
                       str('./model/' + NAME + '_relation_network_' + str(n_way) + 'way_' + str(n_shot) + 'shot_' + str(
                           episode)
                           + 'epoch_' + str(num_fea) + 'conv_lr' + str(LEARNING_RATE) + '_m4_ct_3kd_t' + str(T) + '.pkl'))

    print('time = ', time.time()-a)



    f = open('./record/' + NAME + '_finetune_loss_' +str(n_way) + 'way_' + str(n_shot) + 'shot_' + str(EPISODE)
                                                 + 'epoch_' +str(num_fea) + 'conv_lr' + str(LEARNING_RATE) + '_m4_ct_3kd_t' + str(T) + '.txt', 'w')
    for i in range(np.array(loss_).shape[0]):
        f.write(str(loss_[i]) + '\n')
    f = open('./record/' + NAME + '_finetune_accuracy_' +str(n_way) + 'way_' + str(n_shot) + 'shot_' + str(EPISODE)
                                                 + 'epoch_' +str(num_fea) + 'conv_lr' + str(LEARNING_RATE) + '_m4_ct_3kd_t' + str(T) + '.txt', 'w')
    for i in range(np.array(accuracy_).shape[0]):
        f.write(str(accuracy_[i]) + '\n')

    f = open('./record/' + NAME + '_finetune_time' + str(EPISODE)+ 'epoch' + '_m4_ct_3kd_t' + str(T) + '.txt', 'w')
    f.write(str(time.time()-a))



if __name__ == '__main__':
    train()