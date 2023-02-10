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
NAME = "IP"

# seed_number = "1"

def kappa(testData, k): #testData表示要计算的数据，k表示数据矩阵的是k*k的
    dataMat = np.mat(testData)
    s = dataMat.sum()
    #print(dataMat.shape)
    print(dataMat)
    P0 = 0.0
    for i in range(k):
        P0 += dataMat[i, i]*1.0
    xsum = np.sum(dataMat, axis=1)
    ysum = np.sum(dataMat, axis=0)
    #xsum是个k行1列的向量，ysum是个1行k列的向量
    #Pe = float(ysum * xsum) / float(s * 1.0) / float(s * 1.0)
    Pe = float(ysum * xsum) / float(s ** 2)
    print("Pe = ", Pe)
    P0 = float(P0/float(s*1.0))
    #print("P0 = ", P0)
    cohens_coefficient = float((P0-Pe)/(1-Pe))

    a = []
    a = dataMat.sum(axis=0)
    a = np.float32(a)
    a = np.array(a)
    a = np.squeeze(a)

    print(a)

    for i in range(k):
        #print(dataMat[i, i])
        a[i] = float(dataMat[i, i]*1.0)/float(a[i]*1.0)
    print(a*100)
    #print(a.shape)
    print("AA: ", a.mean()*100)
    return cohens_coefficient, a.mean()*100, a*100


parser = argparse.ArgumentParser(description="One Shot Visual Recognition")
#parser.add_argument("-f","--feature_dim",type = int, default = 512)              # 最后一个池化层输出的维度
#parser.add_argument("-r","--relation_dim",type = int, default = 128)               # 第一个全连接层维度
parser.add_argument("-w","--n_way",type = int, default = 16)                      # way
parser.add_argument("-s","--n_shot",type = int, default = 5)       # support set per class
# parser.add_argument("-b","--n_query",type = int, default = 3)       # query set per class
# parser.add_argument("-e","--episode",type = int, default= 1000)
#-----------------------------------------------------------------------------------#
#parser.add_argument("-t","--test_episode", type = int, default = 600)
parser.add_argument("-l","--learning_rate", type = float, default = 0.001)
parser.add_argument("-g","--gpu",type=int, default=0)
args = parser.parse_args()


# Hyper Parameters
#FEATURE_DIM = args.feature_dim
#RELATION_DIM = args.relation_dim
n_way = args.n_way
n_shot = args.n_shot
# n_query = args.n_query
# EPISODE = args.episode
#-----------------------------------------------------------------------------------#
#TEST_EPISODE = args.test_episode
LEARNING_RATE = args.learning_rate
GPU = args.gpu

n_examples = 5  # 训练数据集中每类200个样本
channel_hsi = 200
im_width, im_height, channels = 28, 28, 100

num_fea = 128
num_fea_2 = num_fea*2
num_fea_3 = num_fea_2*2
num_fea_4 = num_fea_3*2

class ChannelTransformation(nn.Module):
    """docstring for ClassName"""

    # Conv3d(in_depth, out_depth, kernel_size, stride=1, padding=0)
    # nn.Conv2d(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True))

    def __init__(self, channel_hsi, channels):
        super(ChannelTransformation, self).__init__()

        self.layer = nn.Sequential(
                        nn.Conv2d(channel_hsi, channels, kernel_size=1, padding=0),
                        nn.BatchNorm2d(channels),
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
                        nn.Conv2d(channels, num_fea, kernel_size=1, padding=0),
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

channel_transformation = ChannelTransformation(channel_hsi, channels)
feature_encoder = CNNEncoder()
relation_network = RelationNetwork()

channel_transformation.cuda(GPU)
feature_encoder.cuda(GPU)
relation_network.cuda(GPU)

channel_transformation_optim = torch.optim.Adam(channel_transformation.parameters(), lr=LEARNING_RATE)
channel_transformation_scheduler = StepLR(optimizer=channel_transformation_optim, step_size=5000, gamma=0.5)
feature_encoder_optim = torch.optim.Adam(feature_encoder.parameters(), lr=LEARNING_RATE)
feature_encoder_scheduler = StepLR(feature_encoder_optim, step_size=100000, gamma=0.5)
relation_network_optim = torch.optim.Adam(relation_network.parameters(), lr=LEARNING_RATE)
relation_network_scheduler = StepLR(relation_network_optim, step_size=100000, gamma=0.5)

T = 6

channel_transformation.load_state_dict(torch.load(str("model/KD_CROSS_MULTI_IP_channel_transformation_16way_2shot_1000epoch_128conv_lr0.0001_m4_ct_3kd_t" + str(T) + ".pkl")))
print("load channel_transformation success")

feature_encoder.load_state_dict(torch.load(str("model/KD_CROSS_MULTI_IP_feature_encoder_16way_2shot_1000epoch_128conv_lr0.0001_m4_ct_3kd_t" + str(T) + ".pkl")))
print("load feature encoder success")

relation_network.load_state_dict(torch.load(str("model/KD_CROSS_MULTI_IP_relation_network_16way_2shot_1000epoch_128conv_lr0.0001_m4_ct_3kd_t" + str(T) + ".pkl")))
print("load relation network success")

channel_transformation.eval()
feature_encoder.eval()
relation_network.eval()

def rn_predict(support_images, test_images, num):

    support_tensor = channel_transformation(torch.from_numpy(support_images).cuda(GPU))
    query_tensor = channel_transformation(torch.from_numpy(test_images).cuda(GPU))

    # calculate features
    sample_features = feature_encoder(Variable(support_tensor).cuda(GPU))  # 数量*通道*高度*宽度
    #print( list(sample_features.size()) ) # [9, 32, 6, 3, 3]
    sample_features = sample_features.view(n_way, n_shot, list(sample_features.size())[-3],
                                           list(sample_features.size())[-2], list(sample_features.size())[
                                               -1])  # view函数改变shape: 5way, 5shot, 64, 19, 19
    # sample_features = torch.sum(sample_features, 1).squeeze(1)  # 同类样本作和
    sample_features = torch.mean(sample_features, 1).squeeze(1)  # 同类样本取平均
    #print( list(sample_features.size()) ) # [9, 32, 6, 3, 3]
    batch_features = feature_encoder(Variable(query_tensor).cuda(GPU))  # 20x64*5*5
    #print(list(batch_features.size())) # [1000, 32, 6, 3, 3]

    ################################################################################################################
    sample_features = sample_features.view(n_way, list(sample_features.size())[1] * list(sample_features.size())[2],
                                           list(sample_features.size())[-2], list(sample_features.size())[-1])
    batch_features = batch_features.view(num,
                                         list(batch_features.size())[1] * list(batch_features.size())[2],
                                         list(batch_features.size())[-2], list(batch_features.size())[-1])
    #print(list(sample_features.size())) # [9, 192, 3, 3]
    #print(list(batch_features.size())) # [1000, 192, 3, 3]
    ################################################################################################################

    # calculate relations
    # 支撑样本和查询样本进行连接
    sample_features_ext = sample_features.repeat(num, 1, 1, 1, 1)  # # repeat函数沿着指定的维度重复tensor
    #print(list(sample_features_ext.size())) # [380, 20, 128, 5, 5]
    batch_features_ext = batch_features.repeat(n_way, 1, 1, 1, 1)
    batch_features_ext = torch.transpose(batch_features_ext, 0, 1)
    #print(list(batch_features_ext.size())) # [380, 20, 128, 5, 5]

    relation_pairs = torch.cat((sample_features_ext, batch_features_ext), 2)
    # print(list(relation_pairs.size())) # [380, 20, 256, 5, 5]
    relation_pairs = relation_pairs.view(-1, list(relation_pairs.size())[-3], list(relation_pairs.size())[-2],
                                         list(relation_pairs.size())[-1])
    # print(list(relation_pairs.size())) # [7600, 256, 5, 5]

    relations = relation_network(relation_pairs)
    #print(list(relations.size())) # [9000, 1]
    relations = relations.view(-1, n_way)
    #print(list(relations.size())) # [1000, 9]

    # 得到预测标签
    _, predict_label = torch.max(relations.data, 1)
    # print('predict_label', predict_label)

    return predict_label


def test(im_width, im_height, channels):

    # 加载支撑数据
    f = h5py.File('data/IP_' + str(im_width) + '_' + str(im_height) + '_' + str(channel_hsi) + '_support' + str(args.n_shot) + '.h5', 'r')
    support_images = np.array(f['data_s'])  # (5, 8100)
    support_images = support_images.reshape(-1, im_width, im_height, channel_hsi).transpose((0, 3, 1, 2))
    print('support_images = ', support_images.shape)  # (9, 1, 100, 9, 9)
    f.close()

    # 加载测试
    f = h5py.File(r'./data/IP_28_28_200_test.h5', 'r')  # 路径
    test_images = np.array(f['data'])  # (42776, 8100)
    test_images = test_images.reshape(-1, im_width, im_height, channel_hsi).transpose((0, 3, 1, 2))
    print('test_images = ', test_images.shape)  # (42776, 1, 100, 9, 9)
    test_labels = f['label'][:]  # (42776, )
    f.close()

    #epi_classes = np.random.permutation(test_images.shape[0])
    #test_images = test_images[epi_classes, :, :, :, :]
    #test_labels = test_labels[epi_classes]

    predict_labels = []  # 记录预测标签
    # S1
    for i in range(10): #10249
        test_images_ = test_images[1000 * i:1000 * (i + 1), :, :, :]
        predict_label = rn_predict(support_images, test_images_, num = 1000)
        predict_labels.extend(predict_label.cpu().numpy().tolist())

    # S2
    test_images_ = test_images[-249:, :, :, :]
    predict_label = rn_predict(support_images, test_images_, num = 249)
    predict_labels.extend(predict_label.cpu().numpy().tolist())

    # S3
    #print(test_labels.shape) # (42776,)
    print(np.unique(predict_labels))
    #print(np.array(predict_labels).shape) # (42776,)
    rewards = [1 if predict_labels[j] == test_labels[j] else 0 for j in range(test_images.shape[0])]
    # print(rewards)
    total_rewards = np.sum(rewards)
    # print(total_rewards)

    accuracy = total_rewards / test_images.shape[0]
    print("accuracy:", accuracy)

    # f = open('./result/prediction.txt', 'w')
    # for i in range(test_images.shape[0]):
    #     f.write(str(predict_labels[i]) + '\n')


################################################################
    n = 10249
    matrix = np.zeros((16, 16), dtype=np.int)
    print(len(predict_labels))
    for j in range(n):
        matrix[test_labels[j], predict_labels[j]] += 1  # 构建混淆矩阵
        # f.write(str(predictions[j]) + '\n')
    # print(matrix)
    # print(np.sum(np.trace(matrix)))  # np.trace 对角线元素之和
    print("OA: ", np.sum(np.trace(matrix)) / float(n) * 100)

    from sklearn import metrics
    kappa_true = metrics.cohen_kappa_score(test_labels, predict_labels)

    kappa_temp, aa_temp, ca_temp = kappa(matrix, 16)

    print(kappa_temp * 100)
    f = open('IP/IP_' + str(np.sum(np.trace(matrix)) / float(n) * 100) + '_m4_t' + str(T) + '.txt', 'w')
    for index in range(len(ca_temp)):
        f.write(str(ca_temp[index]) + '\n')
    f.write(str(np.sum(np.trace(matrix)) / float(n) * 100) + '\n')
    f.write(str(aa_temp) + '\n')
    f.write(str(kappa_true* 100) + '\n')

    from scipy.io import loadmat
    gt = loadmat('D:\hyperspectral_data\Indian_pines_gt.mat')['indian_pines_gt']

    # # 将预测的结果匹配到图像中
    new_show = np.zeros((gt.shape[0], gt.shape[1]))
    k = 0
    for i in range(gt.shape[0]):
        for j in range(gt.shape[1]):
            if gt[i][j] != 0:
                new_show[i][j] = predict_labels[k]
                new_show[i][j] += 1
                k += 1

    # print new_show.shape

    # 展示地物
    import matplotlib as mpl
    import matplotlib.pyplot as  plt

    colors = ['black', 'gray', 'lime', 'cyan', 'forestgreen', 'hotpink', 'saddlebrown',
              'purple', 'red', 'yellow', 'blue', 'steelblue', 'olive', 'sandybrown', 'mediumaquamarine',
              'darkorange',
              'whitesmoke']

    # colors = ['gray', 'lime', 'cyan', 'forestgreen', 'hotpink', 'saddlebrown', 'purple', 'red', 'yellow']
    cmap = mpl.colors.ListedColormap(colors)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(new_show, cmap=cmap)
    plt.savefig("IP/IP_" + str(str(np.sum(np.trace(matrix)) / float(n) * 100)) + "_m4_t" + str(T) + ".png", dpi=1000)  # 保存图像
    # plt.savefig("predict_all.png")#保存图像




test(im_width, im_height, channels)