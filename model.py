import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class feature_extractor(nn.Module): # feature extractor（ResNet50）
    def __init__(self, pre=True):
        super(feature_extractor, self).__init__()
        res50 = models.resnet50(pretrained=pre)
        self.feature_ex = nn.Sequential(*list(res50.children())[:-1])
    def forward(self, input):
        x = input.squeeze(0)
        feature = self.feature_ex(x)
        feature = feature.squeeze(2).squeeze(2)
        return feature

class fc_sub(nn.Module): # sub network
    def __init__(self):
        super(fc_sub, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features=2048, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=512)
        )
    def forward(self, input):
        x = input.squeeze(0)
        z = self.fc(x)
        return z

class fc_label(nn.Module): # fc layer
    def __init__(self, i_dim, m_dim, n_class):
        super(fc_label, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features=i_dim, out_features=m_dim),
            nn.ReLU(),
            nn.Linear(in_features=m_dim, out_features=n_class)
        )
    def forward(self, input):
        x = input.squeeze(0)
        z = self.fc(x)
        return z

class MLP_attention(nn.Module): # attention network
    def __init__(self, i_dim, m_dim):
        super(MLP_attention, self).__init__()
        # attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(i_dim, m_dim),
            nn.Tanh(),
            nn.Linear(m_dim, 1)
        )
    def forward(self, input):
        A = self.attention(input)
        return A

class fc_FCM(nn.Module): # MLP for FCM
    def __init__(self, i_dim, m_dim):
        super(fc_FCM, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=i_dim, out_features=m_dim),
            nn.ReLU(),
        )
    def forward(self, input):
        x = input.squeeze(0)
        z1 = self.fc1(x)
        return z1

class MLP_mmgating(nn.Module): # gating by image and FCM
    def __init__(self, MLP_FCM, num_ex):
        super(MLP_mmgating, self).__init__()
        self.MLP_FCM = MLP_FCM
        self.num_ex = num_ex
        self.fc = nn.Sequential(
            nn.Linear(in_features=(2048+128), out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=num_ex)
        )
    def forward(self, input1, input2):
        input1 = input1.squeeze(0) # 100 x 2048 dim
        input2 = input2.repeat(int(input1.shape[0]),1) # 100 x 18
        feature2 = self.MLP_FCM(input2) # FCM feature 100 x 128
        input_mm = torch.cat([input1, feature2], dim=1) # 100 x (2048 + 128)
        class_prob = self.fc(input_mm).reshape(input1.shape[0],self.num_ex) # 100 x 3
        return class_prob

class MLP_fcmgating(nn.Module): # gating by FCM
    def __init__(self, num_ex):
        super(MLP_fcmgating, self).__init__()
        self.num_ex = num_ex
        self.fc = nn.Sequential(
            nn.Linear(in_features=18, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=num_ex)
        )
    def forward(self, input2):
        input2 = input2.repeat(100,1) # 100 x 18
        class_prob = self.fc(input2).reshape(100,self.num_ex) # 100 x 3
        return class_prob

class MLP_gating(nn.Module): # gating by image
    def __init__(self, num_ex):
        super(MLP_gating, self).__init__()
        self.num_ex = num_ex
        self.fc = nn.Sequential(
            nn.Linear(in_features=2048, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=num_ex)
        )
    def forward(self, input1):
        input1 = input1.squeeze(0)
        weight = self.fc(input1).reshape(input1.shape[0],self.num_ex)
        return weight

class MLP_fcm(nn.Module): # predict class from FCM
    def __init__(self, i_dim, m_dim, n_class):
        super(MLP_fcm, self).__init__()
        self.n_class = n_class
        self.fc = nn.Sequential(
            nn.Linear(i_dim, m_dim),
            nn.ReLU(),
            nn.Linear(m_dim, n_class)
        )
    def forward(self, input):
        x = input.squeeze(0)
        class_prob = self.fc(x).reshape(1,self.n_class)
        class_softmax = F.softmax(class_prob, dim=1)
        class_hat = int(torch.argmax(class_softmax, 1))
        return class_prob, class_hat

class MIL(nn.Module): # MIl without sub network
    def __init__(self, feature_ex, attention, fc_label, n_class):
        super(MIL, self).__init__()
        self.feature_ex = feature_ex
        self.attention = attention
        self.fc_label = fc_label
        self.n_class = n_class
    def forward(self, input):
        x = input.squeeze(0)
        feature = self.feature_ex(x)
        A = self.attention(feature)
        A_t = torch.transpose(A, 1, 0)
        A_n = F.softmax(A_t, dim=1)
        M = torch.mm(A_n, feature)
        class_prob = self.fc_label(M).reshape(1,self.n_class)
        class_softmax = F.softmax(class_prob, dim=1)
        class_hat = int(torch.argmax(class_softmax, 1))
        return class_prob, class_hat, A

class MIL2(nn.Module): # MIL with sub network
    def __init__(self, feature_ex, attention, fc_sub, fc_label, n_class):
        super(MIL2, self).__init__()
        self.feature_ex = feature_ex
        self.attention = attention
        self.fc_sub = fc_sub
        self.fc_label = fc_label
        self.n_class = n_class
    def forward(self, input):
        x = input.squeeze(0)
        feature = self.feature_ex(x)
        feature2 = self.fc_sub(feature)
        A = self.attention(feature2)
        A_t = torch.transpose(A, 1, 0)
        A_n = F.softmax(A_t, dim=1)
        M = torch.mm(A_n, feature2)
        class_prob = self.fc_label(M).reshape(1,self.n_class)
        class_softmax = F.softmax(class_prob, dim=1)
        class_hat = int(torch.argmax(class_softmax, 1))
        return class_prob, class_hat, A

class imgMoE(nn.Module): # MoE using image gating
    def __init__(self, feature_ex, expert1, expert2, expert3, fc_gating, MLP_attention, fc_label, n_class, t):
        super(imgMoE, self).__init__()
        self.feature_ex = feature_ex
        self.expert1 = expert1
        self.expert2 = expert2
        self.expert3 = expert3
        self.fc_gating = fc_gating
        self.MLP_attention = MLP_attention
        self.fc_label = fc_label
        self.n_class = n_class
        self.t = t

    def forward(self, input1):
        input1 = input1.squeeze(0)
        feature = self.feature_ex(input1) # 100 x 2048 dim
        # encoding features by sub networks
        feature1 = self.expert1(feature) # 100 x 512 dim
        feature2 = self.expert2(feature) # 100 x 512 dim
        feature3 = self.expert3(feature) # 100 x 512 dim
        # calculate gating weights
        g_prob = self.fc_gating(feature) # 100 x 3
        g_w = F.softmax(g_prob/self.t, dim=1) # 100 x 3 dim

        feature_gated1 = torch.bmm(g_w[:,0].unsqueeze(1).unsqueeze(2), feature1.unsqueeze(1)).squeeze(1)
        feature_gated2 = torch.bmm(g_w[:,1].unsqueeze(1).unsqueeze(2), feature2.unsqueeze(1)).squeeze(1)
        feature_gated3 = torch.bmm(g_w[:,2].unsqueeze(1).unsqueeze(2), feature3.unsqueeze(1)).squeeze(1)
        # 3 x 100 x 512

        feature_stack = torch.stack([feature_gated1, feature_gated2, feature_gated3],0)
        feature_mean = torch.mean(feature_stack, dim=0) # 100 x 512
        A = self.MLP_attention(feature_mean) # 100 x 1
        A_t = torch.transpose(A, 1, 0) # 1 x 100
        A_n = F.softmax(A_t, dim=1) # 1 x 100
        M = torch.mm(A_n, feature_mean) # 1 x 512

        class_prob = self.fc_label(M).reshape(1,self.n_class)

        class_softmax = F.softmax(class_prob, dim=1).reshape(1,self.n_class)
        class_hat = int(torch.argmax(class_softmax, 1))

        return class_prob, class_hat, g_w, A

class fcmMoE(nn.Module): # MoE using fcm gating
    def __init__(self, feature_ex, expert1, expert2, expert3, gating, MLP_attention, fc_label, n_class, t):
        super(fcmMoE, self).__init__()
        self.feature_ex = feature_ex
        self.expert1 = expert1
        self.expert2 = expert2
        self.expert3 = expert3
        self.gating = gating
        self.MLP_attention = MLP_attention
        self.fc_label = fc_label
        self.n_class = n_class
        self.t = t

    def forward(self, input1, input2):
        input1 = input1.squeeze(0)
        feature = self.feature_ex(input1) # 100 x 2048 dim

        feature1 = self.expert1(feature) # 100 x 512 dim
        feature2 = self.expert2(feature) # 100 x 512 dim
        feature3 = self.expert3(feature) # 100 x 512 dim

        input2 = input2.squeeze(0) # 18 dim

        # calculate gating weights
        g_prob = self.gating(input2)
        g_w = F.softmax(g_prob/self.t, dim=1) # 100 x 3 dim

        feature_gated1 = torch.bmm(g_w[:,0].unsqueeze(1).unsqueeze(2), feature1.unsqueeze(1)).squeeze(1)
        feature_gated2 = torch.bmm(g_w[:,1].unsqueeze(1).unsqueeze(2), feature2.unsqueeze(1)).squeeze(1)
        feature_gated3 = torch.bmm(g_w[:,2].unsqueeze(1).unsqueeze(2), feature3.unsqueeze(1)).squeeze(1)
        # 3 x 100 x 512

        feature_stack = torch.stack([feature_gated1, feature_gated2, feature_gated3],0)
        feature_mean = torch.mean(feature_stack, dim=0) # 100 x 2048
        A = self.MLP_attention(feature_mean) # 100 x 1
        A_t = torch.transpose(A, 1, 0) # 1 x 100
        A_n = F.softmax(A_t, dim=1) # 1 x 100
        M = torch.mm(A_n, feature_mean) # 1 x 2048

        class_prob = self.fc_label(M).reshape(1,self.n_class)

        class_softmax = F.softmax(class_prob, dim=1).reshape(1,self.n_class)
        class_hat = int(torch.argmax(class_softmax, 1))

        return class_prob, class_hat, g_w, A

class mmMoE(nn.Module): # MoE using multimodal gating
    def __init__(self, feature_ex, expert1, expert2, expert3, MM_gating, MLP_attention, fc_label, n_class, t):
        super(mmMoE, self).__init__()
        self.feature_ex = feature_ex
        self.expert1 = expert1
        self.expert2 = expert2
        self.expert3 = expert3
        self.MM_gating = MM_gating
        self.MLP_attention = MLP_attention
        self.fc_label = fc_label
        self.n_class = n_class
        self.t = t

    def forward(self, input1, input2):
        input1 = input1.squeeze(0)
        feature = self.feature_ex(input1) # 100 x 2048 dim

        feature1 = self.expert1(feature) # 100 x 512 dim
        feature2 = self.expert2(feature) # 100 x 512 dim
        feature3 = self.expert3(feature) # 100 x 512 dim

        input2 = input2.squeeze(0) # 18 dim

        # calculate gating weights
        g_prob = self.MM_gating(feature,input2)
        g_w = F.softmax(g_prob/self.t, dim=1) # 100 x 3 dim

        feature_gated1 = torch.bmm(g_w[:,0].unsqueeze(1).unsqueeze(2), feature1.unsqueeze(1)).squeeze(1)
        feature_gated2 = torch.bmm(g_w[:,1].unsqueeze(1).unsqueeze(2), feature2.unsqueeze(1)).squeeze(1)
        feature_gated3 = torch.bmm(g_w[:,2].unsqueeze(1).unsqueeze(2), feature3.unsqueeze(1)).squeeze(1)
        # 3 x 100 x 512

        feature_stack = torch.stack([feature_gated1, feature_gated2, feature_gated3],0)
        feature_mean = torch.mean(feature_stack, dim=0) # 100 x 2048
        A = self.MLP_attention(feature_mean) # 100 x 1
        A_t = torch.transpose(A, 1, 0) # 1 x 100
        A_n = F.softmax(A_t, dim=1) # 1 x 100
        M = torch.mm(A_n, feature_mean) # 1 x 2048

        class_prob = self.fc_label(M).reshape(1,self.n_class)

        class_softmax = F.softmax(class_prob, dim=1).reshape(1,self.n_class)
        class_hat = int(torch.argmax(class_softmax, 1))

        return class_prob, class_hat, g_w, A
