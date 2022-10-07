import torch
import torch.nn as nn
from torch.nn import functional as F



class Learner(nn.Module):
    def __init__(self, input_dim=2048, drop_p=0.0):
        super(Learner, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(512, 32),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        self.drop_p = 0.6
        self.weight_init()
        self.vars = nn.ParameterList()

        for i, param in enumerate(self.classifier.parameters()):
            self.vars.append(param)

    def weight_init(self):
        for layer in self.classifier:
            if type(layer) == nn.Linear:
                nn.init.xavier_normal_(layer.weight)

    def forward(self, x, vars=None):
        if vars is None:
            vars = self.vars
        x = F.linear(x, vars[0], vars[1])
        x = F.relu(x)
        x = F.dropout(x, self.drop_p, training=self.training)
        x = F.linear(x, vars[2], vars[3])
        x = F.dropout(x, self.drop_p, training=self.training)
        x = F.linear(x, vars[4], vars[5])
        return torch.sigmoid(x)

    def parameters(self):
        """
        override this function since initial parameters will return with a generator.
        :return:
        """
        return self.vars



class LearnerK(nn.Module):
    '''Same params as Learner but implemented with differing names and structure'''
    def __init__(self, input_dim=2048, drop_p=0.0):
        super(LearnerK, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.6)

        self.fc2 = nn.Linear(512, 32)
        self.dropout2 = nn.Dropout(0.6)

        self.fc3 = nn.Linear(32, 1)
        self.sig = nn.Sigmoid()

        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.xavier_normal_(self.fc3.weight)

    def forward(self, x, num_segs=None):
        x = self.dropout1(self.relu1(self.fc1(x)))
        x = self.dropout2(self.fc2(x))
        x = self.sig(self.fc3(x))
        return x
