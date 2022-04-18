import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.preprocessing import label_binarize
import pandas as pd

class FocaLoss(nn.Module):
    def __init__(self,device,batch_size, class_num, alpha=None, gamma=None, size_average=True):
        super(FocaLoss, self).__init__()
        self.batch_size = batch_size
        if alpha is None:
            self.alpha = torch.ones((self.batch_size))
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average
        self.device = device

    def forward(self, inputs, targets):
        """labels_one_hot = F.one_hot(targets,2)
        weights = torch.ones((1,2)).float()
        weights = weights.unsqueeze(0)
        weights = weights.repeat(labels_one_hot.shape[0],1) * labels_one_hot
        weights = weights.sum(1)
        weights = weights.unsqueeze(1)
        weights = weights.repeat(1,2)
        self.alpha=weights"""
        N = inputs.size(0)
        #C = inputs.size(1)
        #P = F.softmax(inputs,dim=1)
        logits=torch.sigmoid(inputs)
        #logits=F.softmax(inputs, dim=1)
        BCLoss = F.binary_cross_entropy_with_logits(input = inputs, target = targets,reduction = "none")

        if self.gamma == 0.0:
            modulator = 1.0
        else:
            modulator = torch.exp(-self.gamma * targets * logits - self.gamma * torch.log(1 + 
            torch.exp(-1.0 * logits)))

        loss = modulator * BCLoss

        weighted_loss = self.alpha * loss
        focal_loss = torch.sum(weighted_loss)

        focal_loss /= torch.sum(targets)
        return focal_loss

class CB_loss(nn.Module):
    def __init__(self,key,CLI_NUM,device,batch_size, samples_per_cls,samples_per_participant,class_num,loss_type,beta, gamma):
        super(CB_loss, self).__init__()
        self.batch_size = batch_size
        self.gamma = gamma
        self.class_num = class_num
        self.device = device
        self.beta = beta
        self.gamma=gamma
        #self.samples_per_cls=samples_per_cls
        #self.pos_weight=torch.tensor([5.0])
        self.loss_type=loss_type
        self.epsilon=1
        if key==CLI_NUM:
            self.pos_weight=torch.tensor([5.0])
            self.samples_per_cls=samples_per_cls
        elif key<=CLI_NUM - 1:
            for k, samples in samples_per_participant.items():
                if k==key:
                    self.samples_per_cls =samples
                    div_a=samples[0]/(samples[1]+self.epsilon)
                    seq=torch.tensor([div_a])
                    
                    self.pos_weight=torch.round(seq)
        #print('self.pos_weight_client', self.pos_weight)
        #print("self.pos_weight", self.pos_weight)
       

    def forward(self, inputs, targets):
        """labels_one_hot = F.one_hot(targets,2)
        weights = torch.ones((1,2)).float()
        weights = weights.unsqueeze(0)
        weights = weights.repeat(labels_one_hot.shape[0],1) * labels_one_hot
        weights = weights.sum(1)
        weights = weights.unsqueeze(1)
        weights = weights.repeat(1,2)
        self.alpha=weights"""
        N = inputs.size(0)
        #C = inputs.size(1)
        #P = F.softmax(inputs,dim=1)
        logits=inputs
        #targets= torch.FloatTensor(targets)
        #logits=F.softmax(inputs, dim=1)
       
        effective_num = 1.0 - np.power(self.beta, self.samples_per_cls)
        #rint("effective num of samples",effective_num)
        weights = (1.0 - self.beta) / (np.array(effective_num)+self.epsilon)
        weights = weights / np.sum(weights) * self.class_num
        #print (targets.shape)
        labels_one_hot = targets
       
        #labels_one_hot=label_binarize(targets.numpy(), classes=[0,1])
        #labels_one_hot = pd.get_dummies(targets.numpy())
        ##labels_one_hot = F.one_hot(targets,self.class_num).float()
        #print (labels_one_hot.shape)
        
        weights = torch.tensor(weights).float()
        weights = weights.unsqueeze(0)
        #labels_one_hot = torch.tensor(labels_one_hot.values).float()
        #print (weights.shape)
        weights = weights.repeat(labels_one_hot.shape[0],1) * labels_one_hot
        #print (weights.shape)
        weights = weights.sum(1)
        #print (weights.shape)
        weights = weights.unsqueeze(1)
        #print (weights.shape)
        weights = weights.repeat(1,self.class_num)

        if self.loss_type == "focal":
            logits=torch.sigmoid(inputs)
            #logits=inputs
            BCLoss = F.binary_cross_entropy_with_logits(input = logits, target = labels_one_hot,reduction="mean",pos_weight=self.pos_weight)
            #BCLoss = nn.BCEWithLogitsLoss(reduction = "none",pos_weight=self.pos_weight, weight=weights)

            if self.gamma == 0.0:
                modulator = 1.0
            else:
                modulator = torch.exp(-self.gamma * labels_one_hot * logits - self.gamma * torch.log(1 + torch.exp(-1.0 * logits)))

            loss = modulator * BCLoss

            weighted_loss = weights * loss
            focal_loss = torch.sum(weighted_loss)

            cb_loss =focal_loss/torch.sum(targets)
        elif self.loss_type == "sigmoid":
            cb_loss = F.binary_cross_entropy_with_logits(input = logits,target = labels_one_hot,weight=weights,pos_weight=self.pos_weight)
            #cb_loss = nn.BCEWithLogitsLoss(weight=weights,pos_weight=self.pos_weight)
        elif self.loss_type == "softmax":
            pred = inputs.softmax(dim = 1)
            cb_loss = F.binary_cross_entropy_with_logits(input = pred, target = labels_one_hot,weight=weights,pos_weight=self.pos_weight)
            #cb_loss = nn.BCEWithLogitsLoss(weight=weights, pos_weight=self.pos_weight)
        elif self.loss_type == "ratio":
            logits=torch.sigmoid(inputs)
            #logits=inputs
            BCLoss = F.binary_cross_entropy_with_logits(input = logits, target = labels_one_hot,reduction="mean",pos_weight=self.pos_weight)
            #BCLoss = nn.BCEWithLogitsLoss(reduction = "none",pos_weight=self.pos_weight, weight=weights)

            if self.gamma == 0.0:
                modulator = 1.0
            else:
                modulator = weights * labels_one_hot * logits

            loss = modulator * BCLoss

            weighted_loss = loss
            ratio_loss = torch.sum(weighted_loss)

            cb_loss =ratio_loss/torch.sum(targets)
            
        return cb_loss
       





def CB_loss_old(labels, logits, samples_per_cls, no_of_classes, loss_type, beta, gamma,device):
    """Compute the Class Balanced Loss between `logits` and the ground truth `labels`.
    Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
    where Loss is one of the standard losses used for Neural Networks.
    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      samples_per_cls: A python list of size [no_of_classes].
      no_of_classes: total number of classes. int
      loss_type: string. One of "sigmoid", "focal", "softmax".
      beta: float. Hyperparameter for Class balanced loss.
      gamma: float. Hyperparameter for Focal loss.
    Returns:
      cb_loss: A float tensor representing class balanced loss
    """
    effective_num = 1.0 - np.power(beta,samples_per_cls)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * class_num

    labels_one_hot = F.one_hot(labels, no_of_classes).float()

    weights = torch.tensor(weights).float()
    weights = weights.unsqueeze(0)
    weights = weights.repeat(labels_one_hot.shape[0],1) * labels_one_hot
    weights = weights.sum(1)
    weights = weights.unsqueeze(1)
    weights = weights.repeat(1,no_of_classes)

    if loss_type == "focal":
        #cb_loss = FocaLoss(device,logits,labels_one_hot, weights, gamma,True)
        cb_loss = FocaLoss(device,BATCH_SIZE, class_num=2, alpha=weights, gamma=gamma, size_average=True)
    elif loss_type == "sigmoid":
        cb_loss = F.binary_cross_entropy_with_logits(input = logits,target = labels_one_hot, weights = weights)
    elif loss_type == "softmax":
        pred = logits.softmax(dim = 1)
        cb_loss = F.binary_cross_entropy(input = pred, target = labels_one_hot, weight = weights)
    return cb_loss

class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in 
        Focal Loss for Dense Object Detection.
            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])
        The losses are averaged across observations for each minibatch.
        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5), 
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
    """
    def __init__(self,device, class_num, alpha=None, gamma=2, size_average=True):
        #self.args = args
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average
        self.device = device

    def forward(self, inputs, targets):
        N = inputs.size(0)
        #C = inputs.size(1)
        #P = F.softmax(inputs,dim=1)
        P=torch.sigmoid(inputs)
        class_mask = inputs.data.new(N).fill_(0)
        class_mask = Variable(class_mask)
        #class_mask = class_mask.view(-1)
        ids = targets.long().view(-1)
        class_mask.scatter_(-1, ids.data, 1.)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        #probs = (P*class_mask).sum(1).view(-1)
        probs=P.view(-1)
        #probs=P.sum(1).view(-1)
        log_p = probs.log()
        log_p = log_p.detach().numpy()

        #alpha = alpha.to(self.device)
        #probs = probs.to(self.device)
        #log_p = log_p.to(self.device)

        #batch_loss = -alpha * torch.pow((1 - probs), self.gamma) * log_p
        batch_loss =0
        inputt= inputs.numpy()
        for i in range (len(inputt)):
            batch_loss += -(inputt[i]*self.gamma*log_p[i]+(1-inputt[i])(1-log_p[i]))
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss



class Ratio_Cross_Entropy(nn.Module):
    r"""
            This criterion is a implemenation of Ratio Loss, which is proposed to solve the imbalanced
            problem in Fderated Learning
                Loss(x, class) = - \alpha  \log(softmax(x)[class])
            The losses are averaged across observations for each minibatch.
            Args:
                alpha(1D Tensor, Variable) : the scalar factor for this criterion
                size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                    However, if the field size_average is set to False, the losses are
                                    instead summed for each minibatch.
        """

    def __init__(self,device,class_num, alpha=None, size_average=True):
        super(Ratio_Cross_Entropy, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(1,2))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.class_num = class_num
        self.size_average = size_average
        self.device = device

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C=inputs.size(1)
        #P = F.softmax(inputs,dim=1)
        #print(inputs)
        #print(P)
        #print(inputs.shape)
        #print("y_batch",targets.shape)
        P=torch.sigmoid(inputs)

        class_mask = inputs.data.new(N,C).fill_(0)
        #print("class_mask",class_mask.shape)
        
        class_mask = Variable(class_mask)
        class_mask = class_mask.view(-1)
        ids = targets.long().view(-1)
        #ids = torch.FloatTensor(targets)
        #class_mask.squeeze(1)
        
        class_mask.scatter_(-1, ids.data, 1.)
        #print(class_mask.size())
        #print(ids.size()) 



        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.view(-1)]
        #alpha=self.alpha

        probs = (P * class_mask).sum(1).view(-1, 1)
 
        #probs=P.sum(1).view(-1,1)

        log_p = probs.log()

        alpha = alpha.to(self.device)
        probs = probs.to(self.device)
        log_p = log_p.to(self.device)

        batch_loss = - alpha * log_p

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss

def _expand_binary_labels(labels, label_weights, label_channels):
    bin_labels = labels.new_full((labels.size(0), label_channels), 0)
    inds = torch.nonzero(labels >= 1).squeeze()
    if inds.numel() > 0:
        bin_labels[inds, labels[inds] - 1] = 1
    bin_label_weights = label_weights.view(-1, 1).expand(
        label_weights.size(0), label_channels)
    return bin_labels, bin_label_weights


class GHMC(nn.Module):
    """GHM Classification Loss.
    Details of the theorem can be viewed in the paper
    "Gradient Harmonized Single-stage Detector".
    https://arxiv.org/abs/1811.05181
    Args:
        bins (int): Number of the unit regions for distribution calculation.
        momentum (float): The parameter for moving average.
        use_sigmoid (bool): Can only be true for BCE based loss now.
        loss_weight (float): The weight of the total GHM-C loss.
    """
    def __init__(
            self,
            bins=30,
            momentum=0,
            use_sigmoid=True,
            loss_weight=1.0):
        super(GHMC, self).__init__()
        self.bins = bins
        self.momentum = momentum
        self.edges = torch.arange(bins + 1).float().cuda() / bins
        self.edges[-1] += 1e-6
        if momentum > 0:
            self.acc_sum = torch.zeros(bins).cuda()
        self.use_sigmoid = use_sigmoid
        if not self.use_sigmoid:
            raise NotImplementedError
        self.loss_weight = loss_weight

    def forward(self, pred, target, label, *args, **kwargs):
        """Calculate the GHM-C loss.
        Args:
            pred (float tensor of size [batch_num, class_num]):
                The direct prediction of classification fc layer.
            target (float tensor of size [batch_num, class_num]):
                Binary class target for each sample.
            label_weight (float tensor of size [batch_num, class_num]):
                the value is 1 if the sample is valid and 0 if ignored.
        Returns:
            The gradient harmonized loss.
        """
        # the target should be binary class label
        edges = self.edges
        mmt = self.momentum
        

        N = pred.size(0)
        C = pred.size(1)
        P = F.softmax(pred)

        class_mask = pred.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = label.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)

        # gradient length
        g = torch.abs(pred.sigmoid().detach() - target)
        g = (g * class_mask).sum(1).view(-1, 1)
        weights = torch.zeros_like(g)

        # valid = label_weight > 0
        # tot = max(valid.float().sum().item(), 1.0)
        tot = pred.size(0)
        n = 0  # n valid bins
        for i in range(self.bins):
            inds = (g >= edges[i]) & (g < edges[i+1]) 
            num_in_bin = inds.sum().item()
            # print(num_in_bin)
            if num_in_bin > 0:
                if mmt > 0:
                    self.acc_sum[i] = mmt * self.acc_sum[i] \
                        + (1 - mmt) * num_in_bin
                    weights[inds] = tot / self.acc_sum[i]
                else:
                    weights[inds] = tot / num_in_bin
                n += 1
        if n > 0:
            weights = weights / n

        
        # print(pred)
        # pred = P * weights
        # print(pred)

        probs = (P * class_mask).sum(1).view(-1, 1)

        log_p = probs.log()
        # print(probs)
        # print(probs)
        # print(log_p.size(), weights.size())

        batch_loss = -log_p * weights / tot
        # print(batch_loss)
        loss = batch_loss.sum()
        # print(loss)
        return loss



class DatasetSplit(Dataset):
    def __init__(self, dataset, labels, idxs):
        self.dataset = dataset
        self.labels = labels
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        if len(self.labels) != 0:
            image = self.dataset[self.idxs[item]]
            label = self.labels[self.idxs[item]]
        else:
            image, label = self.dataset[self.idxs[item]]
        return image, label

def AccuarcyCompute(pred, label):
    pred = pred.cpu().data.numpy()
    label = label.cpu().data.numpy()
    test_np = np.argmax(pred, 1)
    count = 0
    for i in range(len(test_np)):
        if test_np[i] == label[i]:
            count += 1
    return np.sum(count), len(test_np)

class LocalUpdate(object):
    def __init__(self, args, dataset=None, label=None, idxs=None, alpha=None, size_average=False):
        self.args = args
        if self.args.loss == 'mse':
            self.loss_func = nn.MSELoss(reduction='mean')
        elif self.args.loss == 'focal':
            self.loss_func = FocalLoss(class_num=26, alpha=alpha, args=args, size_average=size_average)
        elif self.args.loss == 'ratio' or self.args.loss == 'ce':
            self.loss_func = Ratio_Cross_Entropy(class_num=26, alpha=alpha, size_average=size_average, args=args)
        elif self.args.loss == 'ghm':
            self.loss_func = GHMC()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, label, idxs), batch_size=self.args.local_bs, shuffle=True)

    def train(self, net):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=5e-4)

        epoch_loss = []
        epoch_ac = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            batch_ac = []
            batch_whole = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                # print(images, labels)
                images = images.unsqueeze(1)
                # labels = labels.unsqueeze(0)
                images, labels = images.to(self.args.device, dtype=torch.float), labels.to(self.args.device, dtype=torch.long)
                net.zero_grad()
                log_probs = net(images)
                ac, whole = AccuarcyCompute(log_probs, labels)

                # ghm loss
                if self.args.loss == 'ghm':
                    y_pred = log_probs.data.max(1, keepdim=True)[1]
                    target_one_hot = label_binarize(labels.data.view_as(y_pred).cpu().numpy(), [i for i in range(26)])
                    target_one_hot = torch.from_numpy(target_one_hot)
                    target_one_hot = target_one_hot.to(self.args.device)
                    loss = self.loss_func(log_probs, target_one_hot, labels)
                else:
                    loss = self.loss_func(log_probs, labels)

                loss.backward()
                optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(iter, batch_idx * len(images), len(self.ldr_train.dataset),100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
                batch_ac.append(ac)
                batch_whole.append(whole)
            epoch_ac.append(sum(batch_ac) / sum(batch_whole))
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), sum(epoch_ac) / len(epoch_ac)
