import torch
import torch.nn.functional as F

def MIL(y_pred, batch_size, is_transformer=0,num_segs=32):
    loss = torch.tensor(0.).cuda()
    loss_intra = torch.tensor(0.).cuda()
    sparsity = torch.tensor(0.).cuda()
    smooth = torch.tensor(0.).cuda()
    if is_transformer==0:
        y_pred = y_pred.view(batch_size, -1)
    else:
        y_pred = torch.sigmoid(y_pred)

    for i in range(batch_size):
        anomaly_index = torch.randperm(num_segs-2).cuda()
        normal_index = torch.randperm(num_segs-2).cuda()

        y_anomaly = y_pred[i, :num_segs][anomaly_index]
        y_normal  = y_pred[i, num_segs:][normal_index]

        y_anomaly_max = torch.max(y_anomaly) # anomaly
        y_anomaly_min = torch.min(y_anomaly)

        y_normal_max = torch.max(y_normal) # normal
        y_normal_min = torch.min(y_normal)

        loss += F.relu(1.-y_anomaly_max+y_normal_max)

        sparsity += torch.sum(y_anomaly)*0.00008
        smooth += torch.sum((y_pred[i,:num_segs-1] - y_pred[i,1:num_segs])**2)*0.00008
    loss = (loss+sparsity+smooth)/batch_size

    return loss

def MIL_K(y_pred, batch_size, is_transformer=0,num_segs=32):
    loss = torch.tensor(0.).cuda()
    loss_intra = torch.tensor(0.).cuda()
    sparsity = torch.tensor(0.).cuda()
    smooth = torch.tensor(0.).cuda()
    if is_transformer==0:
        y_pred = y_pred.view(batch_size, -1)
    else:
        y_pred = torch.sigmoid(y_pred)

    for i in range(batch_size):
        anomaly_index = torch.randperm(num_segs-2).cuda()
        normal_index = torch.randperm(num_segs-2).cuda()

        y_anomaly = y_pred[i, :num_segs][anomaly_index]
        y_normal  = y_pred[i, num_segs:][normal_index]
        
        kval = 4

        y_anomaly_maxes = y_anomaly.topk(k=4,dim=-1)[0]
        y_normal_maxes = y_normal.topk(k=4,dim=-1)[0]#+y_normal.topk(k=2,largest=False,dim=-1)[0]

        # loss += F.relu(3.-torch.sum(y_anomaly_maxes,dim=-1)+torch.sum(y_normal_maxes,dim=-1))
        loss += F.relu(4.-torch.sum(y_anomaly_maxes,dim=-1)+torch.sum(y_normal_maxes,dim=-1))

        sparsity += torch.sum(y_anomaly)*0.00008
        smooth += torch.sum((y_pred[i,:num_segs-1] - y_pred[i,1:num_segs])**2)*0.00008
    loss = (loss+sparsity+smooth)/batch_size

    return loss





def og_MIL(y_pred, batch_size, is_transformer=0):
    loss = torch.tensor(0.).cuda()
    loss_intra = torch.tensor(0.).cuda()
    sparsity = torch.tensor(0.).cuda()
    smooth = torch.tensor(0.).cuda()
    if is_transformer==0:
        y_pred = y_pred.view(batch_size, -1)
    else:
        y_pred = torch.sigmoid(y_pred)

    for i in range(batch_size):
        anomaly_index = torch.randperm(30).cuda()
        normal_index = torch.randperm(30).cuda()

        y_anomaly = y_pred[i, :32][anomaly_index]
        y_normal  = y_pred[i, 32:][normal_index]

        y_anomaly_max = torch.max(y_anomaly) # anomaly
        y_anomaly_min = torch.min(y_anomaly)

        y_normal_max = torch.max(y_normal) # normal
        y_normal_min = torch.min(y_normal)

        loss += F.relu(1.-y_anomaly_max+y_normal_max)

        sparsity += torch.sum(y_anomaly)*0.00008
        smooth += torch.sum((y_pred[i,:31] - y_pred[i,1:32])**2)*0.00008
    loss = (loss+sparsity+smooth)/batch_size

    return loss
