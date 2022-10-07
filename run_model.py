from ast import Num
from torch import gather
from torch.utils.data import DataLoader
from learner import Learner, LearnerK
from loss import *
from dataset import *
import os
from sklearn import metrics
import pandas as pd
import sys
sys.path.insert(1,'/DATA/top-k-Ranking-Loss/network')
from anomaly_detector_model import AnomalyDetector, custom_objective, original_objective, RegularizedLoss

train_batch_size = 30
test_batch_size = 1

device = 'cuda' if torch.cuda.is_available() else 'cpu'

import wandb



model = 0
optimizer = 0
scheduler = 0 
criterion = MIL
epoch_range = 75



def train(epoch,normal_train_loader,anomaly_train_loader,log_wndb,num_segs,setting=''):
    # print(f'Anomoly test loader len: {len(anomaly_test_loader)}')
    # print(f'Normal test loader len: {len(normal_test_loader)}')
    if setting != '':
        # wandb.init(project="CQ-OCC", name = setting, entity="pmcb99",mode='online')
        # wandb.init(project="FullUCF", name=setting, entity="pmcb99",mode='online')
        # wandb.init(project='EuclideanFeet', name=setting, entity="pmcb99",mode='online')
        # wandb.init(project='SEE-OCC-FullTrain', name=setting, entity="pmcb99",mode='online')
        # wandb.init(project="CQ-TheFullSet", name=setting, entity="pmcb99",mode='online')
        wandb.init(project="SEE-Full", name=setting, entity="pmcb99",mode='online')
    else:
        wandb.init(project="CrimeYolo", entity="pmcb99",mode='disabled')
    wandb.config = {
    "learning_rate": 0.001,
    "epochs": epoch_range,
    "batch_size": train_batch_size,
    'dataset_folder':Normal_Loader.which_dataset()
    }

    # Optional
    wandb.watch(model)
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (normal_inputs, anomaly_inputs) in enumerate(zip(normal_train_loader, anomaly_train_loader)):
        inputs = torch.cat([anomaly_inputs, normal_inputs], dim=1)
        batch_size = inputs.shape[0]
        inputs = inputs.view(-1, inputs.size(-1)).to(device)
        outputs = model(inputs)
        score = outputs.cpu().detach().numpy()
        # get_train_auc(score,normal_inputs,anomaly_inputs)
        loss = criterion(outputs, batch_size, num_segs=num_segs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    wandb.log({"loss": loss})
    print('loss = ', train_loss/len(normal_train_loader))
    scheduler.step()


import pickle
def test_abnormal(epoch,normal_test_loader,anomaly_test_loader,num_segs=32,log_wndb=False,gather_data=False):
    model.eval()
    auc = 0
    auc_list = []
    out_data = {}
    y_trues = np.array([])
    y_preds = np.array([])
    # if log_wndb:
    #     wandb.init(project="cq-crime-data", entity="pmcb99",mode='online')
    # else:
    #     wandb.init(project="CrimeYolo", entity="pmcb99",mode='disabled')

    with torch.no_grad():
        for i, (data, data2) in enumerate(zip(anomaly_test_loader, normal_test_loader)):
            inputs, gts, frames = data
            inputs = inputs.view(-1, inputs.size(-1)).to(torch.device('cuda'))
            score = model(inputs)
            score = score.cpu().detach().numpy()
            score_list = np.zeros(frames[0])
            step = np.round(np.linspace(0, torch.div(frames[0],int(num_segs/2),rounding_mode='trunc'), num_segs+1))

            for j in range(num_segs):
                score_list[int(step[j])*int(num_segs/2):(int(step[j+1]))*int(num_segs/2)] = score[j]

            gt_list = np.zeros(frames[0])
            for k in range(len(gts)//2):
                s = gts[k*2]
                e = min(gts[k*2+1], frames)
                gt_list[s-1:e] = 1
            

            inputs2, gts2, frames2 = data2
            inputs2 = inputs2.view(-1, inputs2.size(-1)).to(torch.device('cuda'))
            score2 = model(inputs2)
            score2 = score2.cpu().detach().numpy()
            score_list2 = np.zeros(frames2[0])
            step2 = np.round(np.linspace(0, torch.div(frames2[0],int(num_segs/2),rounding_mode='trunc'), num_segs+1))
            for kk in range(num_segs):
                score_list2[int(step2[kk])*int(num_segs/2):(int(step2[kk+1]))*int(num_segs/2)] = score2[kk]
            gt_list2 = np.zeros(frames2[0])
            score_list3 = np.concatenate((score_list, score_list2), axis=0)
            gt_list3 = np.concatenate((gt_list, gt_list2), axis=0)
            
            if y_trues == np.array([]):
                y_trues = gt_list3
                y_preds = score_list3
            else:
                y_trues = np.concatenate([y_trues, gt_list3])
                y_preds = np.concatenate([y_preds, score_list3])

            fpr, tpr, thresholds = metrics.roc_curve(gt_list3, score_list3, pos_label=1)
            auc += metrics.auc(fpr, tpr)
            
            # '''Anomaly Subset Metric'''
            # if y_trues == np.array([]):
            #     y_trues = gt_list
            #     y_preds = score_list
            # else:
            #     y_trues = np.concatenate([y_trues, gt_list])
            #     y_preds = np.concatenate([y_preds, score_list])

            # fpr, tpr, thresholds = metrics.roc_curve(gt_list, score_list, pos_label=1)
            # auc += metrics.auc(fpr, tpr)
            out_data[f'anom_{i}']=[frames,score_list,gt_list,y_trues,y_preds]
            out_data[f'norm_{i}']=[frames2,score_list2,gt_list2,y_trues,y_preds]

        fprtotal, tprtotal, thresholdstotal = metrics.roc_curve(y_trues, y_preds, pos_label=1)
        auctotal = metrics.auc(fprtotal,tprtotal)
        auc_list.append(auctotal)
        print('auc = ', auc/len(anomaly_test_loader))
        wandb.log({'auc':auc/len(anomaly_test_loader)})
        wandb.log({'actual_auc':auctotal})
        y_trues = np.array([])
        y_preds = np.array([])
        # wandb.save('out_data.pickle')
        return auctotal,out_data 

def set_seed():
    seed = 52345
    # seed = 1345 #Persos
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def run_detection_model(dataset_path,log_wndb=False,num_segs=32,input_dim=2048,setting='',classname=''):
    global model 
    global optimizer 
    global scheduler 
    global criterion

    # model = Learner(input_dim=input_dim, drop_p=0.0).to(device)
    model = LearnerK(input_dim=input_dim, drop_p=0.0).to(device)
    set_seed()
    optimizer = torch.optim.Adagrad(model.parameters(), lr= 0.001, weight_decay=0.0010000000474974513)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25, 50])
    if '4' in setting:
        criterion = MIL_K
    elif '1' in setting:
        criterion = MIL
    # else:
    #     criterion = MIL_K

    epoch_range = 76
    normal_train_dataset = Normal_Loader(is_train=1,path=dataset_path,setting=setting,classname=classname)
    normal_test_dataset = Normal_Loader(is_train=0,path=dataset_path,setting=setting,classname=classname)
    anomaly_train_dataset = Anomaly_Loader(is_train=1,path=dataset_path,setting=setting,classname=classname)
    anomaly_test_dataset = Anomaly_Loader(is_train=0,path=dataset_path,setting=setting,classname=classname)
    shuffle = True
    normal_train_loader = DataLoader(normal_train_dataset, batch_size=train_batch_size, shuffle=shuffle)
    set_seed()
    normal_test_loader = DataLoader(normal_test_dataset, batch_size=test_batch_size, shuffle=shuffle)
    set_seed()
    anomaly_train_loader = DataLoader(anomaly_train_dataset, batch_size=train_batch_size, shuffle=shuffle) 
    set_seed()
    anomaly_test_loader = DataLoader(anomaly_test_dataset, batch_size=test_batch_size, shuffle=shuffle)
    set_seed()
    auc_avgs = [0]
    new_auc = 0
    if os.path.exists('/DATA/best-model-parameters.pt'):
        model = LearnerK(input_dim=input_dim, drop_p=0.0).to(device)
        model.load_state_dict(torch.load('best-model-parameters.pt'))
        auc_avg = test_abnormal(1,normal_test_loader,anomaly_test_loader,num_segs=num_segs,log_wndb=True)
    else:
        # wandb.init(project="cq-crime-data", entity="pmcb99",mode='online')
        for epoch in range(0, epoch_range):
            log_wndb = True
            train(epoch,normal_train_loader, anomaly_train_loader,log_wndb=log_wndb,num_segs=num_segs,setting=setting)
            totalauc, plot_data = test_abnormal(epoch,normal_test_loader,anomaly_test_loader,num_segs=num_segs,log_wndb=log_wndb)
            dirname = 'NEWOCC'
            pickle_name = f'{dirname}/{setting}.pickle'
            os.makedirs(dirname,exist_ok=True)
            if totalauc >= max(auc_avgs):
                print(f'Saving file.. {totalauc}')
                with open(pickle_name, 'wb') as handle:
                    pickle.dump(plot_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
            auc_avgs.append(totalauc)
        wandb.save(pickle_name)
        wandb.log({'auctotal':max(auc_avgs)})
        wandb.finish()
        # if os.path.exists(pickle_name):
        #     os.remove(pickle_name)

            # auc_avgs.append(auc_avg)
            # print(auc_avg,max(auc_avgs))
            # if auc_avg >= max(auc_avgs):
            #     print('SAVING BEST MODEL')
            #     torch.save(model.state_dict(), '/DATA/best-model-parameters.pt')
            # wandb.save('best-model-parameters.pt')

        


# def og_test_abnormal(epoch,normal_test_loader,anomaly_test_loader):
#     model.eval()
#     auc = 0
#     with torch.no_grad():
        
#         for i, (data, data2) in enumerate(zip(anomaly_test_loader, normal_test_loader)):
#             inputs, gts, frames = data
#             inputs = inputs.view(-1, inputs.size(-1)).to(torch.device('cuda'))
#             score = model(inputs)
#             score = score.cpu().detach().numpy()
#             score_list = np.zeros(frames[0])
#             step = np.round(np.linspace(0, torch.div(frames[0],16,rounding_mode='trunc'), 33))

#             for j in range(32):
#                 score_list[int(step[j])*16:(int(step[j+1]))*16] = score[j]

#             gt_list = np.zeros(frames[0])
#             for k in range(len(gts)//2):
#                 s = gts[k*2]
#                 e = min(gts[k*2+1], frames)
#                 gt_list[s-1:e] = 1

#             inputs2, gts2, frames2 = data2
#             inputs2 = inputs2.view(-1, inputs2.size(-1)).to(torch.device('cuda'))
#             score2 = model(inputs2)
#             score2 = score2.cpu().detach().numpy()
#             score_list2 = np.zeros(frames2[0])
#             step2 = np.round(np.linspace(0, torch.div(frames2[0],16,rounding_mode='trunc'), 33))
#             for kk in range(32):
#                 score_list2[int(step2[kk])*16:(int(step2[kk+1]))*16] = score2[kk]
#             gt_list2 = np.zeros(frames2[0])
#             score_list3 = np.concatenate((score_list, score_list2), axis=0)
#             gt_list3 = np.concatenate((gt_list, gt_list2), axis=0)

#             fpr, tpr, thresholds = metrics.roc_curve(gt_list3, score_list3, pos_label=1)
#             auc += metrics.auc(fpr, tpr)
#         print('auc = ', auc/len(anomaly_test_loader))
#         wandb.log({'auc':auc/len(anomaly_test_loader)})