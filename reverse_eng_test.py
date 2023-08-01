from torchvision import datasets, models, transforms
import os
import torch
from torch.autograd import Variable
from skimage import io
from scipy import fftpack
import numpy as np
from torch import nn
import datetime
from models import encoder_rev_eng
from models import fen
import torch.nn.functional as F
import argparse
 


#################################################################################################################
# HYPER PARAMETERS INITIALIZING
parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
parser.add_argument('--data_test',default='/mnt/scratch/asnanivi/GAN_data_6/set_1/test',help='root directory for testing data')
parser.add_argument('--ground_truth_dir',default='./',help='directory for ground truth')
parser.add_argument('--seed', default=1, type=int, help='manual seed')
parser.add_argument('--batch_size', default=16, type=int, help='batch size')
parser.add_argument('--savedir', default='/mnt/scratch/asnanivi/runs')
parser.add_argument('--model_dir', default='./models')
parser.add_argument('--N_given', nargs='+', help='position number of GM from list of GMs used in testing', default=[1,2,3,4,5,6])


opt = parser.parse_args()
print(opt)
print("Random Seed: ", opt.seed)

device=torch.device("cuda:0")
torch.backends.deterministic = True
torch.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)

sig = str(datetime.datetime.now())

set_name="set_1"   
train_path=opt.data_train
test_path=opt.data_test
save_dir=opt.savedir

ground_truth_net_all=torch.from_numpy(np.load(opt.ground_truth_dir+ "ground_truth_net_arch_100_15dim.npy"))
ground_truth_loss_all=torch.from_numpy(np.load(opt.ground_truth_dir+ "ground_truth_loss_100_3dim.npy"))
ground_truth_loss_9_all=torch.from_numpy(np.load(opt.ground_truth_dir+ "ground_truth_loss_100_9dim.npy"))

N_all=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99]
N = [x for x in N_all if x not in opt.N_given]
ground_truth_net=ground_truth_net_all[N]
ground_truth_loss=ground_truth_loss_all[N]
ground_truth_loss_9=ground_truth_loss_9_all[N]


os.makedirs('%s/result_3/%s' % (save_dir, sig), exist_ok=True)


transform_train = transforms.Compose([
transforms.Resize((64,64)),
transforms.ToTensor(),
transforms.Normalize((0.6490, 0.6490, 0.6490), (0.1269, 0.1269, 0.1269))
])


test_set=datasets.ImageFolder(test_path, transform_train)
b_s=opt.batch_size
test_loader = torch.utils.data.DataLoader(test_set,batch_size=b_s,shuffle =True, num_workers=1)

model=fen.DnCNN().to(device)
   
model_params = list(model.parameters())    
optimizer = torch.optim.Adam(model_params, lr=opt.lr)


model_2=encoder_rev_eng.encoder(num_hidden=512).to(device)
optimizer_2 = torch.optim.Adam(model_2.parameters(), lr=opt.lr)

weightn1 = torch.tensor([100/69, 100/19, 100/6,100/6])
weightn2 = torch.tensor([100/6, 100/74, 100/7,100/13])
weightn3 = torch.tensor([100/3, 100/74, 100/21,100/2])
weightn4 = torch.tensor([100/70, 100/30])
weightn5 = torch.tensor([100/34, 100/66])
weightn6 = torch.tensor([100/55, 100/45])

weight3L1 = torch.tensor([100/35, 100/65])
weight3L2 = torch.tensor([100/35, 100/65])
weight3L3 = torch.tensor([100/67, 100/33])

weight9L1 = torch.tensor([100/56, 100/44])
weight9L2 = torch.tensor([100/80, 100/20])
weight9L3 = torch.tensor([100/86, 100/14])
weight9L4 = torch.tensor([100/96, 100/4])
weight9L5 = torch.tensor([100/52, 100/48])
weight9L6 = torch.tensor([100/86, 100/14])
weight9L7 = torch.tensor([100/93, 100/7])
weight9L8 = torch.tensor([100/68, 100/32])




l1=torch.nn.MSELoss().to(device)

l_cn1 = torch.nn.CrossEntropyLoss(weightn1).to(device)
l_cn2 = torch.nn.CrossEntropyLoss(weightn2).to(device)
l_cn3 = torch.nn.CrossEntropyLoss(weightn3).to(device)
l_cn4 = torch.nn.CrossEntropyLoss(weightn4).to(device)
l_cn5 = torch.nn.CrossEntropyLoss(weightn5).to(device)
l_cn6 = torch.nn.CrossEntropyLoss(weightn6).to(device)


l_c3L1 = torch.nn.CrossEntropyLoss(weight3L1).to(device)
l_c3L2 = torch.nn.CrossEntropyLoss(weight3L2).to(device)
l_c3L3 = torch.nn.CrossEntropyLoss(weight3L3).to(device)

l_c9L1 = torch.nn.CrossEntropyLoss(weight9L1).to(device)
l_c9L2 = torch.nn.CrossEntropyLoss(weight9L2).to(device)
l_c9L3 = torch.nn.CrossEntropyLoss(weight9L3).to(device)
l_c9L4 = torch.nn.CrossEntropyLoss(weight9L4).to(device)
l_c9L5 = torch.nn.CrossEntropyLoss(weight9L5).to(device)
l_c9L6 = torch.nn.CrossEntropyLoss(weight9L6).to(device)
l_c9L7 = torch.nn.CrossEntropyLoss(weight9L7).to(device)
l_c9L8 = torch.nn.CrossEntropyLoss(weight9L8).to(device)


state = {
    'state_dict_cnn':model.state_dict(),
    'optimizer_1': optimizer.state_dict(),
    'state_dict_class':model_2.state_dict(),
    'optimizer_2': optimizer_2.state_dict()
    
}

state1 = torch.load(opt.model_dir)
optimizer.load_state_dict(state1['optimizer_1'])
model.load_state_dict(state1['state_dict_cnn'])
optimizer_2.load_state_dict(state1['optimizer_2'])
model_2.load_state_dict(state1['state_dict_class'])




def test(batch,labels):
    model.eval()
    model_2.eval()
    with torch.no_grad():
        y,low_freq_part,max_value ,y_orig,residual, y_trans,residual_gray =model(batch.type(torch.cuda.FloatTensor))
        y_2=torch.unsqueeze(y.clone(),1)
        outn1,outn2,outn3,outn4,outn5 ,outn6,outn7,out3L1,out3L2,out3L3,outh9L1,outh9L2,outh9L3,outh9L4,outh9L5,outh9L6,outh9L7,outh9L8=model_2(residual)
        
    return y,y_orig,residual, outn1,outn2,outn3,outn4,outn5,outn6,outn7,out3L1,out3L2,out3L3,outh9L1,outh9L2,outh9L3,outh9L4,outh9L5,outh9L6,outh9L7,outh9L8


print(len(test_set))
print(test_set.class_to_idx)

epochs=20


for epoch in range(epochs):
    all_y=[]
    all_y_test=[]
    flag=0
    flag1=0
    
    count=0
    number=0
    
        
    
    for batch_idx_test, (inputs_test,labels_test) in enumerate(test_loader):
        out,out_orig,residual,outn1,outn2,outn3,outn4,outn5,outn6,outn7,out3L1,out3L2,out3L3,outh9L1, outh9L2,outh9L3,outh9L4,outh9L5,outh9L6,outh9L7,outh9L8=test(Variable(torch.FloatTensor(inputs_test)),Variable(torch.LongTensor(labels_test)))
        all_y_test.append(np.asarray(labels_test))
        if flag1==0:
            
            flag1=1
            
            
            all_features_net1=outn1.detach()
            all_features_net2=outn2.detach()
            all_features_net3=outn3.detach()
            all_features_net4=outn4.detach()
            all_features_net5=outn5.detach()
            all_features_net6=outn6.detach()
            all_features_net7=outn7.detach()
            
            all_features_loss1=out3L1.detach()
            all_features_loss2=out3L2.detach()
            all_features_loss3=out3L3.detach()
            
            all_features_loss_91=outh9L1.detach()
            all_features_loss_92=outh9L2.detach()
            all_features_loss_93=outh9L3.detach()
            all_features_loss_94=outh9L4.detach()
            all_features_loss_95=outh9L5.detach()
            all_features_loss_96=outh9L6.detach()
            all_features_loss_97=outh9L7.detach()
            all_features_loss_98=outh9L8.detach()
        else:
            
            all_features_net1=torch.cat([all_features_net1,outn1.detach()], dim=0)
            all_features_net2=torch.cat([all_features_net2,outn2.detach()], dim=0)
            all_features_net3=torch.cat([all_features_net3,outn3.detach()], dim=0)
            all_features_net4=torch.cat([all_features_net4,outn4.detach()], dim=0)
            all_features_net5=torch.cat([all_features_net5,outn5.detach()], dim=0)
            all_features_net6=torch.cat([all_features_net6,outn6.detach()], dim=0)
            all_features_net7=torch.cat([all_features_net7,outn7.detach()], dim=0)
            
            all_features_loss1=torch.cat([all_features_loss1,out3L1.detach()], dim=0)
            all_features_loss2=torch.cat([all_features_loss2,out3L2.detach()], dim=0)
            all_features_loss3=torch.cat([all_features_loss3,out3L3.detach()], dim=0)
            
            all_features_loss_91=torch.cat([all_features_loss_91,outh9L1.detach()], dim=0)
            all_features_loss_92=torch.cat([all_features_loss_92,outh9L2.detach()], dim=0)
            all_features_loss_93=torch.cat([all_features_loss_93,outh9L3.detach()], dim=0)
            all_features_loss_94=torch.cat([all_features_loss_94,outh9L4.detach()], dim=0)
            all_features_loss_95=torch.cat([all_features_loss_95,outh9L5.detach()], dim=0)
            all_features_loss_96=torch.cat([all_features_loss_96,outh9L6.detach()], dim=0)
            all_features_loss_97=torch.cat([all_features_loss_97,outh9L7.detach()], dim=0)
            all_features_loss_98=torch.cat([all_features_loss_98,outh9L8.detach()], dim=0)
            
    torch.save(all_features_net1, '%s/result_3/%s/out_features_net_test_1_%d.pickle' % (save_dir, sig, epoch))
    torch.save(all_features_net2, '%s/result_3/%s/out_features_net_test_2_%d.pickle' % (save_dir, sig, epoch))
    torch.save(all_features_net3, '%s/result_3/%s/out_features_net_test_3_%d.pickle' % (save_dir, sig, epoch))
    torch.save(all_features_net4, '%s/result_3/%s/out_features_net_test_4_%d.pickle' % (save_dir, sig, epoch))
    torch.save(all_features_net5, '%s/result_3/%s/out_features_net_test_5_%d.pickle' % (save_dir, sig, epoch))
    torch.save(all_features_net6, '%s/result_3/%s/out_features_net_test_6_%d.pickle' % (save_dir, sig, epoch))
    torch.save(all_features_net7, '%s/result_3/%s/out_features_net_test_7_%d.pickle' % (save_dir, sig, epoch))
    
    torch.save(all_features_loss1, '%s/result_3/%s/out_features_loss3_test_1_%d.pickle' % (save_dir, sig, epoch))
    torch.save(all_features_loss2, '%s/result_3/%s/out_features_loss3_test_2_%d.pickle' % (save_dir, sig, epoch))
    torch.save(all_features_loss3, '%s/result_3/%s/out_features_loss3_test_3_%d.pickle' % (save_dir, sig, epoch))
    
    torch.save(all_features_loss_91, '%s/result_3/%s/out_features_loss9_test_1_%d.pickle' % (save_dir, sig, epoch))
    torch.save(all_features_loss_91, '%s/result_3/%s/out_features_loss9_test_2_%d.pickle' % (save_dir, sig, epoch))
    torch.save(all_features_loss_93, '%s/result_3/%s/out_features_loss9_test_3_%d.pickle' % (save_dir, sig, epoch))
    torch.save(all_features_loss_94, '%s/result_3/%s/out_features_loss9_test_4_%d.pickle' % (save_dir, sig, epoch))
    torch.save(all_features_loss_95, '%s/result_3/%s/out_features_loss9_test_5_%d.pickle' % (save_dir, sig, epoch))
    torch.save(all_features_loss_96, '%s/result_3/%s/out_features_loss9_test_6_%d.pickle' % (save_dir, sig, epoch))
    torch.save(all_features_loss_97, '%s/result_3/%s/out_features_loss9_test_7_%d.pickle' % (save_dir, sig, epoch))
    torch.save(all_features_loss_98, '%s/result_3/%s/out_features_loss9_test_8_%d.pickle' % (save_dir, sig, epoch))
    
    torch.save(all_y_test, '%s/result_3/%s/out_y_test_%d.pickle' % (save_dir, sig, epoch))

    
