import argparse
import time
from torch.autograd import Variable
from torch.utils.data import DataLoader
from net import  Net
from dataset import *
import matplotlib.pyplot as plt
from metrics import *
import numpy as np
import os
from tqdm import tqdm
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

parser = argparse.ArgumentParser(description="PyTorch LESPS train")
parser.add_argument("--model_names", default=['DNANet'], nargs='+', 
                    help="model_name: 'ACM', 'ALCNet', 'DNANet'")                 
parser.add_argument("--dataset_names", default=['ICPR2024'], nargs='+',
                    help="dataset_name: 'NUAA-SIRST', 'NUDT-SIRST', 'IRSTD-1K', 'NUDT-SIRST-Sea', 'SIRST3'")
parser.add_argument("--label_type", default='centroid', type=str, help="label type: centroid, coarse")
# 一开始作者本来是想设定一个在指定训练多少轮后开始更新mask，实质是想先让模型稳定后，再根据预测来更新
# 但是后来作者更换了依据，是loss小于LESPS_Tloss时就代表模型稳定了，可以更新了。
# parser.add_argument("--LESPS_Tepoch", default=50, type=int, help="Initial evolution epoch, default: 50")
parser.add_argument("--LESPS_Tloss", default=10, type=int, help="Tb in evolution threshold, default: 0.5")
parser.add_argument("--LESPS_Tb", default=0.5, type=float, help="Tb in evolution threshold, default: 0.5")
parser.add_argument("--LESPS_k", default=0.5, type=float, help="k in evolution threshold, default: 0.5")
parser.add_argument("--LESPS_f", default=5, type=int, help="Evolution frequency, default: 5")
parser.add_argument("--img_norm_cfg", default=None, type=dict,
                    help="specific a img_norm_cfg, default=None (using img_norm_cfg values of each dataset)")

parser.add_argument("--dataset_dir", default='./datasets/', type=str, help="train_dataset_dir, default: './datasets/SIRST3")
parser.add_argument("--batchSize", type=int, default=2, help="Training batch sizse, default: 16")
parser.add_argument("--patchSize", type=int, default=256, help="Training patch size, default: 256")
parser.add_argument("--save", default='./log/ICPR', type=str, help="Save path, default: './log")
parser.add_argument("--resume", default=None, type=str, help="Resume path, default: None")
parser.add_argument("--nEpochs", type=int, default=400, help="Number of epochs, default: 400")
parser.add_argument("--lr", type=float, default=5e-4, help="Learning Rate, default: 5e-4")
parser.add_argument('--gamma', type=float, default=0.1, help='Gamma, default: 0.1')
parser.add_argument("--step", type=int, default=[200, 300], help="Sets the learning rate decayed by step, default: [200, 300]")
parser.add_argument("--threads", type=int, default=1, help="Number of threads for data loader to use, default: 1")
parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for test, default: 0.5")
parser.add_argument("--cache", default=False, type=str, help="True: cache intermediate mask results, False: save intermediate mask results")

global opt
opt = parser.parse_args()
def train():
    train_set = TrainSetLoader(dataset_dir=opt.dataset_dir, dataset_name=opt.dataset_name, label_type=opt.label_type, patch_size=opt.patchSize, masks_update=opt.masks_update, img_norm_cfg=opt.img_norm_cfg)
    train_loader = DataLoader(dataset=train_set, num_workers=1, batch_size=opt.batchSize, shuffle=True)
    
    epoch_state = 0
    total_loss_list = []
    total_loss_epoch = []
    update_epoch_loss = []
    start_click = 0
    
    net = Net(model_name=opt.model_name, mode='train').cuda()
    if opt.resume:
        ckpt = torch.load(opt.resume)
        net.load_state_dict(ckpt['state_dict'])
        epoch_state = ckpt['epoch']
        total_loss_list = ckpt['total_loss']
        for i in range(len(opt.step)):
            opt.step[i] = opt.step[i] - epoch_state

    optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.step, gamma=opt.gamma)
    
    for idx_epoch in range(epoch_state, opt.nEpochs):
        for idx_iter, (img, gt_mask) in enumerate(train_loader):
            net.train()
            img, gt_mask = Variable(img).cuda(), Variable(gt_mask).cuda()
            pred = net.forward(img)
            loss = net.loss(pred, gt_mask)
            total_loss_epoch.append(loss.detach().cpu())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
        scheduler.step()
        if (idx_epoch + 1) % 1 == 0:
            total_loss_list.append(float(np.array(total_loss_epoch).mean()))    #显示的是一个epoch中所有iter的平均loss
            print(time.ctime()[4:-5] + ' Epoch---%d, total_loss---%f,' % (idx_epoch + 1, total_loss_list[-1]))
            opt.f.write(time.ctime()[4:-5] + ' Epoch---%d, total_loss---%f,\n' % (idx_epoch + 1, total_loss_list[-1]))
            total_loss_epoch = []
        
        # first update
        # 在未更新mask之前是不进行测试的
        # if (idx_epoch + 1) > opt.LESPS_Tepoch and start_click == 0:
        if total_loss_list[-1] < opt.LESPS_Tloss and start_click == 0:
            print('start update gt masks')
            start_click = 1
            save_pth = opt.save + '/' + opt.dataset_name + '/' + opt.save_perdix + '_' + str(idx_epoch + 1) + '.pth.tar'
            # save_checkpoint({
            #     'epoch': idx_epoch + 1,
            #     'state_dict': net.state_dict(),
            #     'total_loss': total_loss_list,
            #     'train_iou_list': opt.train_iou_list,
            #     'test_iou_list': opt.test_iou_list,
            #     }, save_pth)
            save_checkpoint({
                'epoch': idx_epoch + 1,
                'state_dict': net.state_dict(),
                'total_loss': total_loss_list,
                'test_iou_list': opt.test_iou_list,
            }, save_pth)
            update_gt_mask(save_pth, thresh_Tb=opt.LESPS_Tb, thresh_k=opt.LESPS_k)
            #test(save_pth)
            update_epoch_loss.append(total_loss_list[-1])
            
        # subsequent update
        if start_click == 1 and (idx_epoch + 1) % opt.LESPS_f == 0:
            print('update gt masks')
            save_pth = opt.save + '/' + opt.dataset_name + '/' + opt.save_perdix + '_' + str(idx_epoch + 1) + '.pth.tar'
            # save_checkpoint({
            #     'epoch': idx_epoch + 1,
            #     'state_dict': net.state_dict(),
            #     'total_loss': total_loss_list,
            #     'train_iou_list': opt.train_iou_list,
            #     'test_iou_list': opt.test_iou_list,
            #     }, save_pth)
            save_checkpoint({
                'epoch': idx_epoch + 1,
                'state_dict': net.state_dict(),
                'total_loss': total_loss_list,
                'test_iou_list': opt.test_iou_list,
            }, save_pth)
            update_gt_mask(save_pth, thresh_Tb=opt.LESPS_Tb, thresh_k=opt.LESPS_k)
            #test(save_pth)
            update_epoch_loss.append(total_loss_list[-1])

def update_gt_mask(save_pth, thresh_Tb, thresh_k):

    # 这里为什么需要新建一个dataloader呢？因为需要得到当前训练的模型的预测结果啊！
    update_set = Update_mask(dataset_dir=opt.dataset_dir, dataset_name=opt.dataset_name, label_type=opt.label_type, masks_update=opt.masks_update, img_norm_cfg=opt.img_norm_cfg)
    update_loader = DataLoader(dataset=update_set, num_workers=1, batch_size=1, shuffle=False)
    
    net = Net(model_name=opt.model_name, mode='test').cuda()
    ckpt = torch.load(save_pth)
    net.load_state_dict(ckpt['state_dict'])
    net.eval()
    
    # eval_IoU = mIoU()
    #for idx_iter, (img, gt_mask, gt_mask_update, update_dir, size) in tqdm(enumerate(update_loader)):
    for idx_iter, (img, gt_mask_update, update_dir, size) in tqdm(enumerate(update_loader)):

        img, gt_mask_update = Variable(img).cuda(), Variable(gt_mask_update).cuda()
        # 拿到预测结果
        pred = net.forward(img)
        print(size)
        pred = pred[:,:,:size[0],:size[1]]  # 取预测的所有batch的所有通道（都是1）的左上[256,256]
        #gt_mask = gt_mask[:,:,:size[0],:size[1]]    # 取完整mask的所有batch的所有通道（都是1）的左上[256,256]
        gt_mask_update = gt_mask_update[:,:,:size[0],:size[1]]  # 取单点mask的所有batch的所有通道（都是1）的左上[256,256]

        # 哈哈 果然 这里update_gt操作没有用到完整的mask，那至于为什么要取出完整的mask呢？
        # 是因为作者这里在更新完mask后，测评了一下更新后的masks和完整的masks之间的mIoU和Pixel_Acc！！！
        # 这样做的目的应该是在观察和记录所提出的更新方式是否有效！！！在已证明有效的情况下实际训练的时候是不需要这一步操作的
        update_mask = net.update_gt(pred, gt_mask_update, thresh_Tb, thresh_k, size)

        if isinstance(update_dir, torch.Tensor):
            opt.masks_update[update_dir] = update_mask[0,0,:size[0],:size[1]].cpu().detach().numpy()
        else:
            img_save = transforms.ToPILImage()((update_mask[0,:,:size[0],:size[1]]).cpu())
            img_save.save(update_dir[0])
        #eval_IoU.update((update_mask>=opt.threshold).cpu(), gt_mask)
        
    #results1 = eval_IoU.get()
    # opt.f.write("Evolution mask pixAcc, mIoU:\t" + str(results1) + '\n')
    # print("Evolution mask pixAcc, mIoU:\t" + str(results1))
    # opt.train_iou_list.append(results1[1])

def test(save_pth):
    test_set = TestSetLoader(opt.dataset_dir, opt.dataset_name, opt.dataset_name, opt.img_norm_cfg)
    test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)
    
    net = Net(model_name=opt.model_name, mode='test').cuda()
    ckpt = torch.load(save_pth)
    net.load_state_dict(ckpt['state_dict'])
    net.eval()
    
    eval_mIoU = mIoU() 
    eval_PD_FA = PD_FA()
    for idx_iter, (img, gt_mask, size, _) in enumerate(test_loader):
        img = Variable(img).cuda()
        pred = net.forward(img)
        pred = pred[:,:,:size[0],:size[1]]
        gt_mask = gt_mask[:,:,:size[0],:size[1]]
        eval_mIoU.update((pred>opt.threshold).cpu(), gt_mask)
        eval_PD_FA.update((pred[0,0,:,:]>opt.threshold).cpu(), gt_mask[0,0,:,:], size)     
    
    results1 = eval_mIoU.get()
    results2 = eval_PD_FA.get()
    print("Inference mask pixAcc, mIoU:\t" + str(results1))
    print("Inference mask PD, FA:\t" + str(results2))
    opt.f.write("pixAcc, mIoU:\t" + str(results1) + '\n')
    opt.f.write("PD, FA:\t" + str(results2) + '\n')

def save_checkpoint(state, save_path):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    torch.save(state, save_path)

if __name__ == '__main__':
    for dataset_name in opt.dataset_names:
        opt.dataset_name = dataset_name
        for model_name in opt.model_names:
            opt.model_name = model_name
            opt.save_perdix = opt.model_name + '_LESPS_' + opt.label_type
            
            if opt.cache:
                ### cache intermediate mask results
                with open(opt.dataset_dir+'/img_idx/train_' + opt.dataset_name + '.txt', 'r') as f:
                    train_list = f.read().splitlines()
                opt.masks_update = []
                for idx in range(len(train_list)):
                    mask = Image.open(opt.dataset_dir + '/masks_' + opt.label_type + '/' + train_list[idx] + '.png')
                    mask = np.array(mask, dtype=np.float32)  / 255.0
                    opt.masks_update.append(mask)
            else:
                ### save intermediate mask results to 
                opt.masks_update = opt.dataset_dir + '/' + opt.dataset_name + '_' + opt.save_perdix + '_' + (time.ctime()).replace(' ', '_').replace(':','_')

            ### save intermediate loss vaules
            if not os.path.exists(opt.save):
                os.makedirs(opt.save)
            opt.f = open(opt.save + '/' + opt.dataset_name + '_' + opt.model_name + '_LESPS_' + opt.label_type + '_' + (time.ctime()).replace(' ', '_').replace(':', '_') + '.txt', 'w')

            # opt.train_iou_list = []
            opt.test_iou_list = []
            
            print(opt.dataset_name + '\t' + opt.model_name)
            train()
            print('\n')
            opt.f.close()
