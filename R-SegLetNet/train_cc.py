import torch, gc
import read as data
import torch.utils.data as Datas
import CCNet as Network
import metrics as criterion

from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import skimage.io as io
from tqdm import tqdm
from tensorboardX import SummaryWriter
import datetime


import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = data.train_data

dataloder = Datas.DataLoader(dataset=data,batch_size=4,shuffle=True)

writer = SummaryWriter("cc_logpath" + f'/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}/')

# fusenet = Network.FBSNet().to(device)
# fusenet = Network.LETNet().to(device)
fusenet = Network.RSegLETNet().to(device)
# fusenet = Network.SegNetwork().to(device)
# opt = torch.optim.Adam(fusenet.parameters(), lr=3e-4, betas=(0.9, 0.999), weight_decay=1e-5)
opt = torch.optim.Adam(fusenet.parameters(), lr=3e-4, betas=(0.9, 0.999), weight_decay=1e-5)
pretrained_dict = torch.load('/home/tang/Semantic-Aware-Video-Compression-for-Automotive-Cameras/Codes/CCNet/pkl/efficientPS_kitti/model/model.pth', weights_only=True)

# 检查是否存在保存的模型
# checkpoint_path = './pkl/net_epoch_last.pth'
# if os.path.exists(checkpoint_path):
#     checkpoint = torch.load(checkpoint_path)
#     fusenet.load_state_dict(checkpoint['model_state_dict'])
#     opt.load_state_dict(checkpoint['optimizer_state_dict'])
#     start_epoch = checkpoint['epoch']
#     start_step = checkpoint['step']
# else:
#     start_epoch = 0
#     start_step = 0

pretrained_num = 27
model_dict = fusenet.state_dict()
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict)
fusenet.load_state_dict(model_dict, strict=False)

criterion_CE = criterion.crossentry()
criterion_dice = criterion.DiceMeanLoss()
criterion_dice1 = criterion.DiceMeanLoss1()
criterion_iou = criterion.IOU()

# 打开文件
loss_seg_file = open("loss_seg.txt", "w")
lossseg_ce_file = open("lossseg_ce.txt", "w")
lossseg_f1_file = open("lossseg_f1.txt", "w")
meansegdice_file = open("meansegdice.txt", "w")
Mean_iou_file = open("Mean_iou.txt", "w")

total_step = 0

for epoch in range(20):
    meansegdice = 0
    meaniou = 0

    for step, (img, label) in enumerate(tqdm(dataloder)):
        
        img = img.to(device).float()
        label = label.to(device).float()
        b, c, w, h = img.shape
        segresult = fusenet(img)
        
        lossseg_ed_es = criterion_dice1(segresult, label)
        lossseg_ce = criterion_CE(segresult, label)

        if step % 100 == 0:
            segresulti = segresult[0,0,:,:]*0 + segresult[0,1,:,:]*1 + segresult[0,2,:,:]*2 + segresult[0,3,:,:]*3
            io.imsave('./seg'+str(step)+'.png', (segresulti.data.cpu().numpy() * 255).astype(np.uint8))

            segresulti = label[0,0,:,:]*0 + label[0,1,:,:]*1 + label[0,2,:,:]*2 + label[0,3,:,:]*3
            io.imsave('./label'+str(step)+'.png', (segresulti.data.cpu().numpy() * 255).astype(np.uint8))
                
            segresulti = img[0,:,:,:]
            segresulti = segresulti.data.cpu().numpy()
            segresulti = np.transpose(segresulti, (1, 2, 0))
            io.imsave('./img'+str(step)+'.png', (segresulti * 255).astype(np.uint8))

        loss = lossseg_ed_es + lossseg_ce
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 744 == 0:
            torch.save(fusenet.state_dict(), './pkl/net_epoch_' + str(epoch+pretrained_num) + '-fuseNetwork.pkl')


        # if step % 2000 == 0:
        #     torch.save({
        #         'epoch': epoch + 1,
        #         'step': step,
        #         'model_state_dict': fusenet.state_dict(),
        #         'optimizer_state_dict': opt.state_dict(),
        #     }, checkpoint_path)

        meansegdice += lossseg_ed_es.data.cpu().numpy()
        meaniou += criterion_iou(segresult, label).cpu().item()

        # print(f'EPOCH: {epoch} | Step: {step} | loss_seg: {loss.cpu().detach().numpy():.5f} | lossseg_ce: {lossseg_ce.data.cpu().detach().numpy():.5f} | lossseg_f1: {(lossseg_ed_es[0]).data.cpu().detach().numpy():.5f}')
        print('EPOCH:', epoch, '|Step:', step,
              '|loss_seg:', loss.data.cpu().numpy(),'|lossseg_ce:', lossseg_ce.data.cpu().numpy(),'|lossseg_f1:', lossseg_ed_es.data.cpu().numpy())
        
        # 写入文件
        loss_seg_file.write(f'{loss.data.cpu().numpy()}\n')
        lossseg_ce_file.write(f'{lossseg_ce.data.cpu().numpy()}\n')
        lossseg_f1_file.write(f'{lossseg_ed_es.data.cpu().numpy()}\n')

        writer.add_scalars('loss_seg', {'train': loss.data.cpu().numpy()}, total_step)
        writer.add_scalars('lossseg_ce', {'train': lossseg_ce.data.cpu().numpy()}, total_step)
        writer.add_scalars('lossseg_f1', {'train': lossseg_ed_es.data.cpu().detach().numpy()}, total_step)
        loss = 0
        total_step += 1
    
    torch.cuda.synchronize()
    num_steps = len(dataloder)
    # print(f'epoch {epoch} | meansegdice: {meansegdice / num_steps:.5f} | Mean_iou: {meaniou / num_steps:.5f}')
    print('epoch', epoch, '|meansegdice:',(meansegdice / step), '|Mean_iou:',(meaniou/ step))
    # 写入文件
    meansegdice_file.write(f'{meansegdice / step}\n')
    meansegdice_file.flush()
    Mean_iou_file.write(f'{meaniou / step}\n')
    Mean_iou_file.flush()

    writer.add_scalars('meansegdice', {'train': (meansegdice / num_steps)}, total_step)
    writer.add_scalars('Mean_iou', {'train': (meaniou / num_steps)}, total_step)
    
# 关闭文件
loss_seg_file.close()
lossseg_ce_file.close()
lossseg_f1_file.close()
meansegdice_file.close()
Mean_iou_file.close()