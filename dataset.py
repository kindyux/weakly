from utils import *
import matplotlib.pyplot as plt
import os
import shutil
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class TrainSetLoader(Dataset):
    def __init__(self, dataset_dir, dataset_name, label_type, patch_size, masks_update, img_norm_cfg=None):
        super(TrainSetLoader).__init__()
        # 路径的初始化，数据集指向相关的
        self.dataset_dir = dataset_dir + dataset_name
        self.patch_size = patch_size
        self.tranform = augumentation()
        self.masks_update = masks_update
        with open(self.dataset_dir+'/img_idx/train_' + dataset_name + '.txt', 'r') as f:
            self.train_list = f.read().splitlines()
        if img_norm_cfg == None:
            self.img_norm_cfg = get_img_norm_cfg(dataset_name, self.dataset_dir)
        else:
            self.img_norm_cfg = img_norm_cfg
        self.dataset_name = dataset_name

        ### ---------------------- for label update ----------------------
        #在初始化训练所用的Dataset时就把初始的mask（单点）给copy了一份保存下来
        self.label_type = label_type
        if isinstance(masks_update, str):
            if os.path.exists(masks_update):
                shutil.rmtree(masks_update)
            os.makedirs(masks_update)
            for img_idx in self.train_list:
                shutil.copyfile(self.dataset_dir + '/' + '/masks_' + self.label_type + '/' + img_idx + '.png', 
                                masks_update + '/' + img_idx + '.png')
        if isinstance(masks_update, list):            
            self.masks_update = masks_update     
            
    def __getitem__(self, idx):
        img = Image.open(self.dataset_dir + '/images/' + self.train_list[idx] + '.png').convert('I')
        # 在这里他的训练所用的Dataset就是一直从刚才初始化的时候copy的一份的路径取出的
        # 因此如果改变了这个路径下的mask，那么监督的mask就会相应地变化了！巧妙！
        if isinstance(self.masks_update, str):
            mask = Image.open(self.masks_update + '/' + self.train_list[idx] + '.png')
            mask = np.array(mask, dtype=np.float32)  / 255.0
        elif isinstance(self.masks_update, list):
            mask = self.masks_update[idx]

        #一些数据预处理和增强操作，包括正则化、随机裁剪、增强以及转为tensor
        img = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg)
        img_patch, mask_patch = random_crop(img, mask, self.patch_size)
        img_patch, mask_patch = self.tranform(img_patch, mask_patch)
        img_patch, mask_patch = img_patch[np.newaxis,:], mask_patch[np.newaxis,:]
        img_patch = torch.from_numpy(np.ascontiguousarray(img_patch))
        mask_patch = torch.from_numpy(np.ascontiguousarray(mask_patch))
        return img_patch, mask_patch
    def __len__(self):
        return len(self.train_list)

class TrainSetLoader_full(Dataset):
    def __init__(self, dataset_dir, dataset_name, patch_size, img_norm_cfg=None):
        super(TrainSetLoader_full).__init__()
        self.dataset_dir = dataset_dir + dataset_name
        self.patch_size = patch_size
        self.tranform = augumentation()
        with open(self.dataset_dir+'/img_idx/train_' + dataset_name + '.txt', 'r') as f:
            self.train_list = f.read().splitlines()
        if img_norm_cfg == None:
            self.img_norm_cfg = get_img_norm_cfg(dataset_name, self.dataset_dir)
        else:
            self.img_norm_cfg = img_norm_cfg
        self.dataset_name = dataset_name
            
    def __getitem__(self, idx):
        img = Image.open(self.dataset_dir + '/images/' + self.train_list[idx] + '.png').convert('I')
        mask = Image.open(self.dataset_dir + '/masks/' + self.train_list[idx] + '.png')
        mask = np.array(mask, dtype=np.float32)  / 255.0
        if len(mask.shape) > 2:
            mask = mask[:,:,0]
        img = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg)
        img_patch, mask_patch = random_crop(img, mask, self.patch_size)
        img_patch, mask_patch = self.tranform(img_patch, mask_patch)
        img_patch, mask_patch = img_patch[np.newaxis,:], mask_patch[np.newaxis,:]
        img_patch = torch.from_numpy(np.ascontiguousarray(img_patch))
        mask_patch = torch.from_numpy(np.ascontiguousarray(mask_patch))
        return img_patch, mask_patch
    def __len__(self):
        return len(self.train_list)

class Update_mask(Dataset):
    def __init__(self, dataset_dir, dataset_name, label_type, masks_update, img_norm_cfg=None):
        super(Update_mask).__init__()
        self.label_type = label_type
        self.masks_update = masks_update
        self.dataset_dir = dataset_dir + dataset_name
        self.dataset_name = dataset_name
        with open(self.dataset_dir+'/img_idx/train_' + dataset_name + '.txt', 'r') as f:
            self.train_list = f.read().splitlines()
        if img_norm_cfg == None:
            self.img_norm_cfg = get_img_norm_cfg(dataset_name, self.dataset_dir)
        else:
            self.img_norm_cfg = img_norm_cfg
        
    def __getitem__(self, idx):
        img = Image.open(self.dataset_dir + '/images/' + self.train_list[idx] + '.png').convert('I')
        #mask = Image.open(self.dataset_dir + '/masks/' + self.train_list[idx] + '.png')  #update mask时需要用到原mask?
        if isinstance(self.masks_update, str):
            mask_update = Image.open(self.masks_update + '/' + self.train_list[idx] + '.png')   #从copy位置处取mask
            update_dir = self.masks_update + '/' + self.train_list[idx] + '.png'
            mask_update = np.array(mask_update, dtype=np.float32)  / 255.0      #取到的mask进行归一化和转成numpy
            if len(mask_update.shape) > 2:
                mask_update = mask_update[:,:,0]
        elif isinstance(self.masks_update, list):
            mask_update = self.masks_update[idx]
            update_dir = idx


        ##到目前为止：
        # img--->原图像
        # mask--->完整mask
        # mask_update--->单点mask
        img = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg)
        #mask = np.array(mask, dtype=np.float32)  / 255.0    #完整mask也和单点mask一样进行归一化

        # if len(mask.shape) > 2:
        #     mask = mask[:,:,0]
        
        h, w = img.shape    #[256,256]
        times = 32
        #原图像和完整的mask的x轴和y轴均拓宽32个0，即[256,256]--->[288,288]
        img = np.pad(img, ((0, (h//times+1)*times-h),(0, (w//times+1)*times-w)), mode='constant')
        #mask = np.pad(mask, ((0, (h//times+1)*times-h),(0, (w//times+1)*times-w)), mode='constant')

        mask_update = np.pad(mask_update, ((0, (h//times+1)*times-h),(0, (w//times+1)*times-w)), mode='constant')
        #将img、mask和mask_update均unsqueeze一下，即[288,288]--->[1,288,288]
        #img, mask, mask_update = img[np.newaxis,:], mask[np.newaxis,:], mask_update[np.newaxis,:]
        img, mask_update = img[np.newaxis, :], mask_update[np.newaxis, :]

        #转成内存连续的数据方便转化成Tensor
        img = torch.from_numpy(np.ascontiguousarray(img))
        #mask = torch.from_numpy(np.ascontiguousarray(mask))
        mask_update = torch.from_numpy(np.ascontiguousarray(mask_update))

        # 虽然说这里把数据从[256,256]转成了[288,288]，但是返回的size还是[256,256]，明白了，那就是针对分辨率不足[256,256]的那些图而言的吧！
        #return img, mask, mask_update, update_dir, [h,w]
        return img, mask_update, update_dir, [h, w]
    def __len__(self):
        return len(self.train_list) 

class TestSetLoader(Dataset):
    def __init__(self, dataset_dir, train_dataset_name, test_dataset_name, img_norm_cfg=None):
        super(TestSetLoader).__init__()
        self.dataset_dir = dataset_dir + test_dataset_name
        with open(self.dataset_dir+'/img_idx/test_' + test_dataset_name + '.txt', 'r') as f:
            self.test_list = f.read().splitlines()
        if img_norm_cfg == None:
            self.img_norm_cfg = get_img_norm_cfg(train_dataset_name, self.dataset_dir)
        else:
            self.img_norm_cfg = img_norm_cfg
            
    def __getitem__(self, idx):
        img = Image.open(self.dataset_dir + '/images/' + self.test_list[idx] + '.png').convert('I')
        mask = Image.open(self.dataset_dir + '/masks/' + self.test_list[idx] + '.png')
        
        img = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg)
        mask = np.array(mask, dtype=np.float32)  / 255.0
        
        if len(mask.shape) > 2:
            mask = mask[:,:,0]
            
        h, w = img.shape
        times = 32
        img = np.pad(img, ((0, (h//times+1)*times-h),(0, (w//times+1)*times-w)), mode='constant')
        mask = np.pad(mask, ((0, (h//times+1)*times-h),(0, (w//times+1)*times-w)), mode='constant')
        
        img, mask = img[np.newaxis,:], mask[np.newaxis,:]
        
        img = torch.from_numpy(np.ascontiguousarray(img))
        mask = torch.from_numpy(np.ascontiguousarray(mask))
        return img, mask, [h,w], self.test_list[idx]
    def __len__(self):
        return len(self.test_list) 

class InferenceSetLoader(Dataset):
    def __init__(self, dataset_dir, train_dataset_name, test_dataset_name, img_norm_cfg=None):
        super(InferenceSetLoader).__init__()
        self.dataset_dir = dataset_dir + test_dataset_name
        with open(self.dataset_dir+'/img_idx/test_' + test_dataset_name + '.txt', 'r') as f:
            self.test_list = f.read().splitlines()
        if img_norm_cfg == None:
            self.img_norm_cfg = get_img_norm_cfg(train_dataset_name, self.dataset_dir)
        else:
            self.img_norm_cfg = img_norm_cfg
            
    def __getitem__(self, idx):
        img = Image.open(self.dataset_dir + '/images/' + self.test_list[idx] + '.png').convert('I')
        
        img = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg)
            
        h, w = img.shape
        times = 32
        img = np.pad(img, ((0, (h//times+1)*times-h),(0, (w//times+1)*times-w)), mode='constant')
        
        img = img[np.newaxis,:]
        
        img = torch.from_numpy(np.ascontiguousarray(img))
        return img, [h,w], self.test_list[idx]
    def __len__(self):
        return len(self.test_list) 

class augumentation(object):
    def __call__(self, input, target):
        if random.random()<0.5:
            input = input[::-1, :]
            target = target[::-1, :]
        if random.random()<0.5:
            input = input[:, ::-1]
            target = target[:, ::-1]
        if random.random()<0.5:
            input = input.transpose(1, 0)
            target = target.transpose(1, 0)
        return input, target
