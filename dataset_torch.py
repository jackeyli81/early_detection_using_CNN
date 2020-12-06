from torch.utils.data import DataLoader, Dataset
import torch, random, numpy as np, cv2
import torchvision.transforms as transforms


class dataset_hos6(Dataset):
    def __init__(self, data_tensor, target_tensor, ds_name = 'test'):
        super(dataset_hos6, self).__init__()
        self.target_tensor = target_tensor
        if ds_name == 'train':
            self.data_tensor = np.transpose( np.array( data_tensor ), (0, 2, 3, 1) )
            # print('training shape: ', self.data_tensor.shape)
            # 2016, 415, 585, 3
        else:
            self.data_tensor = data_tensor
        self.ds_name = ds_name
        

    def __getitem__(self, index):
        if self.ds_name == 'test':
            return self.data_tensor[index,:,:,:], self.target_tensor[index]
        else:
            img = self.data_tensor[index, :, :, :]
            if random.random()>0.5:
                row, col = img.shape[:2]
                # rows,cols = random.randint(30,row-30), random.randint(30, col-30)
                M2 = cv2.getRotationMatrix2D( (col/2, row/2), random.randint(1,6)*6, 1)
                img = np.transpose( cv2.warpAffine( img, M2, (col, row) ), axes = (2, 0, 1) )
            else:
                img = np.transpose( img, axes = (2, 0, 1) )
            return torch.Tensor( img ), self.target_tensor[index]

        
        
    def __len__(self):
        return self.data_tensor.shape[0]
    def shap_(self):
        return self.data_tensor.shape