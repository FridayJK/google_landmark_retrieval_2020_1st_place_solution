import os
from torch.utils.data import Dataset, DataLoader



class trainset(Dataset):
    def __init__(self, root_path, data_list, loader = None):
        with open(os.path.join(root_path, data_list), "rt") as f:
            self.data_list = f.readlines()
        self.root_path = root_path
        self.data_loader = loader

    def __getitem__(self, index):
        filename, label = self.data_list[index].split(" ")
        img_tensor = self.data_loader(os.path.join(self.root_path, filename))
        target = int(label.strip())

        return img_tensor, target
    
    def __len__(self):
        return len(self.data_list)

class testset(Dataset):
    def __init__(self, root_path, data_list, loader = None):
        self.data_list = data_list
        self.root_path = root_path
        self.data_loader = loader

    def __getitem__(self, index):
        filename   = self.data_list[index]
        filename   = filename[0] +"/"+ filename[1] +"/"+ filename[2] +"/"+ filename + ".jpg"
        img_tensor = self.data_loader(os.path.join(self.root_path, filename))

        return img_tensor
    
    def __len__(self):
        return len(self.data_list)