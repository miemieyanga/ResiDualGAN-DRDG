import torch.utils.data as D
import random
from PIL import Image
import numpy as np
from tqdm import tqdm


class DualDataset(D.Dataset):
    def __init__(self, dsa_path, dsb_path, transform_imgs=None,  transform_dsms=None, random_seed=666, in_memory=True):
        super(DualDataset, self).__init__()

        self.dsa_path = dsa_path
        self.dsb_path = dsb_path
        self.transform_imgs = transform_imgs
        self.transform_dsms = transform_dsms
        self.a_files = []
        self.b_files = []
        self.in_memory = in_memory
        
        file_path = f"{dsa_path}/all.txt"
        with open(file_path, "r") as f:
            lines = f.readlines()
        for line in lines:
            self.a_files.append(line.strip())

        file_path = f"{dsb_path}/all.txt"
        with open(file_path, "r") as f:
            lines = f.readlines()
        for line in lines:
            self.b_files.append(line.strip())
        
        self.a_imgs = []
        self.b_imgs = []
        self.a_dsms = []
        self.b_dsms = []
        if in_memory:
            for a_file_name in tqdm(self.a_files, desc=f"Loading dataset from {dsa_path}"):
                a_path = "{}/images/{}".format(self.dsa_path, a_file_name)
                a_img = np.array(Image.open(a_path))

                try:
                    a_dsm_path = "{}/dsms/{}".format(self.dsa_path, a_file_name)
                    a_dsm = np.array(Image.open(a_dsm_path))
                except:
                    a_dsm = []

                self.a_imgs.append(a_img)
                self.a_dsms.append(a_dsm)
            
            for b_file_name in tqdm(self.b_files, desc=f"Loading dataset from {dsb_path}"):
                b_path = "{}/images/{}".format(self.dsb_path, b_file_name)
                b_img = np.array(Image.open(b_path))

                try:
                    b_dsm_path = "{}/dsms/{}".format(self.dsb_path, b_file_name)
                    b_dsm = np.array(Image.open(b_dsm_path))
                except:
                    b_dsm = []

                self.b_imgs.append(b_img)
                self.b_dsms.append(b_dsm)
            
        random.seed(random_seed)

    def __getitem__(self, item):
        a_index = random.randint(0, len(self.a_files) - 1)
        b_index = random.randint(0, len(self.b_files) - 1)
        
        if self.in_memory:
            a_img = self.a_imgs[a_index]
            a_dsm = self.a_dsms[a_index]
            b_img = self.b_imgs[b_index]
            b_dsm = self.b_dsms[b_index]
        else:
            a_file_name = self.a_files[a_index]
            b_file_name = self.b_files[b_index]

            a_path = "{}/images/{}".format(self.dsa_path, a_file_name)
            b_path = "{}/images/{}".format(self.dsb_path, b_file_name)

            a_img = Image.open(a_path)
            b_img = Image.open(b_path)

            try:
                a_dsm_path = "{}/dsms/{}".format(self.dsa_path, a_file_name)
                b_dsm_path = "{}/dsms/{}".format(self.dsb_path, b_file_name)
                a_dsm = Image.open(a_dsm_path)
                b_dsm = Image.open(b_dsm_path)

            except:
                a_dsm = []
                b_dsm = []


        if self.transform_imgs:
            a_img = self.transform_imgs(a_img)
            b_img = self.transform_imgs(b_img)

        if self.transform_dsms:
            try:
                a_dsm = self.transform_dsms(a_dsm)
                b_dsm = self.transform_dsms(b_dsm)
            except:
                pass

        return a_img, b_img, a_dsm, b_dsm

    def __len__(self):
        return max(len(self.a_files), len(self.b_files))


class TransferDataset(D.Dataset):
    def __init__(self, path, transform, in_memory=False) -> None:
        super(TransferDataset, self).__init__()

        self.path = path
        self.files = []
        self.in_memory = in_memory
        file_path = f"{self.path}/all.txt"
        with open(file_path, mode="r") as f:
            lines = f.readlines()
        for line in lines:
            self.files.append(line.strip())
    
        self.len = len(self.files)
        self.transform = transform
        
        self.items = []
        if self.in_memory:
            for file_name in self.files:
                img_path = f"{self.path}/images/{file_name}"
                lbl_path = f"{self.path}/labels/{file_name}"
                img = np.array(Image.open(img_path))
                lbl = np.array(np.uint8(Image.open(lbl_path)))
                
                self.items.append((file_name, img, lbl))
        
    def __getitem__(self, index):
        if self.in_memory:
            file_name, img, lbl = self.items[index]
        else:
            file_name = self.files[index]
            img_path = f"{self.path}/images/{file_name}"
            lbl_path = f"{self.path}/labels/{file_name}"
            img = np.array(Image.open(img_path))
            lbl = np.array(np.uint8(Image.open(lbl_path)))
        img = self.transform(img)

        return file_name, img, lbl

    def __len__(self):
        return self.len