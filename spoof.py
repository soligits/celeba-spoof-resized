from torch.utils.data import Dataset
import gdown
import csv
import os
import zipfile
from PIL import Image


class SpoofDataset(Dataset):
    link = 'https://drive.google.com/file/d/1VVIt4wuF1CW381GhzFZ9lvODECqI9EXl/view?usp=sharing'
    csv_fieldnames = ['index', 'train', 'spoof', 'class', 'path']

    def __init__(self, root, train=True, download=False, transform=None, target_transform=None, verbose=False, chosen_classes=None, train_split=0.8):
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.verbose = verbose
        if isinstance(chosen_classes, int):
            chosen_classes = [chosen_classes]
        self.chosen_classes = chosen_classes
        
        if download:
            self._download_and_extract()

        self.data, self.targets, self.classes = self._load_data()
        self.n_classes = len(self.classes)
    
    def _download_and_extract(self):
        if not os.path.exists(self.root):
            os.makedirs(self.root)
        
        file_path = os.path.join(self.root, 'celeba-spoof-resized.zip')
        if not os.path.exists(file_path):
            gdown.download(self.link, file_path, fuzzy=True, quiet=not self.verbose)
        
        self.data_path = os.path.join(self.root, 'celeba-spoof-resized')
        if not os.path.exists(self.data_path):
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(self.root)
    

    def _load_data(self):
        csv_path = os.path.join(self.root, 'info.csv')
        info_dict = {}
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f, fieldnames=self.csv_fieldnames)
            for row in reader:
                if row['class'] not in info_dict:
                    info_dict[row['class']] = []
                info_dict[row['class']].append(
                    {
                        'path': row['path'],
                        'train': row['train'],
                        'spoof': row['spoof']
                    }
                )
        
        data = []
        targets = []
        classes = sorted(list(info_dict.keys()))
        if self.verbose:
            print('Number of classes: ', len(classes))
        for cls in classes:
            if self.chosen_classes is not None and cls not in self.chosen_classes:
                continue
            split = int(len(info_dict[cls]) * self.train_split)
            rows_split = info_dict[cls][:split] if self.train else info_dict[cls][split:]
            for row in rows_split:
                data.append(row['path'])
                targets.append(int(row['spoof']))
        
        return data, targets, classes

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.root, self.data[index]))
        target = self.targets[index]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return img, target
    
        

        