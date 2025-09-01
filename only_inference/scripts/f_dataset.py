import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import random
import json
import pandas as pd
from torchvision import transforms


def get_datasets(config, args):
    """
    Check dataset exists (download if not). Create dataset instances and apply transformations specified in config
    """

    dataset_dir = config.DATASET_DIR
    
    print(f"\nDataset loaded from: {dataset_dir}")

    test_dataframe = get_dataframe_test(config, args)
    transform_sequence = get_transform_sequence(config)
    
    # If just SwinV2 backbone
    test_dataset = Endoscapes_Dataset(test_dataframe, transform_sequence)

    return test_dataset


def get_dataloaders(config, test_dataset):
    """
    Create dataloaders from a given training datasets
    """
    print(f"Batch size: {config.TRAIN.BATCH_SIZE}")

    test_dataloader = DataLoader(  test_dataset,
                                    batch_size = 1,
                                    shuffle = False,
                                    pin_memory = True)
    return test_dataloader


def get_dataframe_test(config, args):
    """
    Get images from the dataset directory, create pandas dataframes of image filepaths and ground truths. 
    """

    test_file = f'format_challenge_data/Sages_fold{config.FOLD}_train_data.json'
    test_dataframe = get_dataframe(test_file, args)
    updated_test_dataframe = update_dataframe(test_dataframe, '../Dataset', config)
    return updated_test_dataframe


class Endoscapes_Dataset(Dataset):
    """
    Dataset creator only for backbone - SwinV2 training.
    """
    def __init__(self, image_dataframe, transform_sequence):
        self.image_dataframe = image_dataframe
        self.transforms = transform_sequence
        
    def __len__(self):
        return len(self.image_dataframe)
    
    def __getitem__(self, idx):
        image_info = self.image_dataframe.iloc[idx]
        image_path = image_info['path']
        label = torch.tensor(image_info['classification'])
        video_frame_id = '/'.join(image_path.split('/')[-2:])

        image = Image.open(image_path)
        
        if self.transforms:
            image = self.transforms(image)
            image = (image-torch.min(image)) / (-torch.min(image)+torch.max(image)) #Normalize the image in the interval (0,1)
      
        return image, label, video_frame_id




def get_dataframe(json_path, args):
    """
    Get dataframes of the dataset splits in columns:
    idx | vid | frame | C1 | C2 | C3
    """

    with open(json_path, 'r') as file:
        data = json.load(file)
    vid = []
    frame = []
    C1 = []
    C2 = []
    C3 = []

    # Condicionamos para extraer solamnete ejemplos que sean [1,1,1], que tengan al menos un 1 o pues todos
    
    for i in data['images']:
        # Extract data
        file_name = i['file_name']
        file_name = file_name.split('.')[0]
        file_name = file_name.split('_')
        vid_i = file_name[0]
        frame_i = file_name[1]
        C1_i = round(i['ds'][0]) #Con esto tienen en cuenta el tema de 3 anotadores
        C2_i = round(i['ds'][1])
        C3_i = round(i['ds'][2])

        if args.adversarial_split == 'CVS_only' and (C1_i + C2_i + C3_i) == 3:
            # Put in list
            vid.append(vid_i)
            frame.append(frame_i)
            C1.append(C1_i)
            C2.append(C2_i)
            C3.append(C3_i)
        
        elif args.adversarial_split == 'One_only' and (C1_i + C2_i + C3_i) >= 1:
            # Put in list
            vid.append(vid_i)
            frame.append(frame_i)
            C1.append(C1_i)
            C2.append(C2_i)
            C3.append(C3_i)
        
        elif args.adversarial_split == 'All':
            # Put in list
            vid.append(vid_i)
            frame.append(frame_i)
            C1.append(C1_i)
            C2.append(C2_i)
            C3.append(C3_i)

    data_dict = {'vid': vid,
                'frame': frame,
                'C1': C1,
                'C2': C2,
                'C3': C3}
    data_dataframe = pd.DataFrame(data_dict)
    return data_dataframe




def update_dataframe(dataframe, image_folder, config):
    """
    Function only for creation of dataframes when training backbone - SwinV2. It changes the structure of the dataframe from:
    idx | vid | frame | C1 | C2 | C3
    to:
    idx | path | classification
    where path is a path to a given image and classification is a list of ground truth values for C1-3 as, [C1, C2, C3] e.g. [0.0, 0.0, 1.0] 
    """
    
    
    if config.DATASET == 'Endoscapes':
        dataframe['path'] = dataframe.apply(lambda row: generate_path(row, image_folder), axis=1)
    elif config.DATASET == 'Sages':
        image_folder = os.path.join(image_folder, 'frames')
        dataframe['path'] = dataframe.apply(lambda row: generate_path_sages(row, image_folder), axis=1)

    dataframe['classification'] = dataframe.apply(lambda row: get_class(row), axis=1)
    dataframe.drop(columns=['vid', 'frame', 'C1', 'C2', 'C3'], inplace=True)
    dataframe.reset_index(drop=True, inplace=True)
    return dataframe


def generate_path(row, image_folder):
    vid = row['vid']
    frame = row['frame']
    filename = str(vid) + '_' + str(frame) + '.jpg'
    path = os.path.join(image_folder, filename)
    return str(path)

def generate_path_sages(row, image_folder):

    vid = row['vid']
    frame = row['frame']
    filename = 'video_' + str(vid).zfill(3) + '/' + str(frame).zfill(5) + '.jpg'
    path = os.path.join(image_folder, filename)
    return str(path)

def get_class(row):
    classification = [float(row['C1']), float(row['C2']), float(row['C3'])]
    return classification

def get_endoscapes_mean_std(config):
    mean = config.TRAIN.TRANSFORMS.ENDOSCAPES_MEAN
    std = config.TRAIN.TRANSFORMS.ENDOSCAPES_STD
    return mean, std


def get_transform_sequence(config):
    mean, std = get_endoscapes_mean_std(config)
    transform_sequence = transforms.Compose([   transforms.CenterCrop(config.TRAIN.TRANSFORMS.CENTER_CROP),
                                                transforms.Resize((384, 384)),
                                                transforms.ToTensor(),
                                                transforms.Normalize(
                                                    mean=torch.tensor(mean),
                                                    std=torch.tensor(std))])
    return transform_sequence