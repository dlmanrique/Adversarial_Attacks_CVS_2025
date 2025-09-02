import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import random
import json
import pandas as pd
from torchvision import transforms

g = torch.Generator()
g.manual_seed(5)

def get_datasets(config, args):
    """
    Check dataset exists (download if not). Create dataset instances and apply transformations specified in config
    """
    dataset_dir = config.DATASET_DIR
    
    print(f"\nDataset loaded from: {dataset_dir}")


    train_dataframe, val_dataframe, test_dataframe = get_three_dataframes(dataset_dir, config, args)
    transform_sequence = get_transform_sequence(config)
    
    print(f'Number of keyframes on train split: {len(train_dataframe)}')
    print(f'Number of keyframes on valid split: {len(val_dataframe)}')
    print(f'Number of keyframes on test split: {len(test_dataframe)}')

    # If just SwinV2 backbone
    training_dataset = Endoscapes_Dataset(train_dataframe[::config.TRAIN.LIMIT_DATA_FRACTION], transform_sequence)
    val_dataset = Endoscapes_Dataset(val_dataframe[::config.TRAIN.LIMIT_DATA_FRACTION], transform_sequence)
    test_dataset = Endoscapes_Dataset(test_dataframe[::config.TRAIN.LIMIT_DATA_FRACTION], transform_sequence)


    return training_dataset, val_dataset, test_dataset


def get_dataloaders(config, train_dataset, valid_dataset, test_dataset):
    """
    Create dataloaders from a given training datasets
    """
    print(f"Batch size: {config.TRAIN.BATCH_SIZE}")

    train_dataloader = DataLoader(  train_dataset,
                                        batch_size = config.TRAIN.BATCH_SIZE,
                                        shuffle = True,
                                        pin_memory = True,
                                        generator=g)
    
    valid_dataloader = DataLoader(  valid_dataset,
                                    batch_size = 1,
                                    shuffle = False,
                                    pin_memory = True)
    
    test_dataloader = DataLoader(  test_dataset,
                                    batch_size = 1,
                                    shuffle = False,
                                    pin_memory = True)
    return train_dataloader, valid_dataloader, test_dataloader


def get_dataframe_test(config, args):
    """
    Get images from the dataset directory, create pandas dataframes of image filepaths and ground truths. 
    """

    test_file = f'format_challenge_data/Sages_fold{config.FOLD}_train_data.json'
    test_dataframe = get_dataframe(test_file, args)
    updated_test_dataframe = update_dataframe(test_dataframe, '../Dataset', config)
    return updated_test_dataframe

def get_three_dataframes(image_folder, config, args):

    # Mapear rutas según tipo de frame
    train_paths = {
        "Original": f"format_challenge_data/Sages_fold{config.FOLD}_train_data.json",
        "Preprocessed": f"format_challenge_data/preprocessed_data/Fold{config.FOLD}/train.json",
        "80:20": 'format_challenge_data/80:20_splits/train.json',
        'Adv_Extended': f"format_challenge_data/adversarial_training_files/fold{config.FOLD}_train.json"
    }

    test_paths = {
        "Original": f"format_challenge_data/Sages_fold{config.FOLD}_test_data.json",
        "Preprocessed": f"format_challenge_data/preprocessed_data/Fold{config.FOLD}/test.json",
        "80:20": 'format_challenge_data/80:20_splits/test.json',
        "Adv_Extended": f"format_challenge_data/Sages_fold{config.FOLD}_test_data.json",
    }

    if args.adv_attack == 'No':
        key_files = 'Adv_Extended'
    else:
        key_files = 'Original'

    
    # Seleccionar según args
    train_file = train_paths.get(key_files)
    val_file = test_paths.get(key_files)
    test_file = test_paths.get(key_files)

    train_dataframe = get_dataframe(train_file)
    val_dataframe = get_dataframe(val_file)
    test_dataframe = get_dataframe(test_file)

    updated_train_dataframe = update_dataframe(train_dataframe, config.DATASET_DIR, config, args, 'train')
    updated_val_dataframe = update_dataframe(val_dataframe, config.DATASET_DIR, config, args, 'val')
    updated_test_dataframe = update_dataframe(test_dataframe, config.DATASET_DIR, config, args, 'test')

    return updated_train_dataframe, updated_val_dataframe, updated_test_dataframe



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
        #print(f"Idx: {idx} -- Name: {'/'.join(image_path.split('/')[-2:])})")
        return image, label




def get_dataframe(json_path):
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

        try:
            C1_i = round(i['ds'][0]) #Con esto tienen en cuenta el tema de 3 anotadores
            C2_i = round(i['ds'][1])
            C3_i = round(i['ds'][2])
        except:
            C1_i = round(i['ds'][0][0])
            C2_i = round(i['ds'][0][1])
            C3_i = round(i['ds'][0][2])

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




def update_dataframe(dataframe, image_folder, config, args, split):
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
        if split == 'val' or split == 'test' or split == 'train':
            image_folder_clean = os.path.join(image_folder, 'frames')
            df_clean = dataframe.copy()
            df_clean["path"] = df_clean.apply(
                lambda row: generate_path_sages(row, image_folder_clean), axis=1
            )
            dataframe = df_clean

        else:
            image_folder_clean = os.path.join(image_folder, 'frames')
            image_folder_adv = os.path.join(image_folder, 'frames_adv')

            df_clean = dataframe.copy()
            df_clean["path"] = df_clean.apply(
                lambda row: generate_path_sages(row, image_folder_clean), axis=1
            )

            # Copia para adversarial
            df_adv = dataframe.copy()
            df_adv["path"] = df_adv.apply(
                lambda row: generate_path_sages(row, image_folder_adv), axis=1
            )

            # Concatenar ambos
            dataframe = pd.concat([df_clean, df_adv], ignore_index=True)

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
    transform_sequence = transforms.Compose([transforms.CenterCrop(config.TRAIN.TRANSFORMS.CENTER_CROP),
                                                transforms.Resize((384, 384)),
                                                transforms.ToTensor(),
                                                transforms.Normalize(
                                                    mean=torch.tensor(mean),
                                                    std=torch.tensor(std))])
    return transform_sequence