#!/usr/bin/env python3
#to load data

import os

# Get the absolute path of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

import os
import random
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import itertools

def get_min_max_lines(folder_path):
    min_lines = float('inf')
    max_lines = 0

    # Construct the absolute path: had to do this as with jobs it code was unable to find the files
    folder_path = os.path.join(script_dir, folder_path)
    print(folder_path)
    # Resolve the absolute path
    folder_path = os.path.abspath(folder_path)
    print(folder_path)
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path) and file_name.endswith(".txt"):
            with open(file_path, "r") as file:
                num_lines = sum(1 for line in file)
                min_lines = min(min_lines, num_lines)
                if min_lines == 0:
                    print(file_name)
                max_lines = max(max_lines, num_lines)

    return min_lines, max_lines

def load_dataset(folder_path, num_lines=51000):
    dataset = []
    labels = []
    len_lines = 0
    folder_path = os.path.join(script_dir, folder_path)
    print(folder_path)
    # Resolve the absolute path
    folder_path = os.path.abspath(folder_path)
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path) and file_name.endswith(".txt"):
            with open(file_path, "r") as file:
                lines = file.readlines()
                # random.shuffle(lines)  # Shuffle the lines randomly
                # lines = lines[:num_lines] if len(lines) >= num_lines else lines + ["0 0 0\n"] * (num_lines - len(lines))
                if len(lines) >= num_lines:
                    selected_indices = sorted(random.sample(range(len(lines)), num_lines))
                    selected_lines = [lines[i] for i in selected_indices]
                    lines = selected_lines
                else:
                    lines = lines + ["0 0 0\n"] * (num_lines - len(lines))
                points = []
                for line in lines:
                    point = list(map(float, line.strip().split()))  # Convert each line to a list of floats
                    
                    # Apply normalization
                    point[0] /= 200.0  # Divide x by 200
                    point[1] /= 200.0  # Divide y by 200
                    point[2] /= 1500.0  # Divide z by 1500
                    
                    points.append(point)
                dataset.append(points)
                labels.append(file_name.split(".")[0])  # Assuming the file name represents the label

    dataset = np.array(dataset)
    # Convert target labels to numerical values
    label_encoder = LabelEncoder()
    targets = label_encoder.fit_transform(labels)
    return dataset, np.array(targets), labels

class DatasetSplitter:
    def __init__(self, validation_ratio=0.2, shuffle=True):
        self.validation_ratio = validation_ratio
        self.shuffle = shuffle
        
    def reset_targets(self, targets, last_class_index=0):
         # Determine the actual classes present in the training data
        true_targets = np.unique(targets)
        # Map the target labels to class indices for training data
        class_mapping = {class_label: class_index for class_index, class_label in enumerate(true_targets, start=last_class_index)}
        mapped_targets = np.array([class_mapping[label] for label in targets])
        
        return mapped_targets, true_targets, class_mapping

    # Function for strong generalization to leave one out
    def split_dataset_by_index(self, dataset, targets, val_target = 0, gender = False):
        num_files, num_lines, num_dimensions = dataset.shape
        # get indices of samples
        indices = np.arange(num_files)
        
        # get the actual index by validation value
        if gender:
            # leave one out target is index
            val_index = np.array([val_target]) 
        else:
            # assigned file value is matched to leave one out index
            val_index = np.where(targets == val_target)[0]
        # Split the dataset and targets into training and validation sets
        train_data = dataset[np.setdiff1d(indices, val_index), :, :]
        train_targets = targets[np.setdiff1d(indices, val_index)]
        val_data = dataset[val_index,:, :]
        val_targets = targets[val_index]
        
        if gender:
            return train_data,train_targets, val_data, np.array(val_targets).reshape(-1)
        
        train_mapped_targets, true_train_targets, train_class_mapping = self.reset_targets(train_targets)
        # Getlast class index of the training data
        train_last_class_index = len(true_train_targets)

        val_mapped_targets, _, val_class_mapping = self.reset_targets(val_targets,train_last_class_index)
        return train_data,train_mapped_targets, val_data, val_mapped_targets, train_class_mapping, val_class_mapping

    # Function to split data based on dimension.
    # Dimension 1 is to split point clouds for train and validation
    def split_dataset(self, dataset, targets,dimension=1):
        num_files, num_lines, num_dimensions = dataset.shape

        # Shuffle the indices along the num_lines axis if shuffle is enabled
        if dimension == 0:
            indices = np.arange(num_files)
            if self.shuffle:
                np.random.shuffle(indices)
            # Calculate the number of sample to include in the validation set
            validation_data = int(num_files * self.validation_ratio)
        elif dimension == 1:
            indices = np.arange(num_lines)
            if self.shuffle:
                np.random.shuffle(indices)
            else: 
                print('No shuffle')
            # Calculate the number of lines to include in the validation set
            validation_data = int(num_lines * self.validation_ratio)
        
        if self.shuffle:
            val_indices = np.random.choice(indices, size=validation_data, replace=False)
        else:
            print('Yes')
            start_of_val_indices = np.random.choice(indices)
            val_indices = indices[start_of_val_indices:start_of_val_indices + validation_data]
        print(val_indices)
        if dimension == 0:
            # Split the dataset and targets into training and validation sets
            train_data = dataset[np.setdiff1d(indices, val_indices), :, :]
            train_targets = targets[np.setdiff1d(indices, val_indices)]
            val_data = dataset[val_indices,:, :]
            val_targets = targets[val_indices]
            
            train_mapped_targets, true_train_targets,_ = self.reset_targets(self, train_targets)
            # Determine the last class index used in the training data
            train_last_class_index = len(true_train_targets)

            val_mapped_targets,_,_ = self.reset_targets(self, val_targets,train_last_class_index)
            
            train_targets = train_mapped_targets
            val_targets = val_mapped_targets
        elif dimension == 1:
            # Split the dataset and targets into training and validation sets
            train_data = dataset[:, np.setdiff1d(indices, val_indices), :]
            train_targets = targets
            val_data = dataset[:, val_indices, :]
            val_targets = targets
        
        return train_data,train_targets, val_data, val_targets

# Custom Dataloader for the contrastive learning
class CustomDataLoader:
    def __init__(self, data, targets, batch_size=32, num_batches=100, subsample_size = 800, dimension=0, shuffle=True, augmentation_by_random_bodypart=False, augmentation_by_cube=False, visualize_cube = False):
        self.data = data
        self.targets = self.targets = torch.tensor(targets.astype(np.int32))
        self.batch_size = batch_size
        self.subsample_size = subsample_size
        self.dimension = dimension
        self.shuffle = shuffle
        self.num_samples = data.shape[dimension]
        self.num_batches = num_batches
        self.curr_batch = 0
        # self.sub_sample_shuffle = sub_sample_shuffle
        self.augmentation_by_random_bodypart = augmentation_by_random_bodypart
        self.augmentation_by_cube = augmentation_by_cube
        self.visualize_cube = visualize_cube
        self.shufflecall()
        
    # Randomly choose the labels to form batches by batch size
    def shufflecall(self):
        if self.shuffle:
            self.labels = torch.randint(self.num_samples, size=(self.num_batches, self.batch_size))
        else:
            self.labels = torch.arange(self.num_samples).repeat(self.num_batches, self.batch_size)
    
    def __num_batches__(self):
        return int(np.ceil(self.num_samples / self.batch_size))
    
    def __getitem_bybatchindex__(self,batch_index):
        indices = self.indices[batch_index]
        if self.dimension == 0:
            batch_data = self.data[indices]
            batch_targets = self.targets[indices]
        elif self.dimension == 1:
            batch_data = self.data[:, indices, :]
            batch_targets = self.targets
        elif self.dimension == 2:
            batch_data = self.data[:, :, indices]
            batch_targets = self.targets
        else:
            raise ValueError("Invalid dimension value. Must be 0, 1, or 2.")
       
        return batch_data, batch_targets

    def augmentation_by_random_bodypart_subsample(self, num_subsamples,subsample_indices,augment_lenght):
        # print(augment_lenght, num_subsamples)
        
        start_of_variant_indices = np.random.choice(range(num_subsamples))
        end_of_variant_indices = start_of_variant_indices + augment_lenght
        if end_of_variant_indices > num_subsamples:
            # Calculate the number of indices that go beyond the end
            excess_indices = end_of_variant_indices - num_subsamples

            # Create variant_b_indices by wrapping around to the beginning
            delete_variant_indices = np.concatenate((
                subsample_indices[start_of_variant_indices:],
                subsample_indices[:excess_indices]
            ))
        else:
            delete_variant_indices = subsample_indices[start_of_variant_indices:end_of_variant_indices]
        partial_indices_array = np.delete(subsample_indices, delete_variant_indices)
        variant_indices = np.random.choice(partial_indices_array, self.subsample_size, replace=False)
        return variant_indices

    def augmentation_by_cube_subsample(self, temp_data, num_subsamples,subsample_indices,augment_lenght):
        # get the bigger box
        min_x, max_x = np.min(temp_data[:, 0]), np.max(temp_data[:, 0])
        min_y, max_y = np.min(temp_data[:, 1]), np.max(temp_data[:, 1])
        min_z, max_z = np.min(temp_data[:, 2]), np.max(temp_data[:, 2])
        
        # get the random point within the box
        random_x = random.uniform(min_x, max_x)
        random_y = random.uniform(min_y, max_y)
        random_z = random.uniform(min_z, max_z)
        
        random_point = (random_x, random_y, random_z)
        small_cube_distance = 0.1 # initial distance
        # randomly subsample the point clouds by random size
        subsample_size =  np.random.choice(range(self.subsample_size, num_subsamples + 1), size=1)[0]
        
        # print(num_subsamples,subsample_size)
        
        variant_indices = random.sample(range(num_subsamples), subsample_size)
        subsampled_data =  temp_data[variant_indices]
        
        while True:
            # form the smaller box
            small_cube_min = (random_x - small_cube_distance, random_y - small_cube_distance, random_z - small_cube_distance)
            small_cube_max = (random_x + small_cube_distance, random_y + small_cube_distance, random_z + small_cube_distance)

            mask = (temp_data[:, 0] >= small_cube_min[0]) & (temp_data[:, 0] <= small_cube_max[0]) & \
           (temp_data[:, 1] >= small_cube_min[1]) & (temp_data[:, 1] <= small_cube_max[1]) & \
           (temp_data[:, 2] >= small_cube_min[2]) & (temp_data[:, 2] <= small_cube_max[2])

            # Use the mask to extract the points within the cube
            points_inside_cube = temp_data[mask]

            # Count the number of points inside the cube
            num_points_inside_cube = len(points_inside_cube)

            # Print the count
            # print("Number of points inside the cube:", num_points_inside_cube)
            
            if (num_points_inside_cube > self.subsample_size):
                break
            if num_points_inside_cube < (self.subsample_size*0.5):
                small_cube_distance = small_cube_distance + small_cube_distance * 1.5  # Increase the learning rate
            else:
                small_cube_distance = small_cube_distance + small_cube_distance * 0.5  # Decrease the learning rate
        # Get visualization for the cube cut
        if self.visualize_cube:
            sampled_points = points_inside_cube
            return sampled_points, random_point, small_cube_min, small_cube_max
        # Subsample exact points
        elif len(points_inside_cube) > self.subsample_size:
            # Randomly sample self.subsample_size points
            sampled_indices = np.random.choice(len(points_inside_cube), self.subsample_size, replace=False)
            sampled_points = points_inside_cube[sampled_indices]
        else:
            # If there are self.subsample_size, keep all of them
            sampled_points = points_inside_cube        
        return sampled_points
        
        
    #for contrastive learning getting augemented data and form matrices
    def __getitem__(self, batch_index):
        batch = ([], [])
        # print(self.data[0].shape[0])
        if self.augmentation_by_random_bodypart:
            augment_lenght = np.random.choice(range(self.data[0].shape[0]//5))
        elif self.augmentation_by_cube:
            augment_length = 1024 # Keep it constant

        
        # Get label for each batch
        for label in self.labels[batch_index]:
            indices = torch.nonzero(self.targets == label).squeeze()
            # print('---',indices)
            temp_data = self.data[indices]
            # return temp_data
            num_subsamples = temp_data.shape[0]
            # print(num_subsamples)
            subsample_indices = np.arange(num_subsamples)
            if self.augmentation_by_random_bodypart:
                variant_a_indices = self.augmentation_by_random_bodypart_subsample(num_subsamples,subsample_indices,augment_lenght)
                variant_b_indices = self.augmentation_by_random_bodypart_subsample(num_subsamples,subsample_indices,augment_lenght)
            elif self.augmentation_by_cube and self.visualize_cube:
                variant, random_point, small_cube_min, small_cube_max = self.augmentation_by_cube_subsample(temp_data,num_subsamples,subsample_indices,augment_length)
                return variant, random_point, small_cube_min, small_cube_max, temp_data
            elif self.augmentation_by_cube and not self.visualize_cube:
                variant_a = self.augmentation_by_cube_subsample(temp_data,num_subsamples,subsample_indices,augment_length)
                variant_b = self.augmentation_by_cube_subsample(temp_data,num_subsamples,temp_data,augment_length)
                
            else:
                variant_a_indices = random.sample(range(num_subsamples), self.subsample_size)
                variant_b_indices = random.sample(range(num_subsamples), self.subsample_size)
        
            if not self.augmentation_by_cube:
                variant_a = temp_data[variant_a_indices]
                variant_b = temp_data[variant_b_indices]
            
            batch[0].append(variant_a)
            batch[1].append(variant_b)
        return tuple(np.array(batch)) # batch
    
    def __len__(self):
        return self.num_batches
    
    def on_epoch_end(self):
        self.shufflecall()
