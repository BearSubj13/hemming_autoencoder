import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
import os
import copy

from data_loading import EnglishWordDataset, HemmingTransform, train_val_split
from model import EncoderRNN, SimpleNet
from recall_top import recall_top2
from utils import yaml_config_hook
from loss_metrics import hemming_distance_batch


def regression_validation(data_loader_val, regressor, model, device='cpu'):
    loss_function = nn.L1Loss()
    regressor.eval()
    model.eval()
    loss_list = []

    for batch_i, x_i in enumerate(data_loader_val):
        x_i = x_i.to(device)
        permutation = torch.randperm(x_i.shape[0])
        x_j = x_i[permutation, :, :]
        ground_truth_hemmig_distance = hemming_distance_batch(x_i, x_j)

        model.initHidden(device, batch_size=x_i.shape[0])
        _, embedding_i = model(x_i)
        embedding_i = embedding_i[0].squeeze().detach()
        model.initHidden(device, batch_size=x_i.shape[0])
        _, embedding_j = model(x_j)
        embedding_j = embedding_j[0].squeeze().detach()
        prediction = regressor(embedding_i, embedding_j)

        loss = loss_function(prediction, ground_truth_hemmig_distance)
        loss_list.append(loss.item())

    return sum(loss_list)/len(loss_list)


def train(data_loader_train, data_loader_val, data_loader_test, regressor, model, optimizer, device='cpu', epoch_number=45):
    loss_function = nn.MSELoss()
    model.eval()

    for epoch in range(epoch_number):
        loss_list = []
        regressor.train()
        if epoch > 40:
            for param in optimizer.param_groups:
                param['lr'] = max(0.0001, param['lr'] / 1.1)
                print('lr: ', param['lr'])

        for batch_i, (x_i, x_j) in enumerate(data_loader_train):
            x_i = x_i.to(device)
            x_j = x_j.to(device)
            permutation = torch.randperm(x_j.shape[0])
            x_j = x_j[permutation, :, :]
            ground_truth_hemmig_distance = hemming_distance_batch(x_i, x_j)

            model.initHidden(device, batch_size=x_i.shape[0])
            _, embedding_i = model(x_i)
            embedding_i = embedding_i[0].squeeze().detach()
            model.initHidden(device, batch_size=x_i.shape[0])
            _, embedding_j = model(x_j)
            embedding_j = embedding_j[0].squeeze().detach()
            # del batch
            # torch.cuda.empty_cache()
            prediction = regressor(embedding_i, embedding_j)

            loss = loss_function(prediction, ground_truth_hemmig_distance)
            loss_list.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss = sum(loss_list) / len(loss_list)
        val_error = regression_validation(data_loader_val, regressor, model, device)
        if epoch == 0:
            min_val_error = val_error
        else:
            min_val_error = min(min_val_error, val_error)
        if min_val_error == val_error:
            best_model = copy.deepcopy(regressor)
        print('loss: {0:2.4f}, validation error: {1:1.3f}'.format(loss, val_error))

    test_error = regression_validation(data_loader_test, best_model, model, device)
    print('test error: {0:1.3f}'.format(test_error))


def main():
    config = yaml_config_hook('config.yaml')
    english_dataset1 = EnglishWordDataset('words_dictionary.json', config['max_sequence_length'], return_token=True)
    transformation = HemmingTransform(english_dataset1.alphabet)
    english_dataset1.__add_transformation__(transformation)
    train_val = train_val_split(english_dataset1, val_split=0.9, fix_split_state=666)
    english_dataset_train = train_val['train']

    english_dataset2 = EnglishWordDataset('words_dictionary.json', config['max_sequence_length'], return_token=True)
    train_val = train_val_split(english_dataset2, val_split=0.9, fix_split_state=666)
    train_val = train_val_split(train_val['val'], val_split=0.7, fix_split_state=666)
    english_dataset_test = train_val['train']
    english_dataset_val = train_val['val']

    english_dataloader_train = DataLoader(english_dataset_train, shuffle=True, batch_size=300, drop_last=True)
    english_dataloader_val = DataLoader(english_dataset_val, shuffle=False, batch_size=config['batch_size'], drop_last=False)
    english_dataloader_test = DataLoader(english_dataset_test, shuffle=False, batch_size=config['batch_size'], drop_last=False)

    encoder = EncoderRNN(hidden_size=config['hidden_size'], input_size=config['token_size'])
    encoder = encoder.to(config['device'])
    load_path = 'weights/metric_learn_encoder_300epoch.pth'
    encoder.load_state_dict(torch.load(load_path))
    encoder.eval()

    regressor = SimpleNet(config['hidden_size'])
    regressor = regressor.to(config['device'])
    optimizer = torch.optim.Adam(regressor.parameters(), lr=0.001)

    train(english_dataloader_train, english_dataloader_val, english_dataloader_test, regressor, encoder, optimizer, device=config['device'], epoch_number=3)

if __name__ == "__main__":
    main()
