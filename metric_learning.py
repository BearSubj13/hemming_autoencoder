import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
import os

from data_loading import EnglishWordDataset, HemmingTransform, train_val_split
from model import EncoderRNN
from recall_top import recall_top2


def yaml_config_hook(config_file):
    """
    Custom YAML config loader, which can include other yaml files (I like using config files
    insteaad of using argparser)
    """

    # load yaml files in the nested 'defaults' section, which include defaults for experiments
    with open(config_file) as f:
        cfg = yaml.safe_load(f)
        for d in cfg.get("defaults", []):
            config_dir, cf = d.popitem()
            cf = os.path.join(os.path.dirname(config_file), config_dir, cf + ".yaml")
            with open(cf) as f:
                l = yaml.safe_load(f)
                cfg.update(l)

    if "defaults" in cfg.keys():
        del cfg["defaults"]

    return cfg


class NT_Xent(nn.Module):
    def __init__(self, batch_size, temperature, device, world_size=1):
        super(NT_Xent, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.world_size = world_size

        self.mask = self.mask_correlated_samples(batch_size, world_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size, world_size):
        N = 2 * batch_size * world_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size * world_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        N = 2 * self.batch_size * self.world_size

        z = torch.cat((z_i, z_j), dim=0)
        # if self.world_size > 1:
        #     z = torch.cat(GatherLayer.apply(z), dim=0)

        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, self.batch_size * self.world_size)
        sim_j_i = torch.diag(sim, -self.batch_size * self.world_size)

        # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss


def train_epoch(train_loader, model, criterion, optimizer, device):
    loss_epoch = 0
    recall10_epoch = 0
    recall_epoch = 0
    model.train()

    for step, (x_i, x_j) in enumerate(train_loader):
        optimizer.zero_grad()
        x_i = x_i.to(device)
        x_j = x_j.to(device)
        # positive pair, with encoding
        model.initHidden(device=device, batch_size=x_i.shape[0])
        _, h_i = model(x_i)
        z_i = model.projector(h_i[0].squeeze())
        _, h_j = model(x_j)
        z_j = model.projector(h_j[0].squeeze())

        recall, recall10 = recall_top2(z_i, z_j)
        recall10_epoch += recall10.item()
        recall_epoch += recall.item()

        loss = criterion(z_i, z_j)
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            print(f"Step [{step}/{len(train_loader)}]\t Loss: {loss.item()} \t Recall@1: {recall.item()} \t Recall@1*10: {recall10.item()}")

        loss_epoch += loss.item()
    loss_epoch = loss_epoch / len(train_loader)
    recall_epoch10 = recall10_epoch / len(train_loader)
    return loss_epoch, recall_epoch10


def train(train_loader, epoch_number, model, device, path_to_save=None):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = NT_Xent(train_loader.batch_size, 1.0, device)
    for epoch in range(epoch_number):
        loss, recall_epoch10 = train_epoch(train_loader, model, criterion, optimizer, device)
        print('epoch {0}, loss:{1}'.format(epoch, round(loss, 5)))
        if path_to_save:
            torch.save(model.state_dict(), path_to_save)


def main():
    config = yaml_config_hook('config.yaml')
    english_dataset1 = EnglishWordDataset('words_dictionary.json', config['max_sequence_length'], return_token=True)
    transformation = HemmingTransform(english_dataset1.alphabet)
    english_dataset1.__add_transformation__(transformation)
    train_val = train_val_split(english_dataset1, val_split=0.9, fix_split_state=666)
    english_dataset_train = train_val['train']

    english_dataset2 = EnglishWordDataset('words_dictionary.json', config['max_sequence_length'], return_token=True)
    train_val = train_val_split(english_dataset2, val_split=0.9, fix_split_state=666)
    english_dataset_val = train_val['val']

    english_dataloader_train = DataLoader(english_dataset_train, shuffle=True, batch_size=config['batch_size'], drop_last=True)
    english_dataloader_val = DataLoader(english_dataset_val, shuffle=False, batch_size=config['batch_size'], drop_last=False)

    encoder = EncoderRNN(hidden_size=config['hidden_size'], input_size=config['token_size'])
    encoder = encoder.to(config['device'])
    load_path = 'weights/metric_learn_encoder_300epoch.pth'
    encoder.load_state_dict(torch.load(load_path))

    save_path = 'weights/metric_learn_encoder.pth'

    train(english_dataloader_train, 300, encoder, config['device'], path_to_save=save_path)


if __name__ == "__main__":
    main()
