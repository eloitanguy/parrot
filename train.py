import torch
from dataset import train_collate_function, ParrotDataset
from tensorboardX import SummaryWriter
from torch.optim import Adam
from model import ParrotModel
from config.train_config import TrainConfig
from utils import printProgressBar, AverageMeter
from torch.utils.data import DataLoader
import os


def train():
    cfg = TrainConfig()
    checkpoint_folder = 'checkpoints/{}/'.format(cfg.experiment_name)

    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)

    tb_folder = 'tb/{}/'.format(cfg.experiment_name)
    if not os.path.exists(checkpoint_folder):
        os.makedirs()

    writer = SummaryWriter(logdir='tb/{}/'.format(cfg.experiment_name), flush_secs=30)
    model = ParrotModel().cuda()
    optimiser = Adam(model.parameters(), lr=cfg.initial_lr, weight_decay=cfg.weight_decay)

    train_dataset = ParrotDataset(cfg.train_labels, cfg.mp3_folder)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, num_workers=cfg.workers,
                              collate_fn=train_collate_function)

    val_dataset = ParrotDataset(cfg.val_labels, cfg.mp3_folder)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, num_workers=cfg.workers,
                            collate_fn=train_collate_function, shuffle=False)

    epochs = cfg.epochs
    init_loss, step = 0., 0
    avg_loss = AverageMeter()
    for epoch in range(epochs):
        loader_length = len(train_loader)
        model = model.train()

        for batch_idx, batch in enumerate(train_loader):
            optimiser.zero_grad()

            # inference
            labels = batch['label']
            model_input = batch['spectrogram'].cuda()
            model_output = model(model_input)

            # loss
            loss = None  # TODO: loss
            loss.backward()

            if epoch == 0 and batch_idx == 0:
                init_loss = loss

            # logging
            avg_loss.update(loss)
            printProgressBar(batch_idx, loader_length, prefix='Epoch [{}/{}]'.format(epoch, epochs),
                             suffix='loss {}/{}'.format(loss, init_loss))

            writer.add_scalar('Iterations/train_loss', loss, step)

            # saving the model
            if step % cfg.checkpoint_every == 0:
                name = '{}/epoch_{}.pth'.format(checkpoint_folder, epoch)
                torch.save({'model': model.state_dict(), 'epoch': epoch, 'batch_idx': batch_idx, 'step': step}, name)

            step += 1
            optimiser.step()

        # end of epoch
        writer.add_scalar('Epochs/train_loss', avg_loss.avg, epoch)
        avg_loss.reset()
        name = '{}/epoch_{}.pth'.format(checkpoint_folder, epoch)
        torch.save({'model': model.state_dict(), 'epoch': epoch, 'batch_idx': loader_length, 'step': step}, name)

        # validation
        model = model.eval()
        total_val_loss = 0.
        for batch_idx, batch in enumerate(val_loader):
            labels = batch['label']
            model_input = batch['spectrogram'].cuda()
            model_output = model(model_input)
            # loss
            loss = None  # TODO: loss
            printProgressBar(batch_idx, loader_length, prefix='Epoch [{}/{}]'.format(epoch, epochs),
                             suffix='Validation ...')
            total_val_loss += loss

        val_loss = total_val_loss/len(val_loader)
        writer.add_scalar('Epochs/val_loss', val_loss, epoch)

    # finished training
    name = '{}/epoch_{}.pth'.format(checkpoint_folder, epochs)
    torch.save({'model': model.state_dict(), 'epoch': epochs, 'batch_idx': 0, 'step': step}, name)
    writer.close()
    print('Training finished :)')
