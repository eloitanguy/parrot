import torch
from dataset import parrot_collate_function, ParrotDataset
from tensorboardX import SummaryWriter
from torch.optim import Adam
from model import ParrotModel
from config.train_config import TrainConfig
from utils import printProgressBar, AverageMeter
from torch.utils.data import DataLoader
from modules import ctc_loss
import time
import datetime
import os


def train():
    print('Initialising ...')
    cfg = TrainConfig()
    checkpoint_folder = 'checkpoints/{}/'.format(cfg.experiment_name)

    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)

    tb_folder = 'tb/{}/'.format(cfg.experiment_name)
    if not os.path.exists(tb_folder):
        os.makedirs(tb_folder)

    writer = SummaryWriter(logdir=tb_folder, flush_secs=30)
    model = ParrotModel().cuda()
    optimiser = Adam(model.parameters(), lr=cfg.initial_lr, weight_decay=cfg.weight_decay)

    train_dataset = ParrotDataset(cfg.train_labels, cfg.mp3_folder)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, num_workers=cfg.workers,
                              collate_fn=parrot_collate_function)

    val_dataset = ParrotDataset(cfg.val_labels, cfg.mp3_folder)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, num_workers=cfg.workers,
                            collate_fn=parrot_collate_function, shuffle=False)

    epochs = cfg.epochs
    init_loss, step = 0., 0
    avg_loss = AverageMeter()
    print('Starting training')
    for epoch in range(epochs):
        loader_length = len(train_loader)
        epoch_start = time.time()
        model = model.train()

        for batch_idx, batch in enumerate(train_loader):
            optimiser.zero_grad()

            # inference
            target = batch['targets'].cuda()
            model_input = batch['spectrograms'].cuda()
            model_output = model(model_input)

            # loss
            input_lengths = batch['input_lengths'].cuda()
            target_lengths = batch['target_lengths'].cuda()
            loss = ctc_loss(model_output, target, input_lengths, target_lengths)
            loss.backward()

            if epoch == 0 and batch_idx == 0:
                init_loss = loss

            # logging
            elapsed = time.time() - epoch_start
            progress = batch_idx / loader_length
            est = datetime.timedelta(seconds=int(elapsed / progress)) if progress > 0.01 else '-'
            avg_loss.update(loss)
            suffix = '\tloss {:.2f}/{:.2f}\tETA [{}/{}]'.format(avg_loss.avg, init_loss,
                                                                datetime.timedelta(seconds=int(elapsed)), est)
            printProgressBar(batch_idx, loader_length, prefix='Epoch [{}/{}]'.format(epoch, epochs), suffix=suffix)

            writer.add_scalar('Iterations/train_loss', loss, step)

            # saving the model
            if step % cfg.checkpoint_every == 0:
                name = '{}/epoch_{}.pth'.format(checkpoint_folder, epoch)
                torch.save({'model': model.state_dict(), 'epoch': epoch, 'batch_idx': batch_idx, 'step': step}, name)

            step += 1
            optimiser.step()

        # end of epoch
        print('')
        writer.add_scalar('Epochs/train_loss', avg_loss.avg, epoch)
        avg_loss.reset()
        name = '{}/epoch_{}.pth'.format(checkpoint_folder, epoch)
        torch.save({'model': model.state_dict(), 'epoch': epoch, 'batch_idx': loader_length, 'step': step}, name)

        # validation
        model = model.eval()
        total_val_loss = 0.
        for batch_idx, batch in enumerate(val_loader):
            # inference
            target = batch['targets'].cuda()
            model_input = batch['spectrograms'].cuda()
            model_output = model(model_input)

            # loss
            input_lengths = batch['input_lengths'].cuda()
            target_lengths = batch['target_lengths'].cuda()
            loss = ctc_loss(model_output, target, input_lengths, target_lengths)

            # log
            printProgressBar(batch_idx, loader_length, prefix='Epoch [{}/{}]'.format(epoch, epochs),
                             suffix='\tValidation ...')
            total_val_loss += loss

        val_loss = total_val_loss / len(val_loader)
        writer.add_scalar('Epochs/val_loss', val_loss, epoch)
        print('')

    # finished training
    name = '{}/epoch_{}.pth'.format(checkpoint_folder, epochs)
    torch.save({'model': model.state_dict(), 'epoch': epochs, 'batch_idx': 0, 'step': step}, name)
    writer.close()
    print('Training finished :)')


if __name__ == '__main__':
    train()
