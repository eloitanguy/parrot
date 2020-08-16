import torch
from dataset import parrot_collate_function, ParrotDataset
from tensorboardX import SummaryWriter
from torch.optim import AdamW
from model import ParrotModel
from config.train_config import TrainConfig
from utils import printProgressBar, AverageMeter
from torch.utils.data import DataLoader
from modules import ctc_loss, test_mp3_file
import time
import datetime
import os


def val(model, val_loader, writer, step):
    """ Computes the loss on the validation set and logs it to tensorboard
     The loss is computed on a fixed subset with the first _ batches, defined in config file"""
    cfg = TrainConfig()
    model = model.eval()
    with torch.no_grad():
        total_val_loss = 0.
        for batch_idx, batch in enumerate(val_loader):

            # run only on a subset
            if batch_idx >= cfg.val_batches:
                break

            # VRAM control by skipping long examples
            if batch['spectrograms'].shape[-1] > cfg.max_time:
                continue

            # inference
            target = batch['targets'].cuda()
            model_input = batch['spectrograms'].cuda()
            model_output = model(model_input)

            # loss
            input_lengths = batch['input_lengths'].cuda()
            target_lengths = batch['target_lengths'].cuda()
            loss = ctc_loss(model_output, target, input_lengths, target_lengths).cpu().item()

            # log
            printProgressBar(batch_idx, cfg.val_batches, suffix='\tValidation ...')
            total_val_loss += loss

    val_loss = total_val_loss / cfg.val_batches
    writer.add_scalar('Steps/val_loss', val_loss, step)
    print('Finished validation with loss {:4f}'.format(val_loss))


def train():
    """ Train the model using the parameters defined in the config file """
    print('Initialising ...')
    cfg = TrainConfig()
    checkpoint_folder = 'checkpoints/{}/'.format(cfg.experiment_name)

    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)

    tb_folder = 'tb/{}/'.format(cfg.experiment_name)
    if not os.path.exists(tb_folder):
        os.makedirs(tb_folder)

    writer = SummaryWriter(logdir=tb_folder, flush_secs=30)
    model = ParrotModel().cuda().train()
    optimiser = AdamW(model.parameters(), lr=cfg.initial_lr, weight_decay=cfg.weight_decay)

    train_dataset = ParrotDataset(cfg.train_labels, cfg.mp3_folder)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, num_workers=cfg.workers,
                              collate_fn=parrot_collate_function, pin_memory=True)

    val_dataset = ParrotDataset(cfg.val_labels, cfg.mp3_folder)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, num_workers=cfg.workers,
                            collate_fn=parrot_collate_function, shuffle=False, pin_memory=True)

    epochs = cfg.epochs
    init_loss, step = 0., 0
    avg_loss = AverageMeter()
    print('Starting training')
    for epoch in range(epochs):
        loader_length = len(train_loader)
        epoch_start = time.time()

        for batch_idx, batch in enumerate(train_loader):
            optimiser.zero_grad()

            # VRAM control by skipping long examples
            if batch['spectrograms'].shape[-1] > cfg.max_time:
                continue

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
            est = datetime.timedelta(seconds=int(elapsed / progress)) if progress > 0.001 else '-'
            avg_loss.update(loss)
            suffix = '\tloss {:.4f}/{:.4f}\tETA [{}/{}]'.format(avg_loss.avg, init_loss,
                                                                datetime.timedelta(seconds=int(elapsed)), est)
            printProgressBar(batch_idx, loader_length, suffix=suffix,
                             prefix='Epoch [{}/{}]\tStep [{}/{}]'.format(epoch, epochs, batch_idx, loader_length))

            writer.add_scalar('Steps/train_loss', loss, step)

            # saving the model
            if step % cfg.checkpoint_every == 0:
                test_name = '{}/test_epoch{}.mp3'.format(checkpoint_folder, epoch)
                test_mp3_file(cfg.test_mp3, model, test_name)
                checkpoint_name = '{}/epoch_{}.pth'.format(checkpoint_folder, epoch)
                torch.save({'model': model.state_dict(), 'epoch': epoch, 'batch_idx': loader_length, 'step': step,
                            'optimiser': optimiser.state_dict()}, checkpoint_name)

            # validating
            if step % cfg.val_every == 0:
                val(model, val_loader, writer, step)
                model = model.train()

            step += 1
            optimiser.step()

        # end of epoch
        print('')
        writer.add_scalar('Epochs/train_loss', avg_loss.avg, epoch)
        avg_loss.reset()
        test_name = '{}/test_epoch{}.mp3'.format(checkpoint_folder, epoch)
        test_mp3_file(cfg.test_mp3, model, test_name)
        checkpoint_name = '{}/epoch_{}.pth'.format(checkpoint_folder, epoch)
        torch.save({'model': model.state_dict(), 'epoch': epoch, 'batch_idx': loader_length, 'step': step,
                    'optimiser': optimiser.state_dict()}, checkpoint_name)

    # finished training
    writer.close()
    print('Training finished :)')


if __name__ == '__main__':
    train()
