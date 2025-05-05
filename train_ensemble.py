import dataloader
import config
import argparse
import os
import sys
import tqdm
import torch
import numpy as np
import model
from utils import *
import random
import coloredlogs
import logging
from torch.utils.tensorboard import SummaryWriter
from IMP_model import InfiniteMixturePrototype
import pandas as pd

LOG = logging.getLogger('base')
coloredlogs.install(level='DEBUG', logger=LOG)
s = 0.01


# def ensemble_forward(nets, x, y):
#     """
#     Forward pass through each net and aggregate outputs.
#     If y is provided, compute individual losses and accuracies and average.
#     Otherwise, return averaged logits.
#     """
#
#     logits_real = nets[0](x[:, 0].unsqueeze(1))
#     logits_imag = nets[1](x[:, 1].unsqueeze(1))
#     z = x[:, 0] + 1j * x[:, 1]
#     ang = torch.angle(z).unsqueeze(1)
#     logits_ph = nets[2](ang)
#     # Average logits
#     logits = (logits_real + logits_imag + logits_ph) / 3
#     loss, acc = loss_fn(logits, y)
#     # logits_list = []
#     # # for net in nets:
#     #     # for inference or train_flag=False, nets return (loss, acc)
#     #     # if y is not None:
#     #     #     loss, acc = net(x, y=y, train_flag=train_flag)
#     #     #     logits_list.append((loss, acc))
#     #     # else:
#     # logits = net[0](x[0])
#     # logits_list.append(logits)
#     #
#     # # if y is not None:
#     # #     # average the losses and accuracies
#     # #     losses, accs = zip(*logits_list)
#     # #     loss = sum(losses) / len(losses)
#     # #     acc = sum(accs) / len(accs)
#     # #     return loss, acc
#     # # else:
#     # # average logits
#     # logits = sum(logits_list) / len(logits_list)
#     # loss, acc = loss_fn(logits, y)
#     return loss, acc


def test_model(test_loader, nets, current_iter, tracker, writer, logger, save_this_iter):
    logger.info("Testing....")
    test_losses_real, test_losses_imag, test_losses_ph = [], [], []
    nets = [net.eval() for net in nets]
    label_list = test_loader.dataset.tensors[1].tolist()
    loss_dict = {0:[], 1:[], 2:[]}
    logit_dict = {0:[], 1:[], 2:[]}
    for i, net in enumerate(nets):
        with torch.no_grad():
            for x, y in tqdm.tqdm(test_loader, dynamic_ncols=True):
                x, y = x.cuda(), y.cuda()
                # loss, acc = ensemble_forward(nets, x, y, train_flag=False)
                # loss, acc = ensemble_forward(nets, x, y)
                # loss, acc = net(x, y)
                if i == 0:
                    logits = net(x[:, 0].unsqueeze(1))
                elif i == 1:
                    logits = net(x[:, 1].unsqueeze(1))
                else:
                    z = x[:, 0] + 1j * x[:, 1]
                    ang = torch.angle(z).unsqueeze(1)
                    logits = net(ang)
                loss, pred_y = loss_fn_2(logits, y)
                # label_list.extend(y)
                loss_dict[i].extend([loss.item()])
                logit_dict[i].extend(pred_y.tolist())
                # test_losses_real.append(loss.item())
                # test_acc.append(acc.item())
    mod_logits = pd.DataFrame(logit_dict).mode(axis=1)[0].to_list()
    test_acc = sum(p == t for p, t in zip(mod_logits, label_list)) / len(label_list)
    # mean_test_acc = np.mean(test_acc)
    # mean_test_loss = np.mean(test_losses)
    writer.add_scalar('Test Loss Real', np.sum(loss_dict[0])/len(test_loader), current_iter)
    writer.add_scalar('Test Loss Imag', np.sum(loss_dict[1])/len(test_loader), current_iter)
    writer.add_scalar('Test Loss Phase', np.sum(loss_dict[2])/len(test_loader), current_iter)
    writer.add_scalar('Test Acc', test_acc, current_iter)
    logger.info(f"Finished Testing! Mean Loss: {np.sum(loss_dict[0])/len(test_loader)}, Acc: {test_acc}.")
    logger.info("Back to Training.")

    # writer.add_scalar('Test Loss', mean_test_loss, current_iter)
    # writer.add_scalar('Test Acc', mean_test_acc, current_iter)
    logger.info("Finished Testing! Mean Loss: {}, Acc: {}.".format(
        np.sum(loss_dict[0])/len(test_loader), test_acc))
    logger.info("Back to Training.")

    if save_this_iter:
        tracker.create_new_save('val_'+str(round(tracker.best_acc, 4))+'_test_'+str(
            round(test_acc, 4))+'_'+str(current_iter+1).zfill(8)+'_real'+'.pth', nets[0])

        tracker.create_new_save('val_' + str(round(tracker.best_acc, 4)) + '_test_' + str(
            round(test_acc, 4)) + '_' + str(current_iter + 1).zfill(8) + '_imag' + '.pth', nets[1])

        tracker.create_new_save('val_' + str(round(tracker.best_acc, 4)) + '_test_' + str(
            round(test_acc, 4)) + '_' + str(current_iter + 1).zfill(8) + '_ph' + '.pth', nets[2])



def val_model(val_loader, nets, current_iter, tracker, writer, logger):
    logger.info("Validating....")
    label_list = val_loader.dataset.tensors[1].tolist()
    loss_dict = {0:[], 1:[], 2:[]}
    logit_dict = {0:[], 1:[], 2:[]}
    for i, net in enumerate(nets):
        with torch.no_grad():
            for x, y in tqdm.tqdm(val_loader, dynamic_ncols=True):
                x, y = x.cuda(), y.cuda()
                # loss, acc = ensemble_forward(nets, x, y, train_flag=False)
                # loss, acc = ensemble_forward(nets, x, y)
                # loss, acc = net(x, y)
                if i == 0:
                    logits = net(x[:, 0].unsqueeze(1))
                elif i == 1:
                    logits = net(x[:, 1].unsqueeze(1))
                else:
                    z = x[:, 0] + 1j * x[:, 1]
                    ang = torch.angle(z).unsqueeze(1)
                    logits = net(ang)
                loss, pred_y = loss_fn_2(logits, y)
                # label_list.extend(y)
                loss_dict[i].extend([loss.item()])
                logit_dict[i].extend(pred_y.tolist())
                # test_losses_real.append(loss.item())
                # test_acc.append(acc.item())
    mod_logits = pd.DataFrame(logit_dict).mode(axis=1)[0].to_list()
    valid_acc = sum(p == t for p, t in zip(mod_logits, label_list)) / len(label_list)
    # mean_val_acc = np.sum(val_acc)/len(val_loader.dataset)
    # mean_val_loss = np.sum(val_losses)/len(val_loader.dataset)
    # mean_val_acc = np.mean(val_acc)
    # mean_val_loss = np.mean(val_losses)
    writer.add_scalar('Validation Loss Real', np.sum(loss_dict[0])/len(val_loader), current_iter)
    writer.add_scalar('Validation Loss Imag', np.sum(loss_dict[1])/len(val_loader), current_iter)
    writer.add_scalar('Validation Loss Phase', np.sum(loss_dict[2])/len(val_loader), current_iter)
    writer.add_scalar('Validation Acc', valid_acc, current_iter)
    logger.info(f"Finished Validation! Loss: {np.sum(loss_dict[0])/len(val_loader)}, Acc: {valid_acc}")
    return tracker.update(valid_acc)


def main(args):
    cfg = config.parse_config(args.config)
    ckpt_dir = args.checkpoint_dir
    setup_ckpt(ckpt_dir)
    setup_logger(LOG, ckpt_dir)
    LOG.info(f"\ncfg: {cfg}")
    LOG.info(f"args: {args}")

    # Tensorboard
    writer = SummaryWriter(os.path.join(ckpt_dir, 'runs'))

    # Seed
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    if args.deterministic:
        torch.use_deterministic_algorithms(True)

    # Instantiate separate models
    net_real = getattr(model, cfg['model']['name'])(**cfg['model']['args']).cuda()
    net_imag = getattr(model, cfg['model']['name'])(**cfg['model']['args']).cuda()
    net_ph   = getattr(model, cfg['model']['name'])(**cfg['model']['args']).cuda()
    nets = [net_real, net_imag, net_ph]

    for net in nets:
        net.train()
    LOG.info(f"# Params per model: {count_params(net_real)}")

    # Combined optimizer
    optimizers = [
        torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)
        for net in nets
    ]
    # all_params = list(net_real.parameters()) + list(net_imag.parameters()) + list(net_ph.parameters())
    # optimizer = torch.optim.AdamW(all_params, lr=args.lr, weight_decay=args.wd)
    # # optimizer_real = torch.optim.AdamW(list(net_real.parameters()), lr=args.lr, weight_decay=args.wd)
    # # optimizer_imag = torch.optim.AdamW(list(net_imag.parameters()), lr=args.lr, weight_decay=args.wd)
    # # # optimizer_ph = torch.optim.AdamW(list(net_ph.parameters()), lr=args.lr, weight_decay=args.wd)
    # #
    # # optimizers = [optimizer_real, optimizer_imag, optimizer_ph]

    # Data loaders
    dataset_fn = getattr(dataloader, cfg['dataset']['function'])
    loader_args = cfg['dataset'].get('args', {})
    train_loader, val_loader, test_loader = dataset_fn(
        seed=args.seed,
        train_batch=args.bs,
        val_batch=args.bs,
        test_batch=args.bs,
        mag_only=args.mag_only,
        **loader_args
    )

    tracker = Model_Tracker(ckpt_dir, LOG)
    save_this_iter = False
    train_iter = iter(train_loader)

    for current_iter in tqdm.trange(args.num_iters + 1, dynamic_ncols=True):
        if ((current_iter % args.val_every) == 0) and current_iter > 0:
            # LOG.info(f"Last Training Batch Acc: {acc.item()}")
            if val_loader:
                save_this_iter = val_model(
                    val_loader, nets, current_iter, tracker, writer, LOG)
            if save_this_iter:
                test_model(test_loader, nets, current_iter,
                           tracker, writer, LOG, save_this_iter)
                save_this_iter = False
            for net in nets:
                net.train()
                net.zero_grad()
        # for optimizer in optimizers:
        loss_list, acc_list = [], []
        loss_dict = {0:[], 1:[], 2:[]}
        for i, (net, optimizer) in enumerate(zip(nets, optimizers)):
            optimizer.zero_grad()
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)

            x, y = x.cuda(), y.cuda()
            # Forward & loss
            # Get logits per model
            if i == 0:
                logits = net(x[:, 0].unsqueeze(1))
            elif i == 1:
                logits = net(x[:, 1].unsqueeze(1))
            else:
                z = x[:, 0] + 1j * x[:, 1]
                ang = torch.angle(z).unsqueeze(1)
                logits = net(ang)
            # logits_real = net_real(x[:, 0].unsqueeze(1))
            # logits_imag = net_imag(x[:, 1].unsqueeze(1))
            # z = x[:, 0] + 1j * x[:, 1]
            # ang = torch.angle(z).unsqueeze(1)
            # logits_ph = net_ph(ang)
            # # Average logits
            # logits = (logits_real + logits_imag + logits_ph) / 3
            loss, _ = loss_fn_2(logits, y)
            loss_dict[i].append(loss.item())
            loss.backward()
            # loss_list.append(loss); acc_list.append(acc)
            # for optimizer in optimizers:
            optimizer.step()

        if current_iter % args.log_every == 0:
            writer.add_scalar('Train Loss Real', loss_dict[0][0], current_iter)
            writer.add_scalar('Train Loss Imag', loss_dict[1][0], current_iter)
            writer.add_scalar('Train Loss Phase', loss_dict[2][0], current_iter)

            # writer.add_scalar('Train Acc Real', acc_list.item(), current_iter)
            # writer.add_scalar('Train Acc Imag', acc_list.item(), current_iter)
            # writer.add_scalar('Train Acc Phase', acc_list.item(), current_iter)

        if args.save_on and current_iter in eval(f'[{args.save_on}]'):
            os.makedirs(f'{ckpt_dir}/saves/', exist_ok=True)
            torch.save({
                'real': net_real.state_dict(),
                'imag': net_imag.state_dict(),
                'ph': net_ph.state_dict()
            }, f"{ckpt_dir}/saves/{current_iter:06d}.pth")

    LOG.info("Done Training!")

    # Final validation
    save_this_iter = val_model(val_loader, nets, current_iter, tracker, writer, LOG)
    LOG.info("Exiting.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config", default='config/default.yml')
    parser.add_argument("--checkpoint_dir", default='./ckpt/')
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--bs", type=int, default=256)
    parser.add_argument("--wd", type=float, default=0)
    parser.add_argument("--num_iters", type=int, default=500000)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--mag_only", action="store_true")
    parser.add_argument("--val_every", type=int, default=1000)
    parser.add_argument("--deterministic", action='store_true')
    parser.add_argument("--save_on", type=str, default=None)
    args = parser.parse_args()
    main(args)
