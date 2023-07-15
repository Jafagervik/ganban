from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
import time
from helpers import datasetup, utils

from torchvision.utils import save_image
from torchvision.utils import make_grid
import os

def train_step(
    gen_AB: nn.Module,
    gen_BA: nn.Module,
    disc_A: nn.Module,
    disc_B: nn.Module,
    train_dataloader: DataLoader,
    criterion_gan: nn.Module,
    criterion_cyc: nn.Module,
    criterion_idn: nn.Module,
    optimizer_gen: torch.optim.Optimizer,
    optimizer_disc_A: torch.optim.Optimizer,
    optimizer_disc_B: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    step: int,
    writer: SummaryWriter,
):
    gen_AB.train()
    gen_BA.train()
    disc_A.train()
    disc_B.train()

    train_gen_loss, train_disc_A_loss, train_disc_B_loss = 0.0, 0.0, 0.0

    for batch_idx, (real_A, real_B) in enumerate(train_dataloader):
        real_A = real_A.to(device)
        real_B = real_B.to(device)
        fake_A = gen_BA(real_B)
        fake_B = gen_AB(real_A)
        real = torch.ones_like(disc_A(fake_A))
        fake = torch.zeros_like(real)

        # Gen
        optimizer_gen.zero_grad()
        # id loss
        id_A = criterion_idn(gen_BA(real_A), real_A)
        id_B = criterion_idn(gen_AB(real_B), real_B)
        loss_id = (id_A + id_B) / 2
        # gan loss
        gan_AB = criterion_gan(disc_B(fake_B), real)
        gan_BA = criterion_gan(disc_A(fake_A), real)
        loss_gan = (gan_AB + gan_BA) / 2
        # cyc loss
        cyc_A = criterion_cyc(gen_BA(fake_B), real_A)
        cyc_B = criterion_cyc(gen_AB(fake_A), real_B)
        loss_cyc = (cyc_A + cyc_B) / 2

        loss_gen = loss_gan + loss_cyc * 10.0 + loss_id * 5.0
        loss_gen.backward()
        optimizer_gen.step()
        train_gen_loss += loss_gen

        # DiscA
        optimizer_disc_A.zero_grad()
        fake_A = gen_BA(real_B)

        fake = torch.zeros_like(real)
        l_real = criterion_gan(disc_A(real_A), real)
        l_fake = criterion_gan(disc_A(fake_A), fake)
        loss_A = (l_real + l_fake) / 2
        loss_A.backward()
        optimizer_disc_A.step()
        train_disc_A_loss += loss_A

        # DiscB
        optimizer_disc_B.zero_grad()
        fake_B = gen_AB(real_A)

        l_real = criterion_gan(disc_B(real_B), real)
        l_fake = criterion_gan(disc_B(fake_B), fake)
        loss_B = (l_real + l_fake) / 2
        loss_B.backward()
        optimizer_disc_B.step()
        train_disc_B_loss += loss_B

        writer.add_scalar("Loss_Gen/train", loss_gen, global_step=step)
        writer.add_scalar("Loss_DiscA/train", loss_A, global_step=step)
        writer.add_scalar("Loss_DiscB/train", loss_B, global_step=step)
        step += 1

    train_gen_loss /= len(train_dataloader)
    train_disc_A_loss /= len(train_dataloader)
    train_disc_B_loss /= len(train_dataloader)

    return train_gen_loss, train_disc_A_loss, train_disc_B_loss

@torch.inference_mode()
def test_step(
    gen_AB: nn.Module,
    gen_BA: nn.Module,
    disc_A: nn.Module,
    disc_B: nn.Module,
    test_dataloader: DataLoader,
    criterion_gan: nn.Module,
    criterion_cyc: nn.Module,
    criterion_idn: nn.Module,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
):
    gen_AB.eval()
    gen_BA.eval()
    disc_A.eval()
    disc_B.eval()

    test_gen_loss, test_disc_A_loss, test_disc_B_loss = 0.0, 0.0, 0.0

    for batch_idx, (real_A, real_B) in enumerate(test_dataloader):
        real_A = real_A.to(device)
        real_B = real_B.to(device)
        fake_A = gen_BA(real_B)
        fake_B = gen_AB(real_A)
        real = torch.ones_like(disc_A(fake_A))
        fake = torch.zeros_like(real)

        # Gen
        # id loss
        id_A = criterion_idn(gen_BA(real_A), real_A)
        id_B = criterion_idn(gen_AB(real_B), real_B)
        loss_id = (id_A + id_B) / 2
        # gan loss
        gan_AB = criterion_gan(disc_B(fake_B), real)
        gan_BA = criterion_gan(disc_A(fake_A), real)
        loss_gan = (gan_AB + gan_BA) / 2
        # cyc loss
        cyc_A = criterion_cyc(gen_BA(fake_B), real_A)
        cyc_B = criterion_cyc(gen_AB(fake_A), real_B)
        loss_cyc = (cyc_A + cyc_B) / 2

        loss_gen = loss_gan + loss_cyc * 10.0 + loss_id * 5.0
        test_gen_loss += loss_gen

        # DiscA
        l_real = criterion_gan(disc_A(real_A), real)
        # paper: last50 queue here
        l_fake = criterion_gan(disc_A(fake_A), fake)
        loss_A = (l_real + l_fake) / 2
        test_disc_A_loss += loss_A

        # DiscB
        l_real = criterion_gan(disc_B(real_B), real)
        # paper: last50 queue here
        l_fake = criterion_gan(disc_B(fake_B), fake)
        loss_B = (l_real + l_fake) / 2
        test_disc_B_loss += loss_B

        if False and batch_idx % 100 == 0:
            print(
                f"[Epoch {epoch}] "
                f"[Batch {batch_idx}/{len(test_dataloader)}] "
                f"[LossG: {loss_gen:.4f}] "
                f"[LossD_A: {loss_A:.4f}] "
                f"[LossD_B: {loss_B:.4f}]"
            )

        if batch_idx == 0: # only at start of epoch
            fake_Acyc = gen_BA(fake_B)
            fake_Bcyc = gen_AB(fake_A)

            real_A = make_grid(real_A, nrow=6, normalize=True)
            real_B = make_grid(real_B, nrow=6, normalize=True)
            fake_A = make_grid(fake_A, nrow=6, normalize=True)
            fake_B = make_grid(fake_B, nrow=6, normalize=True)
            fake_Acyc = make_grid(fake_Acyc, nrow=6, normalize=True)
            fake_Bcyc = make_grid(fake_Bcyc, nrow=6, normalize=True)
            image_grid = torch.cat((real_A, fake_B, fake_Acyc, real_B, fake_A, fake_Bcyc), 1)
            writer.add_image('gen_images/test', image_grid, global_step=epoch)
            save_image(image_grid, os.path.join("runs", f"{epoch:03}.png"), normalize=False)

    test_gen_loss /= len(test_dataloader)
    test_disc_A_loss /= len(test_dataloader)
    test_disc_B_loss /= len(test_dataloader)

    return test_gen_loss, test_disc_A_loss, test_disc_B_loss


def train(
    gen_AB,
    gen_BA,
    disc_A,
    disc_B,
    train_dataloader,
    test_dataloader,
    criterion_gan: nn.Module,
    criterion_cyc: nn.Module,
    criterion_idn: nn.Module,
    optimizer_gen: torch.optim.Optimizer,
    optimizer_disc_A: torch.optim.Optimizer,
    optimizer_disc_B: torch.optim.Optimizer,
    scheduler_gen: StepLR,
    scheduler_disc_A: StepLR,
    scheduler_disc_B: StepLR,
    device: torch.device,
    config,
    args,
    writer: SummaryWriter,
):
    best_gen_loss = float('inf')
    best_epoch = -1

    start = time.time() 
    step = 0
    for epoch in tqdm(range(0, config['epochs'])):
        print("")
        train_loss = train_step(
            gen_AB,
            gen_BA,
            disc_A,
            disc_B,
            train_dataloader,
            criterion_gan,
            criterion_cyc,
            criterion_idn,
            optimizer_gen,
            optimizer_disc_A,
            optimizer_disc_B,
            device,
            epoch,
            step,
            writer,
        )
        step += len(train_dataloader)

        scheduler_gen.step()
        scheduler_disc_A.step()
        scheduler_disc_B.step()

        print(
            f"Epoch: {epoch} | "
            f"train_gen_loss: {train_loss[0]:.4f} | "
            f"train_dcA_loss: {train_loss[1]:.4f} | "
            f"train_dcB_loss: {train_loss[2]:.4f}"
        )

        test_loss = test_step(
            gen_AB,
            gen_BA,
            disc_A,
            disc_B,
            test_dataloader,
            criterion_gan,
            criterion_cyc,
            criterion_idn,
            device,
            epoch,
            writer,
        )

        if test_loss[0] < best_gen_loss:
            best_gen_loss = test_loss[0]
            best_epoch = epoch

        print(
            f"Epoch: {epoch} | "
            f"test_gen_loss: {test_loss[0]:.4f} | "
            f"test_dcA_loss: {test_loss[1]:.4f} | "
            f"test_dcB_loss: {test_loss[2]:.4f}"
        )

        writer.add_scalar("Loss_Gen/test", test_loss[0], global_step=epoch)
        writer.add_scalar("Loss_DiscA/test", test_loss[1], global_step=epoch)
        writer.add_scalar("Loss_DiscB/test", test_loss[2], global_step=epoch)

        if args.dry_run:
            break

        if (True or args.save_model) and best_epoch == epoch:
            torch.save(gen_AB.state_dict(), os.path.join(config['checkpoint_dir'], f"{config['data_set']}_best_gen_AB"))
            torch.save(gen_BA.state_dict(), os.path.join(config['checkpoint_dir'], f"{config['data_set']}_best_gen_BA"))

        if (True or args.save_model) and epoch % 20 == 0:
            torch.save(gen_AB.state_dict(), os.path.join(config['checkpoint_dir'], f"{config['data_set']}_{epoch}_gen_AB"))
            torch.save(gen_BA.state_dict(), os.path.join(config['checkpoint_dir'], f"{config['data_set']}_{epoch}_gen_BA"))
            torch.save(disc_A.state_dict(), os.path.join(config['checkpoint_dir'], f"{config['data_set']}_{epoch}_disc_A"))
            torch.save(disc_B.state_dict(), os.path.join(config['checkpoint_dir'], f"{config['data_set']}_{epoch}_disc_B"))

        # if early_stop: 
        #     break

    end = time.time()

    print(f"Training complete in {end - start} seconds")
    print(f"Best epoch was: {best_epoch}")

    # Save model to file if selected
    if (True or args.save_model):
        torch.save(gen_AB.state_dict(), os.path.join(config['checkpoint_dir'], f"{config['data_set']}_last_gen_AB"))
        torch.save(gen_BA.state_dict(), os.path.join(config['checkpoint_dir'], f"{config['data_set']}_last_gen_BA"))

    #graphs.plot_acc_loss(results)
