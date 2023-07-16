import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from torchvision.utils import make_grid
import os
import time
import tqdm

from config import *

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
        # gan loss
        gan_AB = criterion_gan(disc_B(fake_B), real)
        gan_BA = criterion_gan(disc_A(fake_A), real)
        loss_gan = (gan_AB + gan_BA) / 2
        # cyc loss
        cyc_A = criterion_cyc(gen_BA(fake_B), real_A)
        cyc_B = criterion_cyc(gen_AB(fake_A), real_B)
        loss_cyc = (cyc_A + cyc_B) / 2
        # id loss
        id_A = criterion_idn(gen_BA(real_A), real_A)
        id_B = criterion_idn(gen_AB(real_B), real_B)
        loss_id = (id_A + id_B) / 2

        loss_gen = loss_gan + loss_cyc * LAMBDA1 + loss_id * LAMBDA2
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
        # gan loss
        gan_AB = criterion_gan(disc_B(fake_B), real)
        gan_BA = criterion_gan(disc_A(fake_A), real)
        loss_gan = (gan_AB + gan_BA) / 2
        # cyc loss
        cyc_A = criterion_cyc(gen_BA(fake_B), real_A)
        cyc_B = criterion_cyc(gen_AB(fake_A), real_B)
        loss_cyc = (cyc_A + cyc_B) / 2
        # id loss
        id_A = criterion_idn(gen_BA(real_A), real_A)
        id_B = criterion_idn(gen_AB(real_B), real_B)
        loss_id = (id_A + id_B) / 2

        loss_gen = loss_gan + loss_cyc * LAMBDA1 + loss_id * LAMBDA2
        test_gen_loss += loss_gen

        # DiscA
        l_real = criterion_gan(disc_A(real_A), real)
        l_fake = criterion_gan(disc_A(fake_A), fake)
        loss_A = (l_real + l_fake) / 2
        test_disc_A_loss += loss_A

        # DiscB
        l_real = criterion_gan(disc_B(real_B), real)
        l_fake = criterion_gan(disc_B(fake_B), fake)
        loss_B = (l_real + l_fake) / 2
        test_disc_B_loss += loss_B

        # save img only at start of test epoch
        if batch_idx == 0:
            fake_Acyc = gen_BA(fake_B)
            fake_Bcyc = gen_AB(fake_A)

            image_grid = make_grid([real_A[0], fake_B[0], fake_Acyc[0], real_B[0], fake_A[0], fake_Bcyc[0]], nrow=3, normalize=True)
            writer.add_image('gen_images/test', image_grid, global_step=epoch)

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
    writer: SummaryWriter,
):
    best_gen_loss = float('inf')
    best_epoch = -1
    step = 0

    start = time.time() 
    for epoch in tqdm.tqdm(range(0, EPOCHS)):
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

        print(
            f"Epoch: {epoch} | "
            f"train_gen_loss: {train_loss[0]:.4f} | "
            f"train_discA_loss: {train_loss[1]:.4f} | "
            f"train_discB_loss: {train_loss[2]:.4f}"
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

        print(
            f"Epoch: {epoch} | "
            f"test_gen_loss: {test_loss[0]:.4f} | "
            f"test_discA_loss: {test_loss[1]:.4f} | "
            f"test_discB_loss: {test_loss[2]:.4f}"
        )

        step += len(train_dataloader)
        scheduler_gen.step()
        scheduler_disc_A.step()
        scheduler_disc_B.step()

        if test_loss[0] < best_gen_loss:
            best_gen_loss = test_loss[0]
            best_epoch = epoch

        writer.add_scalar("Loss_Gen/test", test_loss[0], global_step=epoch)
        writer.add_scalar("Loss_DiscA/test", test_loss[1], global_step=epoch)
        writer.add_scalar("Loss_DiscB/test", test_loss[2], global_step=epoch)

        if best_epoch == epoch:
            torch.save(gen_AB.state_dict(), os.path.join(DATA_DIR, f"{DATASET}_best_gen_AB"))
            torch.save(gen_BA.state_dict(), os.path.join(DATA_DIR, f"{DATASET}_best_gen_BA"))

    end = time.time()

    print(f"Training complete in {end - start} seconds")
    print(f"Best epoch was: {best_epoch}")
