from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import time

def train_step(
    gen_AB: nn.Module,
    gen_BA: nn.Module,
    disc_A: nn.Module,
    disc_B: nn.Module,
    dataloader: DataLoader,
    criterion_gan: nn.Module,
    criterion_cyc: nn.Module,
    criterion_idn: nn.Module,
    optimizer_gen: torch.optim.Optimizer,
    optimizer_disc_A: torch.optim.Optimizer,
    optimizer_disc_B: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
):
    gen_AB.train()
    gen_BA.train()
    disc_A.train()
    disc_B.train()

    train_gen_loss, train_disc_A_loss, train_disc_B_loss = 0.0, 0.0, 0.0

    for batch_idx, (real_A, real_B) in enumerate(dataloader):
        real_A = real_A.to(device)
        real_B = real_B.to(device)
        fake_A = gen_BA(real_B)
        fake_B = gen_AB(real_A)
        real = torch.ones_like(disc_A(fake_A))
        fake = torch.zeros_like(real)

        # Gen
        optimizer_gen.zero_grad()
        # id loss
        id_A = criterion_idn(gen_BA(real_B), real_A)
        id_B = criterion_idn(gen_AB(real_A), real_A)
        loss_id = (id_A + id_B) / 2
        # gan loss
        gan_AB = criterion_gan(disc_B(fake_B), real)
        gan_BA = criterion_gan(disc_A(fake_A), real)
        loss_gan = (gan_AB + gan_BA) / 2
        # cyc loss
        cyc_A = criterion_cyc(gen_BA(fake_B), real_A)
        cyc_B = criterion_cyc(gen_AB(fake_A), real_B)
        loss_cyc = (cyc_A + cyc_B) / 2

        loss_gen = loss_gan + loss_cyc * 10.0 + loss_id * 5.0 # add to config
        loss_gen.backward()
        optimizer_gen.step()
        train_gen_loss += loss_gen

        # DiscA
        optimizer_disc_A.zero_grad()
        loss_real = criterion_gan(disc_A(real_A), real)
        # paper: last50 queue here
        loss_fake = criterion_gan(disc_A(real_A), fake)
        loss_A = (loss_real + loss_fake) / 2
        loss_A.backward()
        optimizer_disc_A.step()
        train_disc_A_loss += loss_A

        # DiscB
        optimizer_disc_B.zero_grad()
        loss_real = criterion_gan(disc_B(real_B), real)
        # paper: last50 queue here
        loss_fake = criterion_gan(disc_B(real_B), fake)
        loss_B = (loss_real + loss_fake) / 2
        loss_B.backward()
        optimizer_disc_A.step()
        train_disc_B_loss += loss_B

        #print(
        #    f"[Epoch {epoch}] "
        #    f"[Batch {batch_idx}/{len(dataloader)}] "
        #    f"[LossG: {loss_gen:.4f}] "
        #    f"[LossD_A: {loss_A:.4f}] "
        #    f"[LossD_B: {loss_B:.4f}]"
        #)

    train_gen_loss /= len(dataloader)
    train_disc_A_loss /= len(dataloader)
    train_disc_B_loss /= len(dataloader)

    return train_gen_loss, train_disc_A_loss, train_disc_B_loss

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
):
    highest_acc = 0.0
    best_epoch = -1

    start = time.time() 

    for epoch in tqdm(range(0, config['epochs'])):
        loss = train_step(
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
        )

        # val_step()

        #test_loss, test_acc, highest_acc, best_epoch = test_step(
        #    model, highest_acc, best_epoch, test_dataloader, criterion, device, epoch, config)

        scheduler_gen.step()
        scheduler_disc_A.step()
        scheduler_disc_B.step()

        if args.dry_run:
            break

        print(
            f"Epoch: {epoch} | "
            f"train_gen_loss: {loss[0]:.4f} | "
            f"train_dcA_loss: {loss[1]:.4f} | "
            f"train_dcB_loss: {loss[2]:.4f}"
        )

        # if early_stop: 
        #     break


    end = time.time()

    print(f"Training complete in {end - start} seconds")
    print(f"Best epoch was: {best_epoch}")

    # Save model to file if selected
    if args.save_model:
        torch.save(gen_AB.state_dict(), os.join.path(config['checkpoint_dir'], "last_gen_AB"))
        torch.save(gen_BA.state_dict(), os.join.path(config['checkpoint_dir'], "last_gen_BA"))

    #graphs.plot_acc_loss(results)
