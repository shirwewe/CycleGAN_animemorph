import torch
from config import *
from data_loader import get_data_loader
from train import train_fn
from model import Generator, Discriminator
from utils import save_checkpoint, load_checkpoint
import torch.optim as optim
from torch.cuda.amp import GradScaler

def main():
    train_loader_A, val_loader_A, test_loader_A = get_data_loader(TRAIN_DIR, BATCH_SIZE, image_size=(128, 128))
    train_loader_B, val_loader_B, test_loader_B = get_data_loader(VAL_DIR, BATCH_SIZE, image_size=(128, 128))

    gen_anime = Generator(img_channels=3, num_residuals=9).to(DEVICE)
    gen_human = Generator(img_channels=3, num_residuals=9).to(DEVICE)
    disc_anime = Discriminator(in_channels=3).to(DEVICE)
    disc_human = Discriminator(in_channels=3).to(DEVICE)

    opt_gen = optim.Adam(
        list(gen_anime.parameters()) + list(gen_human.parameters()),
        lr=LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    opt_disc = optim.Adam(
        list(disc_anime.parameters()) + list(disc_human.parameters()),
        lr=LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    l1 = nn.L1Loss()
    mse = nn.MSELoss()

    d_scaler = GradScaler()
    g_scaler = GradScaler()

    if LOAD_MODEL:
        start_epoch = load_latest_checkpoint(gen_anime, opt_gen, CHECKPOINT_GENERATOR_ANIME, LEARNING_RATE, BATCH_SIZE)
        start_epoch = load_latest_checkpoint(gen_human, opt_gen, CHECKPOINT_GENERATOR_HUMAN, LEARNING_RATE, BATCH_SIZE)
        start_epoch = load_latest_checkpoint(disc_anime, opt_disc, CHECKPOINT_DISCRIMINATOR_ANIME, LEARNING_RATE, BATCH_SIZE)
        start_epoch = load_latest_checkpoint(disc_human, opt_disc, CHECKPOINT_DISCRIMINATOR_HUMAN, LEARNING_RATE, BATCH_SIZE)
        print(f"Resuming from epoch {start_epoch}")
    else:
        start_epoch = 0
    num_epochs = 6  # Total number of epochs to run
    remaining_epochs = num_epochs - start_epoch
    print(f"Starting training for {remaining_epochs} epochs from epoch {start_epoch + 1}")


    train_fn(disc_human=disc_human, disc_anime=disc_anime, gen_anime=gen_anime, gen_human=gen_human, 
        train_loader_A=train_loader_A, train_loader_B=train_loader_B, 
        opt_disc=opt_disc, opt_gen=opt_gen, 
        l1=l1, mse=mse, d_scaler=d_scaler, g_scaler=g_scaler, 
        num_epochs=remaining_epochs, start_epoch=start_epoch)


    gan_model_path = get_model_name("loss", epoch=NUMS_EPOCHS, learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE)
    plot_training_curve(gan_model_path)

if __name__ == "__main__":
    main()
