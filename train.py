import os
import numpy as np
import torch
from torchvision.utils import save_image
from tqdm import tqdm

def train_fn(disc_human, disc_anime, gen_anime, gen_human, train_loader_A, train_loader_B, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler, num_epochs=1, start_epoch=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loss_anime_history = []
    train_loss_human_history = []
    val_loss_anime_history = []
    val_loss_human_history = []

    # Ensure the outputs directory exists
    os.makedirs('outputs', exist_ok=True)

    for epoch in range(num_epochs):
        current_epoch = start_epoch + epoch
        print(f"Epoch [{current_epoch + 1}/{start_epoch + num_epochs}]")
        epoch_loop = tqdm(zip(train_loader_A, train_loader_B), total=min(len(train_loader_A), len(train_loader_B)), leave=True, desc="Iteration")

        total_train_loss_anime = 0.0
        total_train_loss_human = 0.0
        total_batches = 0

        H_reals = 0
        H_fakes = 0

        for idx, ((human, _), (anime, _)) in enumerate(epoch_loop):
            human = human.to(device)
            anime = anime.to(device)

            # Train Discriminators Anime and Human
            with torch.cuda.amp.autocast():
                fake_anime = gen_anime(human)
                D_anime_real = disc_anime(anime)
                D_anime_fake = disc_anime(fake_anime.detach())
                H_reals += D_anime_real.mean().item()
                H_fakes += D_anime_fake.mean().item()
                anime_real_loss = mse(D_anime_real, torch.ones_like(D_anime_real))
                anime_fake_loss = mse(D_anime_fake, torch.zeros_like(D_anime_fake))
                D_anime_loss = anime_real_loss + anime_fake_loss

                fake_human = gen_human(anime)
                D_human_real = disc_human(human)
                D_human_fake = disc_human(fake_human.detach())
                human_real_loss = mse(D_human_real, torch.ones_like(D_human_real))
                human_fake_loss = mse(D_human_fake, torch.zeros_like(D_human_fake))
                D_human_loss = human_real_loss + human_fake_loss

                D_loss = (D_anime_loss + D_human_loss) / 2

            opt_disc.zero_grad()
            d_scaler.scale(D_loss).backward()
            d_scaler.step(opt_disc)
            d_scaler.update()

            # Train Generators Anime and Human
            with torch.cuda.amp.autocast():
                # Adversarial losses
                D_anime_fake = disc_anime(fake_anime)
                D_human_fake = disc_human(fake_human)
                loss_G_anime = mse(D_anime_fake, torch.ones_like(D_anime_fake))
                loss_G_human = mse(D_human_fake, torch.ones_like(D_human_fake))

                # Cycle losses
                cycle_human = gen_human(fake_anime)
                cycle_anime = gen_anime(fake_human)
                cycle_human_loss = l1(human, cycle_human)
                cycle_anime_loss = l1(anime, cycle_anime)

                # Identity losses (optional)
                identity_human = gen_human(human)
                identity_anime = gen_anime(anime)
                identity_human_loss = l1(human, identity_human)
                identity_anime_loss = l1(anime, identity_anime)

                # Total loss
                G_loss = (
                    loss_G_anime
                    + loss_G_human
                    + cycle_human_loss * LAMBDA_CYCLE
                    + cycle_anime_loss * LAMBDA_CYCLE
                    + identity_anime_loss * LAMBDA_IDENTITY
                    + identity_human_loss * LAMBDA_IDENTITY
                )

            opt_gen.zero_grad()
            g_scaler.scale(G_loss).backward()
            g_scaler.step(opt_gen)
            g_scaler.update()

            if idx % 200 == 0:
                # Save the fake anime and human images
                save_image(fake_anime * 0.5 + 0.5, f"outputs/fake_anime_{idx}.png")
                save_image(fake_human * 0.5 + 0.5, f"outputs/fake_human_{idx}.png")
                save_image(human * 0.5 + 0.5, f"outputs/real_human_{idx}.png")
                save_image(anime * 0.5 + 0.5, f"outputs/real_anime_{idx}.png")

            # Update iteration progress bar
            epoch_loop.set_postfix(D_loss=D_loss.item(), G_loss=G_loss.item(), H_real=H_reals / (idx + 1), H_fake=H_fakes / (idx + 1))
            epoch_loop.update(1)

            # Accumulate training losses
            total_train_loss_anime += loss_G_anime.item()
            total_train_loss_human += loss_G_human.item()
            total_batches += 1

        epoch_loop.close()

        # Calculate average training losses
        avg_train_loss_anime = total_train_loss_anime / total_batches
        avg_train_loss_human = total_train_loss_human / total_batches

        # Evaluate models
        val_loss_anime = evaluate(gen_anime, disc_anime, val_loader_A, mse, device)
        val_loss_human = evaluate(gen_human, disc_human, val_loader_B, mse, device)
        
        # Print epoch summary
        print(f"Epoch {current_epoch + 1}: Train Loss Anime: {avg_train_loss_anime}, Train Loss Human: {avg_train_loss_human}, Validation Loss Anime: {val_loss_anime}, Validation Loss Human: {val_loss_human}")

        # Save checkpoint at the end of each epoch
        save_checkpoint(gen_anime, opt_gen, filename=get_model_name("gen_anime", current_epoch, LEARNING_RATE, BATCH_SIZE))
        save_checkpoint(gen_human, opt_gen, filename=get_model_name("gen_human", current_epoch, LEARNING_RATE, BATCH_SIZE))
        save_checkpoint(disc_anime, opt_disc, filename=get_model_name("disc_anime", current_epoch, LEARNING_RATE, BATCH_SIZE))
        save_checkpoint(disc_human, opt_disc, filename=get_model_name("disc_human", current_epoch, LEARNING_RATE, BATCH_SIZE))

        print(f"Epoch {current_epoch + 1} checkpoint saved.")

        # Save training and validation losses
        train_loss_anime_history.append(avg_train_loss_anime)
        train_loss_human_history.append(avg_train_loss_human)
        val_loss_anime_history.append(val_loss_anime)
        val_loss_human_history.append(val_loss_human)


    save_losses(train_loss_anime_history, train_loss_human_history, val_loss_anime_history, val_loss_human_history, 
                get_model_name("loss", current_epoch, LEARNING_RATE, BATCH_SIZE))
