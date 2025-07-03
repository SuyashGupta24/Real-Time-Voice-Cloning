from pathlib import Path
import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from encoder.params_model import *
model_hidden_size = 384      # Updated for improved temporal feature extraction
model_embedding_size = 384   # Increased for more detailed speaker representation
model_num_layers = 4         # One extra layer to capture deeper sequential patterns

from encoder.data_objects import SpeakerVerificationDataLoader, SpeakerVerificationDataset
from encoder.model import SpeakerEncoder
from encoder.params_model import learning_rate_init, speakers_per_batch, utterances_per_speaker
from encoder.visualizations import Visualizations
from utilsm.profiler import Profiler


def sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def train(run_id: str, clean_data_root: Path, models_dir: Path, umap_every: int, save_every: int,
          backup_every: int, vis_every: int, force_restart: bool, visdom_server: str,
          no_visdom: bool):
    # Build the dataset and corresponding dataloader
    dataset = SpeakerVerificationDataset(clean_data_root)
    loader = SpeakerVerificationDataLoader(
        dataset,
        speakers_per_batch,
        utterances_per_speaker,
        num_workers=4,
    )

    # Set computation devices: GPU if available, otherwise CPU; loss is calculated on CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_device = torch.device("cpu")

    # Create the model instance and optimizer with the updated hyperparameters
    model = SpeakerEncoder(device, loss_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate_init)
    init_step = 1

    # Define the directory for saving the model checkpoints
    model_dir = models_dir / run_id
    model_dir.mkdir(exist_ok=True, parents=True)
    state_fpath = model_dir / "encoder.pt"

    # If a checkpoint exists and restart is not forced, load the previous state
    if not force_restart:
        if state_fpath.exists():
            print("Resuming training for model \"%s\"." % run_id)
            checkpoint = torch.load(state_fpath)
            init_step = checkpoint["step"]
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            optimizer.param_groups[0]["lr"] = learning_rate_init
        else:
            print("No existing checkpoint for \"%s\"; starting fresh." % run_id)
    else:
        print("Forced restart: beginning training from scratch.")
    model.train()

    # Initialize visualization tools for tracking progress
    vis = Visualizations(run_id, vis_every, server=visdom_server, disabled=no_visdom)
    vis.log_dataset(dataset)
    vis.log_params()
    device_name = str(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    vis.log_implementation({"Device": device_name})
    
    # Start the profiler for monitoring performance metrics
    profiler = Profiler(summarize_every=10, disabled=False)
    
    # Main training loop
    for step, speaker_batch in enumerate(loader, init_step):
        profiler.tick("Waiting for batch")
        inputs = torch.from_numpy(speaker_batch.data).to(device)
        sync(device)
        profiler.tick("Data transferred to " + str(device))
        
        # Compute embeddings using the model
        embeds = model(inputs)
        sync(device)
        profiler.tick("Forward pass completed")
        
        # Reshape embeddings and transfer to loss device for loss computation
        embeds_loss = embeds.view((speakers_per_batch, utterances_per_speaker, -1)).to(loss_device)
        loss, eer = model.loss(embeds_loss)
        sync(loss_device)
        profiler.tick("Loss evaluation complete")
        
        # Backpropagation and optimization
        model.zero_grad()
        loss.backward()
        profiler.tick("Backpropagation complete")
        model.do_gradient_ops()
        optimizer.step()
        profiler.tick("Model parameters updated")
        
        # Update the visualization interface with current loss and error metrics
        vis.update(loss.item(), eer, step)
        
        # Generate UMAP projections and save the visual output periodically
        if umap_every != 0 and step % umap_every == 0:
            print("Generating UMAP visualization (step %d)" % step)
            projection_fpath = model_dir / f"umap_{step:06d}.png"
            embeds_np = embeds.detach().cpu().numpy()
            vis.draw_projections(embeds_np, utterances_per_speaker, step, projection_fpath)
            vis.save()
        
        # Save a checkpoint of the model periodically
        if save_every != 0 and step % save_every == 0:
            print("Saving checkpoint (step %d)" % step)
            torch.save({
                "step": step + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            }, state_fpath)
        
        # Create periodic backups of the model state
        if backup_every != 0 and step % backup_every == 0:
            print("Creating backup (step %d)" % step)
            backup_fpath = model_dir / f"encoder_{step:06d}.bak"
            torch.save({
                "step": step + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            }, backup_fpath)
        
        profiler.tick("Iteration complete (visualizations and saving)")
