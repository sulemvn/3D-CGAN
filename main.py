# Cell 1: Hyperparameters and Imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt

# Hyperparameters
BATCH_SIZE = 40
LEARNING_RATE_DISCRIMINATOR = 0.00015
LEARNING_RATE_GENERATOR = 0.0002
BETAS = (0.5, 0.999)
NUM_EPOCHS = 8
NOISE_SIZE = 200
CRITERION = nn.BCELoss()

# Cell 2: Load data and create DataLoader
data = np.load("modelnet10.npz", allow_pickle=True)
train_voxel = torch.tensor(data["train_voxel"], dtype=torch.float32).unsqueeze(1)
dataset = TensorDataset(train_voxel)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Cell 3: Generator Network
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.network = nn.Sequential(
            nn.ConvTranspose3d(NOISE_SIZE, 512, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm3d(512),
            nn.ReLU(True),

            nn.ConvTranspose3d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(256),
            nn.ReLU(True),

            nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(128),
            nn.ReLU(True),

            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(True),

            nn.ConvTranspose3d(64, 1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.network(z)


# Cell 4: Discriminator Network
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.network = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(512, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)


# Cell 5: Initialize Networks and Optimizers
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = Generator().to(device)
discriminator = Discriminator().to(device)
optimizer_discriminator = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE_DISCRIMINATOR, betas=BETAS)
optimizer_generator = optim.Adam(generator.parameters(), lr=LEARNING_RATE_GENERATOR, betas=BETAS)

# Cell 6: Training Loop
generator_losses = []
discriminator_losses = []

for epoch in range(NUM_EPOCHS):
    for batch_index, data in enumerate(train_loader, 0):
        
        # --- Train the Discriminator ---
        # Zero out the gradients for the discriminator
        discriminator.zero_grad()
        
        # Train with real data
        real_voxels = data[0].to(device)
        real_labels = torch.full((real_voxels.size(0),), 1., device=device)
        output_real = discriminator(real_voxels).view(-1)
        error_discriminator_real = CRITERION(output_real, real_labels)
        error_discriminator_real.backward()

        # Train with fake data
        noise = torch.randn(real_voxels.size(0), NOISE_SIZE, 1, 1, 1, device=device)
        fake_voxels = generator(noise)
        fake_labels = torch.full((real_voxels.size(0),), 0., device=device)
        output_fake = discriminator(fake_voxels.detach()).view(-1)
        error_discriminator_fake = CRITERION(output_fake, fake_labels)
        error_discriminator_fake.backward()

        # Update discriminator
        discriminator_accuracy = ((output_real > 0.5).float().mean() + (output_fake < 0.5).float().mean()) / 2
        if discriminator_accuracy < 0.8:
            optimizer_discriminator.step()

        # --- Train the Generator ---
        # Zero out the gradients for the generator
        generator.zero_grad()
        
        # Create inverted labels for the fake loss calculation
        fake_labels = torch.full((real_voxels.size(0),), 1., device=device)
        output_fake = discriminator(fake_voxels).view(-1)
        error_generator = CRITERION(output_fake, fake_labels)
        error_generator.backward()

        # Update generator
        optimizer_generator.step()

        # Save Losses for plotting later
        generator_losses.append(error_generator.item())
        discriminator_losses.append((error_discriminator_real + error_discriminator_fake).item())

        # Output training stats
        if batch_index % 50 == 0:
            print(f'[{epoch+1}/{NUM_EPOCHS}][{batch_index}/{len(train_loader)}] '
                  f'Loss_D: {error_discriminator_real.item()+error_discriminator_fake.item():.4f} '
                  f'Loss_G: {error_generator.item():.4f} '
                  f'D(x): {output_real.mean().item():.4f} '
                  f'D(G(z)): {output_fake.mean().item():.4f}')


# Cell 7: Utility Function to Count Parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Cell 8: Output Number of Parameters
print(f'Total parameters in Generator: {count_parameters(generator)}')
print(f'Total parameters in Discriminator: {count_parameters(discriminator)}')

# Cell 9: Loss Tracking and Plotting
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(generator_losses, label="Generator Loss")
plt.plot(discriminator_losses, label="Discriminator Loss")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Cell 10: Save Models
torch.save(generator.state_dict(), 'generator.pth')
torch.save(discriminator.state_dict(), 'discriminator.pth')

# Cell 11: Plot the Training Losses
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(generator_losses, label="Generator Loss")
plt.plot(discriminator_losses, label="Discriminator Loss")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Cell 12: Shows samples generated 


# Assuming the rest of the code is present and the models are defined as generator and discriminator

# Load the trained models
generator.load_state_dict(torch.load('generator.pth'))
discriminator.load_state_dict(torch.load('discriminator.pth'))

# Load labels and class map for annotation
data = np.load("modelnet10.npz", allow_pickle=True)
class_map = data['class_map'].item()  # Assuming class_map is stored as a numpy array
train_labels = data['train_labels']  # Assuming you are using training labels for annotation

# Visualize samples generated by the generator and label them with class names
def visualize_generated_samples(generator_model, device, labels, class_mapping, num_samples=4):
    generator_model.eval()  # Set the generator to evaluation mode
    with torch.no_grad():
        # Generate latent vectors and create generated samples with the generator
        latent_vectors = torch.randn(num_samples, 200, 1, 1, 1, device=device)
        generated_voxels = generator_model(latent_vectors).cpu()

    # Plotting the generated samples
    fig = plt.figure(figsize=(15, 3))
    for i in range(num_samples):
        ax = fig.add_subplot(1, num_samples, i+1, projection='3d')
        # Apply a threshold to visualize the voxels
        ax.voxels(generated_voxels[i].squeeze() > 0.5)
        # Fetch the class label from class_map using the label index
        class_label = class_mapping.get(str(labels[i]), "Unknown") if i < len(labels) else "Unknown"
        ax.set_title(f'Sample {i+1}: {class_label}')
    plt.show()

# Assuming 'device' is already defined
visualize_generated_samples(generator, device, train_labels, class_map)
