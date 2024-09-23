import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm


# Define a transform to convert the dataset to grayscale and then to a tensor
transform = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
    transforms.ToTensor()                         # Convert image to tensor
])

dataset = datasets.ImageFolder("./data/train", transform=transform, 
											is_valid_file=lambda x: not os.path.basename(x).startswith('.'))
# Create a DataLoader
loader = DataLoader(dataset, batch_size=64, shuffle=False)

# Function to calculate mean and std
def calculate_mean_std(loader):
    mean = 0.0
    std = 0.0
    total_images_count = 0

    for images, _ in tqdm(loader):
        # Images is a batch of images, we flatten the batch into a single tensor
        batch_images_count = images.size(0)  # Number of images in the batch
        images = images.view(batch_images_count, images.size(1), -1)  # Flatten height and width
        mean += images.mean(2).sum(0)  # Mean across width/height and sum over batch
        std += images.std(2).sum(0)  # Std across width/height and sum over batch
        total_images_count += batch_images_count

    # Calculate mean and std for the entire dataset
    mean /= total_images_count
    std /= total_images_count
    return mean, std

# Calculate mean and standard deviation
mean, std = calculate_mean_std(loader)

print(f"Mean: {mean.item()}, Std: {std.item()}")

# Mean: 0.8611186742782593, Std: 0.2043401449918747

