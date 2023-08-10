import os
import nibabel as nib  # reading the libable images 
import torch   # for tensor operations
from torch.utils.data import Dataset, DataLoader  # define datasets and define data loader
import torchvision.transforms as transforms      # for transformation task

class IBSDDataset(Dataset):  # lets create data loader class
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir  # = DATA/data2 -> images -> labels
        self.image_folder = os.path.join(data_dir, "images")  # define the image folder
        self.mask_folder = os.path.join(data_dir, "labels")   # define the lable folder,  [1-14] labels are there
        self.image_files = os.listdir(self.image_folder) #
        self.mask_files = os.listdir(self.mask_folder) # mask files from mask folder
        print(f'images files : {type(self.image_files[0])}')  # printing the first images 
        print(f'length of images file :  {len(self.image_files)}')  # printing the length of the images / total images
        print(f'length of mask file   : {len(self.mask_files)}')  # length of the mask
        print(f'1st images : {self.image_files[0]}')
        print(f'1st mask   : {self.mask_files[0]}')   # printing teh forst mask
        print(f'12th images----->{self.image_files[12]}')
        print(f'12th mask------->{self.mask_files[12]}')
        print(f'50th images----->{self.image_files[50]}')
        print(f'50th mask------->{self.mask_files[50]}')
        print(f'100th images----->{self.image_files[100]}')
        print(f'100th mask------->{self.mask_files[100]}')
        self.transform = transform                # performing some transforming operations

    def __len__(self):
#         print(f'length of images files :  {len(self.image_files)}')  # printing the length of the images / total images
#         print(f'length of mask files : {len(self.mask_files)}')  # length of the mask
#         print(f'first images----->{self.image_files[0]}')
#         print(f'first mask------->{self.mask_files[0]}')
        return len(self.image_files) # return the length of the images

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_folder, self.image_files[idx])
        mask_path = os.path.join(self.mask_folder, self.mask_files[idx])
#         mask_path = os.path.join(self.mask_folder, self.image_files[idx])# .replace('_T1w', '_masks'))

        image_nifti = nib.load(image_path)
        mask_nifti = nib.load(mask_path)

        image_data = image_nifti.get_fdata() #.astype(torch.double)
        mask_data = mask_nifti.get_fdata() #.astype(torch.double)

        # Normalize the image data if required
        # You can add more preprocessing steps here

        if self.transform:
            image_data = self.transform(image_data )
            mask_data = self.transform(mask_data)

        return image_data , mask_data   # return image data and mask data

# Define data transformations
transform = transforms.Compose([
    transforms.ToTensor()
    # Add more transforms as needed
])

# Define the path to the IBSD dataset folder
data_dir = 'D:/NEURO_DATA/IBRS/3Dunet/data'


if __name__=="__main__":
    dataset = IBSDDataset(data_dir, transform=transform)
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True)
    print(type(data_loader))
    print(data_loader)
    print(dataset)






