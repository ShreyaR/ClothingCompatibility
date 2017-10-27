from PIL import Image
from torchvision.transforms import ToTensor, Normalize, Resize, RandomCrop

class SiameseNetworkDataset(Dataset):
    
    def __init__(self,imageFile,transform=None):
        self.imageFile = imageFile    
        self.transform = transform
        
    def __getitem__(self,index):
        with open(self.imageFile) as f:
            for line in f:

                im1, im2, label = line.rstrip().split()
                im1 = ToTensor(Image.open(im1))
                im2 = ToTensor(Image.open(im2))

                print im1.size
                
                

        
                yield img0, img1 , torch.from_numpy(np.array([int(img1_tuple[1]!=img0_tuple[1])],dtype=np.float32))
    
    def __len__(self):
        return len(self.imageFile.imgs)