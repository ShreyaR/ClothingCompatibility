from PIL import Image
from torchvision.transforms import ToTensor, Normalize, Resize, RandomCrop, Compose

class SiameseNetworkDataset:
    
    def __init__(self,imageFile,transform=None):
        self.imageFile = imageFile    
	# self.totensor = ToTensor()
	self.resize = Resize(224)
	#self.normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	#self.randomcrop = RandomCrop(224)
	self.transforms = Compose([RandomCrop(224), ToTensor(),
		Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        
    def __getitem__(self):
        with open(self.imageFile) as f:
            for line in f:

                im1, im2, label = line.rstrip().split()
                im1 = Image.open(im1)
                im2 = Image.open(im2)
		
		if min(im1[0,0,:].size()[0], im1[0,:,0].size()[0]) < 224:
			im1 =  Resize.__call__(im1)
		if min(im2[0,0,:].size()[0], im2[0,:,0].size()[0]) < 224:
			im2 = Resize.__call__(im2)
		
		im1 = self.transforms.__call__(im1)
		im2 = self.transforms.__call__(im2)

     		yield im1, im2, int(label)   	
                # yield img0, img1 , torch.from_numpy(np.array([int(img1_tuple[1]!=img0_tuple[1])],dtype=np.float32))
    
    #def __len__(self):
    #    return len(self.imageFile.imgs)
