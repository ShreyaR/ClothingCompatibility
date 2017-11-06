from PIL import Image
from torchvision.transforms import ToTensor, Normalize, Resize, RandomCrop, Compose

class SiameseNetworkDataset:

	def __init__(self,imageFile,inputSize):
		self.inputSize = inputSize
		self.imageFile = imageFile
		self.resize = Resize(self.inputSize)
		self.transforms = Compose([RandomCrop(self.inputSize), ToTensor(),
		Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
		
	def __getitem__(self):
		with open(self.imageFile) as f:
			for line in f:

				# im1, im2, label = line.rstrip().split()
				eg_info = line.rstrip().split()
				im1 = Image.open(eg_info[1])
				im2 = Image.open(eg_info[2])
	
				if min(im1.size) < self.inputSize:
					im1 =  self.resize.__call__(im1)
				if min(im2.size) < self.inputSize:
					im2 = self.resize.__call__(im2)
				
				im1 = self.transforms.__call__(im1)
				im2 = self.transforms.__call__(im2)

				# yield im1, im2, int(label)   	
	 			yield [eg_info[0]] + [im1, im2] + eg_info[3:]
