from torchvision.transforms import ToTensor
from dataset_loading import SiameseNetworkDataset

x = SiameseNetworkDataset("/data/srajpal2/AmazonDataset/test_train.txt")

for im1, im2, label in x.__getitem__():
	print im1.size(), im2.size(), type(label)

