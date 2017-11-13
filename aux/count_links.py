import json

compatibility_links = 0
similarity_links = 0
all_links = 0

# with open("/data/srajpal2/AmazonDataset/updated_categories_meta_tbs.json") as f:
# with open("/data/srajpal2/AmazonDataset/val_images.json") as f:
# with open("/data/srajpal2/AmazonDataset/testing_images.json") as f:
with open("/data/srajpal2/AmazonDataset/training_images.json") as f:
	for line in f:
		info = json.loads(line.rstrip())
		compatibility_links += len(info["related"]["compatible"])
		similarity_links += len(info["related"]["similar"])
		all_links += len(info["related"]["compatible"])+len(info["related"]["similar"])

print compatibility_links, similarity_links, all_links
