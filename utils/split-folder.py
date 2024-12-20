import splitfolders

dataset_path = r"rock-scissorsp-paper\dataset\rps-cv-images"
output_path = r"rock-scissorsp-paper\dataset\splitted-dataset"
splitfolders.ratio(dataset_path, output=output_path, seed=1337, ratio=(0.8, 0.1, 0.1))