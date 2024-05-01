import cv2
import numpy as np
import os
import pickle
from tqdm import tqdm

from sklearn.cluster import KMeans
from sklearn.neighbors import BallTree
import matplotlib.pyplot as plt
import VPR

codebook_path = r'./data/images_codebook/'
database_path = r'./data/images/'
query_path = r'./data/query_images/'
position_path = r'./data/positions.txt'
SIFT_extractor = cv2.SIFT_create()

# codebook = VPR.create_codebook(SIFT_extractor, codebook_path)
codebook = pickle.load(open("codebook.pkl", "rb"))
# tree, database_name = VPR.create_tree(SIFT_extractor, codebook, database_path)
[tree, database_name] = pickle.load(open('tree.pkl', 'rb'))
indeces, value_list = VPR.find_target(SIFT_extractor, query_path, codebook, tree, database_name)
positions, goal_indeces = VPR.match_images(query_path, value_list, database_path, position_path, indeces)
VPR.plot_positions(positions,goal_indeces)