import cv2
import numpy as np
import os
import pickle
from tqdm import tqdm

from sklearn.cluster import KMeans
from sklearn.neighbors import BallTree
import matplotlib.pyplot as plt

# database_path = r'./data/images_codebook/'
# query_path = r'./data/train_images/'
# SIFT_extractor = cv2.SIFT_create()
# codebook = pickle.load(open("codebook.pkl", "rb"))

def extract_aggregate_feature(extractor, folder_path):
    all_descriptors = list()
    database_feature = dict()
    img_paths = os.listdir(folder_path)
    img_paths = sorted(img_paths, key=lambda x: int(x.split('.')[0]))

    print("Extracting SIFT Features:")
    for img_name in tqdm(img_paths):
        img = cv2.imread(folder_path + img_name)
        kp, des = extractor.detectAndCompute(img, None)

        all_descriptors.extend(des)

    all_descriptors = np.asarray(all_descriptors)
    return all_descriptors, img_paths

def create_codebook(SIFT_extractor, database_path):
    print("Creating Codebook:")
    database_entire_des, img_paths = extract_aggregate_feature(SIFT_extractor, database_path)
    codebook = KMeans(n_clusters=64, init='k-means++', n_init=10, verbose=1).fit(database_entire_des)
    pickle.dump(codebook, open('codebook.pkl', 'wb'))
    return codebook


def get_VLAD(img, codebook):
    predictedLabels = codebook.predict(img)
    centroids = codebook.cluster_centers_
    labels = codebook.labels_
    k = codebook.n_clusters

    n, d = img.shape
    VLAD_feature = np.zeros([k, d])

    for i in range(k):
        if np.sum(predictedLabels == i) > 0:
            VLAD_feature[i] = np.sum(img[predictedLabels==i,:] - centroids[i], axis=0)

    VLAD_feature = VLAD_feature.flatten()

    VLAD_feature = np.sign(VLAD_feature) * np.sqrt(np.abs(VLAD_feature)) 

    VLAD_feature = VLAD_feature/np.linalg.norm(VLAD_feature)
    return VLAD_feature

def create_tree(SIFT_extractor, codebook, database_path):
    database_VLAD = list()
    database_name = list()  
    img_paths = os.listdir(database_path)
    img_paths = sorted(img_paths, key=lambda x: int(x.split('.')[0]))
    print("Constructing VLAD:")
    for img_name in tqdm(img_paths):
        img = cv2.imread(database_path + img_name)
        kp, des = SIFT_extractor.detectAndCompute(img, None)
        VLAD = get_VLAD(des, codebook)
        database_VLAD.append(VLAD)
        database_name.append(img_name)

    database_VLAD = np.asarray(database_VLAD)
    tree = BallTree(database_VLAD, leaf_size = 60)
    pickle.dump([tree, database_name], open('tree.pkl', 'wb'))
    return tree, database_name

def find_target(SIFT_extractor, query_path, codebook, tree, database_name):
    value_list = list()
    dist_list = list()
    indeces = []
    query_paths = os.listdir(query_path)
    query_paths = sorted(query_paths, key=lambda x: int(x.split('.')[0]))


    print("Querying:")
    for img_name in tqdm(query_paths):
        query = cv2.imread(query_path + img_name)
        q_kp, q_des = SIFT_extractor.detectAndCompute(query, None)
        query_VLAD = get_VLAD(q_des, codebook).reshape(1, -1)

        dist, index = tree.query(query_VLAD, 1)

        if index is None:
            index[0][0] = -1
            value_name = ""
        else:
            value_name = database_name[index[0][0]]
        
        indeces.append(index)
        value_list.append(value_name)
        dist_list.append(dist)
    
    return indeces, value_list
    
def match_images(query_path, value_list, database_path, position_path, indeces):
    positions = np.loadtxt(position_path, delimiter=',')
    recovered_imgs = []
    query_imgs = []
    query_paths = os.listdir(query_path)
    query_paths = sorted(query_paths, key=lambda x: int(x.split('.')[0]))
    indeces_filtered = []

    query_paths_filtered = []
    value_list_filtered = []
    for i in range(len(query_paths)):
        if value_list[i] == "" or value_list[i] is None:
            print("skipped")
            continue
        else:
            img = cv2.imread(os.path.join(query_path, query_paths[i]))
            query_imgs.append(img[:,:,::-1])
            recovered_imgs.append(cv2.imread(os.path.join(database_path, value_list[i]))[:,:,::-1])
            query_paths_filtered.append(query_paths[i])
            value_list_filtered.append(value_list[i])
            indeces_filtered.append(indeces[i][0])
            
    num_samples = len(recovered_imgs)

    fig, axs = plt.subplots(num_samples,2,figsize=(6,10))

    for idx,(i,j) in enumerate(zip(query_imgs,recovered_imgs)):
        #Plot
        axs[idx,0].imshow(i)
        axs[idx,1].imshow(j)
        
        # #Set title to ref img names
        axs[idx,0].title.set_text("Query: " + query_paths_filtered[idx])
        axs[idx,1].title.set_text("Match: " + value_list_filtered[idx])
        # x = positions[indeces_filtered[idx]][0][0]
        # y = positions[indeces_filtered[idx]][0][1]
        # z = positions[indeces_filtered[idx]][0][2]
        # print(x, y, z)
        
        # #Tidy things up
        axs[idx,0].set_yticks([])
        axs[idx,0].set_xticks([])
        axs[idx,1].set_yticks([])
        axs[idx,1].set_xticks([])
        
    fig.tight_layout()
    # plt.savefig('results.jpg')
    plt.show(block=False)
    return positions, np.asarray(indeces_filtered).flatten()

def plot_positions(positions, goal_indeces):
    fig = plt.figure(figsize=(6, 6))
    plt.plot(positions[:, 0], positions[:, 1], 'b')
    plt.scatter(positions[goal_indeces, 0], positions[goal_indeces, 1], c='r')
    count = 0
    for idx in goal_indeces:
        plt.annotate(str(count), (positions[idx][0]+1, positions[idx][1]+1))
        count += 1
    plt.show()