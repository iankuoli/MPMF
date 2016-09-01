import os
import numpy as np
import pickle

import LoadFile
import ManifoldPMF
import Measure

"""
Test Type = >
 1: toy graph
 2: JAIN
 3: IRIS
 4: YEAST
 5: Last.fm
 6: UCL Million Song Dataset
 7: The Echo Nest Taste Profile Subset  (http://labrosa.ee.columbia.edu/millionsong/tasteprofile)
 8: MovieLens 20M Dataset  (http://grouplens.org/datasets/movielens/)
 9: Last.fm Dataset - 360K users  (http://www.dtic.upf.edu/~ocelma/MusicRecommendationDataset/index.html)
 10: R1 - Yahoo! Music User Ratings of Musical Artists, version 1.0  (http://webscope.sandbox.yahoo.com/myrequests.php)
 11: Book-Crossing Dataset  (http://www2.informatik.uni-freiburg.de/~cziegler/BX/)
 12: Amazon product data  (http://jmcauley.ucsd.edu/data/amazon/)
 """
TEST_TYPE = 8

""" Enviornment = > 1: OSX 2: Windows """
ENV = 2

""" ^^^---------- Finish: Global settings ----------^^^ """

""" number of topics """
k = 8

meta_info = "'"
if TEST_TYPE == 1:

    meta_info = "SmallToy"

    if ENV == 1:
        matX, matX_test, matX_valid = LoadFile.load_small_toy("/Users/iankuoli/Dataset/small_toy/toy_graph.csv")
    elif ENV == 2:
        matX, matX_test, matX_valid = LoadFile.load_small_toy("/home/iankuoli/dataset/small_toy/toy_graph.csv")

elif TEST_TYPE == 2:

    # Read JAIN
    meta_info = "JAIN"

    if ENV == 1:
        matX, vecLabel = LoadFile.load_small_toy("/Users/iankuoli/Dataset/jain.csv")
    elif ENV == 2:
        matX, vecLabel = LoadFile.load_small_toy("/home/iankuoli/dataset/jain.csv")

elif TEST_TYPE == 3:

    # Read IRIS
    meta_info = "Iris"

    if ENV == 1:
        matX, vecLabel = LoadFile.load_iris('/Users/iankuoli/Dataset/IRIS/iris.data')
    elif ENV == 2:
        matX, vecLabel = LoadFile.load_iris('/home/iankuoli/dataset/IRIS/iris.data')

elif TEST_TYPE == 4:

    # Read YEAST
    meta_info = "Yeast"

    if ENV == 1:
        matX, vecLabel = LoadFile.load_yeast('/Users/iankuoli/Dataset/YEAST/yeast.data')
    elif ENV == 2:
        matX, vecLabel = LoadFile.load_yeast('/home/iankuoli/dataset/YEAST/yeast.data')

elif TEST_TYPE == 5:

    # Read Last.fm data(User - Item - Word)
    meta_info = "LastFm"

    if ENV == 1:
        item_file_path = '/Users/iankuoli/Dataset/LastFm2/artists2.txt'
        word_file_path = '/Users/iankuoli/Dataset/LastFm2/tags.dat'
        UI_file_path = '/Users/iankuoli/Dataset/LastFm2/user_artists.dat'
        UIW_file_path = '/Users/iankuoli/Dataset/LastFm2/user_taggedartists.dat'
    elif ENV == 2:
        item_file_path = '/home/iankuoli/dataset/LastFm2/artists2.txt'
        word_file_path = '/home/iankuoli/dataset/LastFm2/tags.dat'
        UI_file_path = '/home/iankuoli/dataset/LastFm2/user_artists.dat'
        UIW_file_path = '/home/iankuoli/dataset/LastFm2/user_taggedartists.dat'

        # [U, D, W] = LoadLastFmData(item_filepath, word_filepath, UI_filepath, UIW_filepath)

elif TEST_TYPE == 6:

    # UCL Million Song Dataset
    # This dataset does not provide the user-item relationship
    meta_info = "UCLSongDB"

    if ENV == 1:
        matX = LoadFile.load_UCI_MSD('/Users/iankuoli/Dataset/UCI_MillionSongDataset/YearPredictionMSD.txt')
    elif ENV == 2:
        matX = LoadFile.load_UCI_MSD('/home/iankuoli/dataset/UCI_MillionSongDataset/YearPredictionMSD.txt')
elif TEST_TYPE == 7:

    # ----- The Echo Nest Taste Profile Subset -----
    # 1,019,318 unique users
    # 384,546 unique MSD songs
    # 48,373,586 user - song - play count triplets
    meta_info = "EchoNest"

    if ENV == 1:
        matX, matX_test, matX_valid = LoadFile.load_EchoNest('/Users/iankuoli/Dataset/EchoNest/train_triplets.txt')
    elif ENV == 2:
        matX, matX_test, matX_valid = LoadFile.load_EchoNest('/home/iankuoli/dataset/EchoNest/train_triplets.txt')

elif TEST_TYPE == 8:

    # ----- MovieLens 20M Dataset -----
    # 138,000 users
    # 27, 000movies
    # 20 million ratings
    # 465, 000 tag applications
    meta_info = "MovieLens20M"

    if ENV == 1:
        matX, matX_test, matX_valid = LoadFile.load_MovieLens('/Users/iankuoli/Dataset/MovieLens_20M/ratings.csv')
    elif ENV == 2:
        matX, matX_test, matX_valid = LoadFile.load_MovieLens('/home/iankuoli/dataset/MovieLens_20M/ratings.csv')

elif TEST_TYPE == 9:

    # ----- Last.fm Dataset - 360K users -----
    # 360,000 users
    # ??? artists
    # ??? consumings
    meta_info = "LastFm360K"

    if ENV == 1:
        matX, matX_test, matX_valid, dict_user2id, dict_id2item = LoadFile.load_MovieLens(
            '/Users/iankuoli/Dataset/MovieLens_20M/ratings.csv')
    elif ENV == 2:
        matX, matX_test, matX_valid, dict_user2id, dict_id2item = LoadFile.load_MovieLens(
            '/home/iankuoli/dataset/MovieLens_20M/ratings.csv')

elif TEST_TYPE == 10:

    # ----- R1 - Yahoo! Music User Ratings of Musical Artists, version 1.0 -----
    # ??? users
    # ??? artists
    # ??? consumings
    meta_info = "YahooR1"

    if ENV == 1:
        matX, matX_test, matX_valid = LoadFile.load_YahooR1(
            '/Users/iankuoli/Dataset/Yahoo_R1/ydata-ymusic-user-artist-ratings-v1_0.txt')
    elif ENV == 2:
        matX, matX_test, matX_valid = LoadFile.load_YahooR1(
            '/home/iankuoli/dataset/Yahoo_R1/ydata-ymusic-user-artist-ratings-v1_0.txt')

elif TEST_TYPE == 11:

    # ----- Book-Crossing Dataset -----
    # 278,858 users
    # 271,379 books
    # 1,149,780 ratings
    meta_info = "BookCrossDB"

    if ENV == 1:
        matX, matX_test, matX_valid = LoadFile.load_BX('/Users/iankuoli/Dataset/BX-CSV-Dump/BX-Book-Ratings.csv')
    elif ENV == 2:
        matX, matX_test, matX_valid = LoadFile.load_BX('/home/iankuoli/dataset/BX-CSV-Dump/BX-Book-Ratings.csv')

elif TEST_TYPE == 12:

    # ----- Amazon product data -----
    # ??? users
    # ??? books
    # ??? ratings

    dict_type = {1: "ratings_CDs_and_Vinyl.csv",
                 2: "ratings_Clothing_Shoes_and_Jewelry.csv",
                 3: "ratings_Pet_Supplies.csv",
                 4: "ratings_Baby.csv"
                 }
    meta_info = "Amazon_" + 1

    if ENV == 1:
        matX, matX_test, matX_valid = LoadFile.load_Amazon('/Users/iankuoli/Dataset/amazon/', dict_type[1], '.csv')
    elif ENV == 2:
        matX, matX_test, matX_valid = LoadFile.load_Amazon('/home/dataset/amazon/', dict_type[1], '.csv')

M, N = matX.shape


# Initial the model
MPMF = pickle.load(open("../model/MPMF_", meta_info, ".p", 'rb'))

# Recommendation
list_topK = [5, 10, 15, 20, 30, 40, 50]
vec_query_user_index = np.zeros(0)
mat_rec_items = MPMF.recommend_for_users(vec_query_user_index)

list_precision_topk = []
list_recall_topk = []
for k in list_topK:
    sum_precision = 0
    sum_recall = 0

    for u in range(len(vec_query_user_index)):
        user = vec_query_user_index[u]
        precision, recall = Measure.precision_recall_at_k(matX[user, :], mat_rec_items[u, :], k)
        sum_precision += precision
        sum_recall += recall

    list_precision_topk.append(sum_precision / len(vec_query_user_index))
    list_recall_topk.append(sum_recall / len(vec_query_user_index))
