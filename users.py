__author__ = 'MegEllis'

import json
import collections
import numpy as np

import os
import sys

# Path for spark source folder
os.environ['SPARK_HOME']= "/Users/MegEllis/Desktop/spark-1.6.0-bin-hadoop2.6_2"

sys.path.append("/Users/MegEllis/Desktop/spark-1.6.0-bin-hadoop2.6_2/python/")

from pyspark import SparkConf, SparkContext
conf = (SparkConf().setMaster("local").setAppName("My app").set("spark.executor.memory", "1g"))
sc = SparkContext(conf = conf)

users_open = open('/Users/MegEllis/Desktop/aml_proj/userData.txt', 'r+')

def nest_dict(filename):
    listed_dict = []
    for line in filename:
        json_lines = json.loads(line)
        rec_dict = collections.defaultdict(dict)
        user = json_lines['user']
        in_response = json_lines['ownedGames']['response']
        if 'games' in in_response:
            for i in in_response['games']:
                rec_dict[user][i['name']] = i['playtime_forever']
                listed_dict.append(rec_dict)
    return listed_dict


final_list = nest_dict(users_open)

final = sc.parallelize(final_list)















# game_user_dict = {}
# for i in range(len(final)):
#     for j in range(len(final[i].values()[0])):
#         if final[i].values()[0].keys()[j] not in game_user_dict.keys():
#             game_user_dict[final[i].values()[0].keys()[j]] = [final[i].values()[0].values()[j]]
#         else:
#             game_user_dict[final[i].values()[0].keys()[j]].append([final[i].values()[0].values()[j]])



#def users_per_game:
