__author__ = 'Brynne Lycette'

import json
from sklearn.feature_extraction import DictVectorizer
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from kmodes import kmodes
import sys

def get_game_profiles():
    """
    - game ID
    - achievements (binary)
    - price/is_free
    - platforms: mac, windows, linux
    - genres
    - categories
    - publishers
    - release date
    - metacritic score
    - game type

    # k Means
    # User - content profiles
    """
    
    #game_jsons = list()
    with open('gameData.txt', 'r') as inFile:
        game_jsons = inFile.readlines()

    game_profiles = list()
    #print len(game_jsons)
    
    for game in game_jsons:
            game_json = json.loads(game.strip())
            
            game_features = dict()

            details = game_json['details']
            schema = game_json['schema']
            gameID = str(game_json['appid'])
            name = game_json['name']
            tags = game_json['tags']
            #print gameID

            if details[gameID]['success'] != False: #and len(schema) != 0:
                data = details[gameID]['data']
                keys = data.keys()

                if 'type' in keys:
                    game_features['type'] = data['type']
                if 'platforms' in keys:
                    game_features['mac'] = (1 if data['platforms']['mac'] == True else 0)
                    game_features['windows'] = (1 if data['platforms']['windows'] == True else 0)
                    game_features['linux'] = (1 if data['platforms']['linux'] == True else 0)                   
                if 'achievements' in keys:
                    game_features['achievs'] = (0 if data['achievements']['total'] == 0 else 1)
                if 'price' in keys:
                    game_features['price'] = float(data['price_overview']['initial'])
                if 'genres' in keys:
                    game_features['genres'] = [g['description'] for g in data['genres']] #list
                if 'developers' in keys:
                    game_features['dev'] = data['developers'] 
                game_features['pub'] = data['publishers']
                game_features['free'] = int(data['is_free'])
                if 'categories' in keys:
                    game_features['cat'] = [c['description'] for c in data['categories']] #list 
                game_features['year'] = data['release_date']['date'][-4:]
                game_features['appID'] = gameID
                game_features['name'] = name.encode('ascii', 'ignore')
                game_features['tags'] = tags
                
                game_profiles.append(game_features)
                #game_dict[name] = game_features


    return game_profiles

def encode(df, col):
    col_list = df[col].tolist()
    columns = set()

    # Collecting unique set
    for entry in col_list:
        if type(entry) != float:
            for e in entry:
                columns.add(e)

    # Initializing binary genre columns
    for c in columns:
        col_name = col+"_"+c
        df[c] = 0

    # Encoding column values
    for n in range(len(df)):
        items = df['genres'].ix[n]
        if type(items) != float:
            for i in items:
                df.set_value(n, i, 1)

    return df


def preprocess(df):

    listed_categorical = ['genres', 'cat', 'tags', 'type']
    for lc in listed_categorical:
        df = encode(df, lc)

    #output = df.copy()
    categorical = ['dev', 'pub', 'achievs', 'free', 'year'] 

    for cat in categorical:
        le = LabelEncoder()
        df[cat] = le.fit_transform(df[cat])
        #le.inverse_transform(df[cat])

    return df
        
def get_kmodes(n):        
    game_profiles = get_game_profiles()
    df = pd.DataFrame(game_profiles)

    df_clean = preprocess(df)

    # drop columns not encoded
    df_x = df.copy()
    df_x = df_x.drop('genres', 1)
    df_x = df_x.drop('name', 1)
    df_x = df_x.drop('appID', 1)
    df_x = df_x.drop('cat', 1)
    df_x = df_x.drop('tags', 1)
    df_x = df_x.drop('type', 1)

    print df_x.ix[:1]

    # kModes clustering
    kmodes_huang = kmodes.KModes(n_clusters = n, init = 'Huang', verbose = 1)
    clusters = kmodes_huang.fit_predict(df_x)

    df['cluster'] = clusters
    #print df
    
    print 'Cluster\tSize'
    for i in xrange(n):
        print i,'\t',len(df.loc[df['cluster'] == i])

    return df

if __name__ == '__main__':

    df = get_kmodes(40)
    df.to_csv('clusteredGames_k40.csv')
    

##    game_profiles = get_game_profiles()
##    df = pd.DataFrame(game_profiles)
##    df_clean = preprocess(df)
##
##    # Pass to file to be passed into spark
##    df_clean.to_csv('KModesDF_tester.csv')

    print '\n\n'
    print df.loc[df['cluster'] == 24]['name'] #replace number for any given cluster
