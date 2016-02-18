__author__ = 'Brynne Lycette'

import json
from sklearn.feature_extraction import DictVectorizer
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from kmodes import kmodes

def get_game_profiles():
    """
    - game ID
    - achievements (binary)
    - price/is_free
    - platforms
    - genres
    - categories
    - game description LDA
    - publishers
    - release date

    # k Means
    # User - content profiles
    """
    
    #game_jsons = list()
    with open('game_dataset.txt', 'r') as inFile:
        game_jsons = inFile.readlines()

    game_profiles = list()
    #print len(game_jsons)
    
    for game in game_jsons:
            game_json = json.loads(game.strip())
            
            game_features = dict()
            
            details = game_json['details']
            schema = game_json['schema']['game']
            gameID = str(game_json['appid'])
            name = game_json['name']
            #print gameID

            if details[gameID]['success'] != False: #and len(schema) != 0:
                data = details[gameID]['data']
                keys = data.keys()


                if 'achievements' in keys:
                    game_features['achievs'] = (0 if data['achievements']['total'] == 0 else 1)
                if 'price' in keys:
                    game_features['price'] = float(data['price_overview']['initial'])
                if 'genres' in keys:
                    game_features['genres'] = [g['description'] for g in data['genres']] #list
                #else:
                    #game_features['genres'] = []
                    
                # Developers
                if 'developers' in keys:
                    game_features['dev'] = data['developers']
                #else:
                    #game_features['dev'] = ''
                    
                game_features['pub'] = data['publishers']
                game_features['free'] = int(data['is_free'])

                # Categories
                if 'categories' in keys:
                    game_features['cat'] = [c['description'] for c in data['categories']] #list
               # else:
                    #game_features['cat'] = []
                    
                game_features['year'] = data['release_date']['date'][-4:]

                game_features['appID'] = gameID
                game_features['name'] = name

                game_profiles.append(game_features)
                #game_dict[name] = game_features


    return game_profiles
            

def preprocess(df):

    # Encoding genres
    genres_all = df['genres'].tolist()
    genres = set()
    for entry in genres_all:
        if type(entry) != float:
            for g in entry:
                genres.add(g)

    # initialize binary genre columns
    for genre in genres:
        df[genre] = 0

    for i in range(len(df)):
        gs = df['genres'].ix[i]
        if type(gs) != float:
            for g in gs:
                #df[g].ix[i] = 1
                df.set_value(i, g, 1)
                
    #output = df.copy()
    categorical = ['dev', 'pub', 'cat', 'achievs', 'free', 'year'] #, 'genres']

    for cat in categorical:
        le = LabelEncoder()
        df[cat] = le.fit_transform(df[cat])
        #le.inverse_transform(df[cat])

    return df
        
def get_kmodes(n):        
    game_profiles = get_game_profiles()
    df = pd.DataFrame(game_profiles)

    df_clean = preprocess(df)

    # drop 'genres' column
    df_x = df.copy()
    df_x = df_x.drop('genres', 1)
    df_x = df_x.drop('name', 1)
    df_x = df_x.drop('appID', 1)

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

    df = get_kmodes(30)

    print '\n\n'
    print df.loc[df['cluster'] == 20]['name'] #replace number for any given cluster
