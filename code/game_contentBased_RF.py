__author__ = 'Brynne Lycette'

import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import random
import build_user_dataset

from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.regression import LabeledPoint
import os
import sys

os.environ['SPARK_HOME'] = "~/brynne/Downloads/spark-1.6.0-bin-hadoop2.6"
sys.path.append("~/brynne/Downloads/spark-1.6.0-bin-hadoop2.6_2/python/")

from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.sql import SQLContext, Row
conf = (SparkConf().setMaster("local").setAppName("My app").set("spark.executor.memory", "1g"))
sc = SparkContext(conf = conf)
sqlContext = SQLContext(sc)

def convertUni(uni):
    st = uni.encode('ascii', 'ignore')
    return st.replace('"', '')


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
    with open('/media/sf_AdvancedML/Final/gameData.txt', 'r') as inFile:
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

                game_features['appID'] = gameID
                game_features['name'] = name.encode('ascii', 'ignore')

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
                game_features['tags'] = tags

                if 'type' in keys and game_features['type'] != 'video':
                    game_profiles.append(game_features)
                elif 'type' not in keys:
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

def get_gamesOwned(user_data):
    """
    :param user_data: path to user data
    :return: dictionary {user: {game: playtime}}
    """
    gamesOwned = dict()
    
    with open(user_data, 'r') as inFile:
        for line in inFile:
            info = json.loads(line.strip())
            user = info['user']

            # Check if public profile
            public = len(info['ownedGames']['response']) is not 0
            games = dict()
            
            if public:
                count = info['ownedGames']['response']['game_count']
                if count != 0:
                    games_json = info['ownedGames']['response']['games']
                    for chunk in games_json:
                        if chunk['playtime_forever'] > 0:
                            if user not in games.keys():
                                games[user] = {chunk['name'] : chunk['playtime_forever']}
                            else:
                                name = chunk['name']
                                playtime = chunk['playtime_forever']
                                games[user][name] = playtime

            gamesOwned[user] = games

    return gamesOwned


def get_games(game_path):
    """
    :param game_path: path to gameData.txt
    :return: list of games
    """
    games = list()
    with open(game_path, 'r') as inFile:
        game_jsons = inFile.readlines()

    for game in game_jsons:
        game_json = json.loads(game.strip())

        details = game_json['details']
        gameID = str(game_json['appid'])
        name = game_json['name']

        if details[gameID]['success'] != False:
            games.append(name)
        
    return games


def cross_validate(df_clean, games, n):
    """
    :param k n: number of users for CV
    :return: list of accuracies for each of n users, avg acc
    """
    missing = set()

    ### COLLECTING LIBRARIES ###
    gamesOwned = get_gamesOwned('/media/sf_AdvancedML/Final/userData.txt')
    print "Done collecting ownedGames."

    ### VALIDATING ###
    all_accur = {'model1': [], 'model2': [], 'model3': [], 'model4': []}

    for i in range(n):
        # initialize empty frame with appropriate columns
        df = pd.DataFrame(columns = list(df_clean.columns.values)+['playtime'])

        # Randomly select user's library
        user = random.choice(gamesOwned.values())

        # Connect playtime to game df for games owned
        if len(user.values()) > 0:
            #print user.values()[0]
            for k, v in user.values()[0].iteritems():
                if k in games:
                    row = df_clean.loc[df_clean['name'] == k]
                    row['playtime'] = np.log(v)
                    df = df.append(row)
                else:
                    missing.add(k)

        df = df.drop('genres', 1)
        df = df.drop('name', 1)
        df = df.drop('appID', 1)
        df = df.drop('cat', 1)
        df = df.drop('tags', 1)
        df = df.drop('type', 1)

        # Pass User DF to Spark
        df.to_csv('/media/sf_AdvancedML/Final/RF_train.csv')

        data = sc.textFile('/media/sf_AdvancedML/Final/RF_train.csv')
        header = data.first()
        data = data.filter(lambda x: x != header)
        data = data.map(lambda line: convertUni(line))
        data = data.map(lambda line: line.split(','))

        # RDD of (label, features) pairs
        data = data.map(lambda line: LabeledPoint(line[-1], line[:len(line)]))

        # Split into training, test
        (trainingData, testData) = data.randomSplit([0.8, 0.2])

        try:
            # Training model
            model1 = RandomForest.trainRegressor(trainingData, categoricalFeaturesInfo = {},
                                                numTrees = 70, featureSubsetStrategy = "auto",
                                                impurity = 'variance', maxDepth = 4)
            model2 = RandomForest.trainRegressor(trainingData, categoricalFeaturesInfo = {},
                                                numTrees = 100, featureSubsetStrategy = "auto",
                                                impurity = 'variance', maxDepth = 4)
            model3 = RandomForest.trainRegressor(trainingData, categoricalFeaturesInfo = {},
                                                numTrees = 120, featureSubsetStrategy = "auto",
                                                impurity = 'variance', maxDepth = 4)
            model4 = RandomForest.trainRegressor(trainingData, categoricalFeaturesInfo = {},
                                                numTrees = 100, featureSubsetStrategy = "auto",
                                                impurity = 'variance', maxDepth = 6)

            models = [model1, model2, model3, model4]
            modelNames = ['model1', 'model2', 'model3', 'model4']
            for i in range(len(models)):
                m = models[i]
                name = modelNames[i]
                # Evaluate on test data, compute error
                predictions = m.predict(testData.map(lambda x: x.features))
                labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
                testMSE = labelsAndPredictions.map(lambda (v, p) : (v-p)*(v-p)).sum() /\
                          float(testData.count())

                all_accur[name] += [testMSE]

        except:
            pass

    avgDict = {}
    for k,v in all_accur.iteritems():
        avgDict[k] = np.mean(v)
    return all_accur, avgDict


def rf(userID, n):
    
    ### CREATING GAME PROFILE DF ####
    game_profiles = get_game_profiles()
    df = pd.DataFrame(game_profiles)
    df_clean = preprocess(df)

    # Full df for games only, no playtimes (for prediction later)
    df_games = df_clean.drop('genres', 1)
    #df_games = df_games.drop('name', 1) 
    df_games = df_games.drop('appID', 1)
    df_games = df_games.drop('cat', 1)
    df_games = df_games.drop('tags', 1)
    df_games = df_games.drop('type', 1)


    games = get_games('/media/sf_AdvancedML/Final/gameData.txt')
    missing = set()

    ### CROSS VALIDATING ###    
    all_accur, avg_accur = cross_validate(df_clean, games, 10)
    print "Accuracies, Average Accuracy"
    print all_accur, avg_accur

    ### TRAIN ON INCOMING USER ###
    ownedGames = build_user_dataset.get_ownedGames(userID) #json object
    with open('/media/sf_AdvancedML/Final/userData'+str(userID)+'.txt', 'w') as outFile:
        if len(ownedGames) == 0:
            print "This user's library is empty or unreachable."
            return
        json.dump({'user': userID, 'ownedGames':ownedGames}, outFile)

    # initialize empty frame with appropriate columns
    df = pd.DataFrame(columns = list(df_clean.columns.values)+['playtime'])

    # Randomly select user's library
    gamesOwned = get_gamesOwned('/media/sf_AdvancedML/Final/userData'+str(userID)+'.txt')
    user = random.choice(gamesOwned.values())
    gamesList = gamesOwned[gamesOwned.keys()[0]].keys()

    # Connect playtime to game df for games owned
    if len(user.values()) > 0:
        #print user.values()[0]
        for k, v in user.values()[0].iteritems():
            if k in games:
                row = df_clean.loc[df_clean['name'] == k]
                row['playtime'] = np.log(v)
                df = df.append(row)
            else:
                missing.add(k)

    df = df.drop('genres', 1)
    df = df.drop('name', 1)
    df = df.drop('appID', 1)
    df = df.drop('cat', 1)
    df = df.drop('tags', 1)
    df = df.drop('type', 1)

    # Pass User DF to Spark
    df.to_csv('/media/sf_AdvancedML/Final/RF.csv')

    data = sc.textFile('/media/sf_AdvancedML/Final/RF.csv')
    header = data.first()
    data = data.filter(lambda x: x != header)
    data = data.map(lambda line: convertUni(line))
    data = data.map(lambda line: line.split(','))

    # RDD of (label, features) pairs
    data = data.map(lambda line: LabeledPoint(line[0], line[1:]))

    model = RandomForest.trainRegressor(data, categoricalFeaturesInfo = {},
                                        numTrees = 3, featureSubsetStrategy = "auto",
                                        impurity = 'variance', maxDepth = 4)

    ### PREDICT ###
    # for every game in Steam library #
    df_games.to_csv('/media/sf_AdvancedML/Final/RF_games_names.csv')
    df_games.drop('name', 1).to_csv('/media/sf_AdvancedML/Final/RF_games.csv')

    data_games = sc.textFile('/media/sf_AdvancedML/Final/RF_games.csv')
    header = data_games.first()
    data_games = data_games.filter(lambda x: x != header)
    data_games = data_games.map(lambda line: convertUni(line))
    data_games = data_games.map(lambda line: line.split(','))

    data_test = sc.textFile('/media/sf_AdvancedML/Final/RF_games_names.csv')
    header2 = data_test.first()
    data_test = data_test.filter(lambda x: x != header2)
    data_test = data_test.map(lambda line: convertUni(line))
    data_test = data_test.map(lambda line: line.split(','))
    
    predictions = model.predict(data_games)
    idPredictions = data_test.map(lambda x: x[6]).zip(predictions)

    # Filter predictions for games owned or trailers/apps
    idPredictions = idPredictions.filter(lambda x: x[0] not in gamesList)

    # Export predictions to pandas df
    predDF = idPredictions.toDF()
    predDF = predDF.toPandas()  # Name, Prediction
    predDF.columns = ['Name', 'PredictedPlaytime']

    # Returning top n not in library
    sorted_predDF = predDF.sort_values(by = 'PredictedPlaytime', ascending = False)
    recs = []
    #while len(recs) <= n:
        # check if rec in library
        #game = 
        # check if game or trailer/app

    return sorted_predDF[:n]
    
    

if __name__ == '__main__':

    recs = rf('76561198043891906', 10)
    print recs
    #scores = cross_validate(1000)
    #76561197991599169
