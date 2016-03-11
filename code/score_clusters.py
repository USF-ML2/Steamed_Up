__author__ = "Brynne Lycette"

import json
import numpy as np
import pandas as pd
import game_contentBased_kModes
import random

def get_gamesOwned(user_data):
    """
    :param user_data: path to user data
    :return: dictionary {user: list of games owned & played > 0min}
    """
    gamesOwned = dict()
    
    with open(user_data, 'r') as inFile:
        for line in inFile:
            info = json.loads(line.strip())
            user = info['user']

            # Check if public profile
            public = len(info['ownedGames']['response']) is not 0
            games = list()
            
            if public:
                count = info['ownedGames']['response']['game_count']
                if count != 0:
                    games_json = info['ownedGames']['response']['games']
                    for chunk in games_json:
                        if chunk['playtime_forever'] > 0:
                            games.append(chunk['name'])

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


def cross_validate(k, n):
    """
    :param k n: number of clusters for kmodes | number of user to check the libraries of
    :return: list of accuracies for each of n users, 
    """
    missing = set()
    
    ### CLUSTERING###
    df = game_contentBased.get_kmodes(k)
    print "Done clustering."

    ### COLLECTING LIBRARIES ###
    gamesOwned = get_gamesOwned('userData.txt')
    print "Done collecting ownedGames."

    games = get_games('gameData.txt')

    ### VALIDATING ###
    all_accur = list()
    
    for i in range(n):
        # Randomly select user's library
        user = random.choice(gamesOwned.values())
        
        # Split library 80-20
        train = list()
        test = list()

        # Randomly split user's library 80% train, 20% test
        choice = np.random.uniform(size = len(user))
        for i in range(len(user)):
            # Check if game in gameData
            if user[i] in games:
                if choice[i] <= 0.8:
                    train.append(user[i])
                else:
                    test.append(user[i])
            else:
                missing.add(user[i])

        # Given all games in training set, what percentage of games in
        # test set have been recommended
        rec = list()
        
        for game in train:
            cluster = int(df.loc[df['name'] == game]['cluster'].tolist()[0])
            # Add all games within this cluster to the recommended list
            rec += df.loc[df['cluster'] == cluster]['name'].tolist()
            
        # Out of all test games, what percent rec'd
        recd = [1 for x in test if x in rec]
        acc = (sum(recd)+1)/(float(len(test))+1)

        all_accur.append(acc)

    return all_accur, len(missing)
            

def score_clusters(k):
    """
    :param k: number of clusters for kmodes
    :return: dictionary of {clusters: average score}

    Each game within a cluster is given a score:
        the average number of other games in the cluster owned
        by a user who owns the initial game

    A cluster's score, is then the average of the game scores within it
    """

    ### CLUSTERING###
    df = game_contentBased.get_kmodes(k)
    print "Done clustering."

    ### COLLECTING LIBRARIES ###
    gamesOwned = get_gamesOwned('userData.txt')
    print "Done collecting ownedGames."

    ### SCORING ###
    clusters = dict()
    for clust in xrange(k):
        print "Cluster", clust
        clust_games = df.loc[df['cluster'] == clust]['name'].tolist()

        clust_games_avgs = list()
        
        for game in clust_games:
            # For each user with game in library
            tot_users = 0
            tot_otherGames = 0
            for user in gamesOwned:
                other_count = 0
                if game in gamesOwned[user]:
                    tot_users += 1 # a user owns the game
                    # Count how many of the other games in the cluster they have
                    for g in clust_games:
                        if g in gamesOwned[user]: other_count += 1
                    #print other_count,"similar games owned by user"
                # sum of other games also owned
                tot_otherGames += other_count 
            if tot_otherGames != 0:
                clust_games_avgs.append(tot_otherGames/float(tot_users))
            else:
                clust_games_avgs.append(0)
                
        #print clust_games_avgs

        clusters[clust] = np.mean(clust_games_avgs)

    return clusters
                                  
    
if __name__ == '__main__':

    ks = [10, 15, 20, 30, 40, 50]
    accs = dict()
    
    for k in ks:
        acc, count_miss = cross_validate(k, 10000)
        accs[k] = np.mean(acc)

    print count_miss
    print accs
    
