__author__ = "Brynne Lycette"

import json
import numpy as np
import pandas as pd
import game_contentBased

def get_gamesOwned(user_data):
    """
    :param user_data: path to user data
    :return: dictionary {user: list of games owned}
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
                        games.append(chunk['name'])

            gamesOwned[user] = games

    return gamesOwned

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

    ks = [10, 20, 30, 40]
    scores = dict()
    results = dict()
    
    for k in ks:
        c = score_clusters(k)
        scores[k] = c
        results[k] = np.mean(c.values())
