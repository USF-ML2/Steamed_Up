__author__ = "Paul Thompson"
### This file extracts features from the games and fits a linear regression model based on a single user's
### game play times. Recommendations are made for that user by predicting play time for the games not played yet
### by the user and sorting from greatest to least play time.
import sys, json, pandas, numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn import linear_model

game_path = "/Users/paulthompson/Documents/MSAN_Files/Spr1_AdvML/Final_Proj/game_dataset.txt"
user_path = "/Users/paulthompson/Documents/MSAN_Files/Spr1_AdvML/Final_Proj/userData.txt"
appID_path = "/Users/paulthompson/Documents/MSAN_Files/Spr1_AdvML/Final_Proj/appIDs.json"

def create_game_profile_df(game_data_path):
    '''
    Parses the game data json file and returns a pandas data frame with both continuous and dummy variables
    :param game_data_path:
    :return: panda data frame with game features
    '''
    with open(game_data_path) as f:
        gameFeatDicts = []
        gameCount = 0
        for line in f:
            record = json.loads(line)
            if 'data' in record['details'][record['details'].keys()[0]].keys():
                gameFeatDict = {}
                try:
                    # print record['details'][record['details'].keys()[0]]['data']
                    gameFeatDict['steam_appid'] = record['appid']
                except:
                    continue
                try:
                    gameFeatDict['mac'] = record['details'][record['details'].keys()[0]]['data']['platforms']['mac']
                except:
                    continue
                try:
                    gameFeatDict['windows'] = record['details'][record['details'].keys()[0]]['data']['platforms']['windows']
                except:
                    continue
                try:
                    gameFeatDict['linux'] = record['details'][record['details'].keys()[0]]['data']['platforms']['linux']
                except:
                    continue
                try:
                    gameFeatDict['gameType'] =  record['details'][record['details'].keys()[0]]['data']['type']
                except:
                    continue
                try:
                    gameFeatDict['releaseYear'] = int(record['details'][record['details'].keys()[0]]['data']['release_date']['date'][-4:])
                except:
                    continue
                try:
                    gameFeatDict['isFree'] = record['details'][record['details'].keys()[0]]['data']['is_free']
                except:
                    continue
                try:
                    gameFeatDict['metacriticScore'] = record['details'][record['details'].keys()[0]]['data']['metacritic']['score']
                except:
                    continue
                try:
                    gameFeatDict['developer'] = record['details'][record['details'].keys()[0]]['data']['developers'][0]
                except:
                    continue
                try:
                    gameFeatDict['requiredAge'] = int(record['details'][record['details'].keys()[0]]['data']['required_age'])
                except:
                    continue
                try:
                    gameFeatDict['genre'] = record['details'][record['details'].keys()[0]]['data']['genres'][0]['description']
                except:
                    continue
                try:
                    categories = record['details'][record['details'].keys()[0]]['data']['categories']
                    allowedCategories = ['Single-player', 'Co-op', 'Multi-player', 'MMO', 'Local Co-op']
                    for category in categories:
                        if str(category['description']) in allowedCategories:
                            gameFeatDict[str(category['description'])] = 'True'
                except:
                    continue
                try:
                    gameFeatDict['fullPrice'] = \
                        record['details'][record['details'].keys()[0]]['data']['price_overview']['initial']
                except:
                    continue
                gameFeatDicts.append(gameFeatDict)
                gameCount += 1
            else:
                continue
            # if gameCount == 40:
            #     # print gameFeatDicts
            #     vec = DictVectorizer()
            #     gameFeatures = vec.fit_transform(gameFeatDicts).toarray()
            #     gameFeaturesNames = vec.get_feature_names()
            #     gameFeaturesDF = pandas.DataFrame(gameFeatures, columns = gameFeaturesNames)
            #     # print gameFeaturesDF
            #     sys.exit()
        vec = DictVectorizer()
        gameFeatures = vec.fit_transform(gameFeatDicts).toarray()
        gameFeaturesNames = vec.get_feature_names()
        gameFeaturesDF = pandas.DataFrame(gameFeatures, columns = gameFeaturesNames)
        gameFeaturesDF.index = gameFeaturesDF['steam_appid']
        gameFeaturesDF = gameFeaturesDF.drop(['steam_appid'], axis=1)
        return gameFeaturesDF

## User Information
def get_user_games(user_data_path):
    '''
    :param user_data_path:
    :return: list of games played by user including play times
    '''
    with open(user_data_path) as f:
        linecount = 0
        for line in f:
            record = json.loads(line)
            if record['ownedGames']['response'].keys():
                gamesPlayedList = []
                for game in record['ownedGames']['response']['games']:
                    if game['playtime_forever'] <> 0:
                        gameinfo = {}
                        gameinfo['playtime_forever'] = game['playtime_forever']
                        gameinfo['name'] = game['name']
                        gameinfo['appid'] = game['appid']
                        gamesPlayedList.append(gameinfo)
                linecount += 1
            if linecount == 1:
                return gamesPlayedList

def addUserPlaytime_gameProfiles():
    '''
    :return: Game profile pandas data frame with an additional column for user play times
    '''
    gameProfiles = create_game_profile_df(game_path)
    gameProfiles['playtime'] = pandas.Series(np.zeros(len(gameProfiles)), index = gameProfiles.index)
    playedGames = get_user_games(user_path)
    gameIDs = []
    for game in playedGames:
        try:
            gameProfiles['playtime'][game['appid']] = game['playtime_forever']
            gameIDs.append(game['appid'])
        except:
            continue
    return gameProfiles, gameIDs



def RankUsingLinReg():
    '''
    :return: List of games and predicted play times sorted from greatest to least
    '''
    userGameProfiles, appIDs = addUserPlaytime_gameProfiles()
    Y_train = userGameProfiles['playtime']
    X_train = userGameProfiles.drop(['playtime'], axis=1)
    X_test = X_train.drop(appIDs)
    regr = linear_model.LinearRegression()
    regr.fit(X_train,Y_train)
    unPlayedGames = list(X_test.index)
    predictions = list(regr.predict(X_test))
    gamePredictions = []
    for i in range(len(unPlayedGames)):
        gamePredictions.append([unPlayedGames[i],predictions[i]])
    Ranking = sorted(gamePredictions, key= lambda x: x[1], reverse = True)
    return Ranking

def getAppNames():
    '''
    :return: Dictionary mapping app id to game name
    '''
    with open(appID_path) as f:
        records = json.load(f)
        IDNameDict = {}
        for game in records['applist']['apps']['app']:
            IDNameDict[str(game['appid'])] = game['name']
    return IDNameDict

if __name__ == '__main__':
    ### This returns list of user games and list of recommended games for comparison
    userGames = get_user_games(user_path)
    appIDRankings = RankUsingLinReg()
    IDNameDict = getAppNames()
    count = 0
    print "Users Current Game List:"
    for game in userGames:
        print game
    print ""
    print "Recommended Games List"
    for app in appIDRankings:
        print IDNameDict[str(int(app[0]))]
        count += 1
        if count == 20:
            sys.exit()




# allowedCategories = ['Includes Source SDK', 'Stats', 'Single-player','Cross-Platform Multiplayer',
#                                          'Captions available', 'Co-op', 'Steam Workshop', 'Partial Controller Support',
#                                          'Commentary available', 'Multi-player', 'Steam Leaderboards', 'Game demo',
#                                          'Downloadable Content', 'Valve Anti-Cheat enabled', 'Full controller support',
#                                          'Steam Trading Cards', 'Steam Achievements', 'MMO', 'Includes level editor',
#                                          'VR Support', 'Steam Cloud', 'Local Co-op']