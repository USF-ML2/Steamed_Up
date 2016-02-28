import sys, json, pandas, numpy as np, copy
from sklearn.feature_extraction import DictVectorizer
from sklearn import linear_model
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

game_path = "/Users/paulthompson/Documents/MSAN_Files/Spr1_AdvML/Final_Proj/gameData.txt"
user_path = "/Users/paulthompson/Documents/MSAN_Files/Spr1_AdvML/Final_Proj/userData.txt"
appID_path = "/Users/paulthompson/Documents/MSAN_Files/Spr1_AdvML/Final_Proj/appIDs.json"


def create_game_profile_df(game_data_path):
    print "Creating Game Profiles"
    with open(game_data_path) as f:
        gameFeatDicts = []
        gameCount = 0
        gameNameDict = {}
        gameNameList = []
        for line in f:
            record = json.loads(line)
            if 'data' in record['details'][record['details'].keys()[0]].keys():
                gameFeatDict = {}
                if record['details'][record['details'].keys()[0]]['data']['type'] in ['demo','dlc','movie',
                                                                                      'advertising','video']:
                    continue
                # try:
                    # print record['details'][record['details'].keys()[0]]['data']
                gameFeatDict['steam_appid'] = record['appid']
                gameNameDict[record['appid']] = record['name']
                # except:
                #     pass
                try:
                    gameFeatDict['mac'] = record['details'][record['details'].keys()[0]]['data']['platforms']['mac']
                except:
                    pass
                try:
                    gameFeatDict['windows'] = record['details'][record['details'].keys()[0]]['data']['platforms']['windows']
                except:
                    pass
                try:
                    gameFeatDict['linux'] = record['details'][record['details'].keys()[0]]['data']['platforms']['linux']
                except:
                    pass
                try:
                    gameFeatDict['type'] =  record['details'][record['details'].keys()[0]]['data']['type']
                except:
                    pass
                try:
                    gameFeatDict['releaseYear'] = int(record['details'][record['details'].keys()[0]]['data']['release_date']['date'][-4:])
                except:
                    pass
                try:
                    gameFeatDict['isFree'] = record['details'][record['details'].keys()[0]]['data']['is_free']
                except:
                    pass
                try:
                    gameFeatDict['metacriticScore'] = record['details'][record['details'].keys()[0]]['data']['metacritic']['score']
                except:
                    pass
                try:
                    gameFeatDict['developer'] = record['details'][record['details'].keys()[0]]['data']['developers'][0]
                except:
                    pass
                try:
                    gameFeatDict['requiredAge'] = int(record['details'][record['details'].keys()[0]]['data']['required_age'])
                except:
                    pass
                try:
                    categories = record['details'][record['details'].keys()[0]]['data']['categories']
                    allowedCategories = ['Single-player', 'Co-op', 'Multi-player', 'MMO', 'Local Co-op']
                    for category in categories:
                        if str(category['description']) in allowedCategories:
                            gameFeatDict[str(category['description'])] = 'True'
                except:
                    pass
                try:
                    gameFeatDict['fullPrice'] = \
                        record['details'][record['details'].keys()[0]]['data']['price_overview']['initial']
                except:
                    pass
                if record['tags'] == []:
                    try:
                        for genre in record['details'][record['details'].keys()[0]]['data']['genres']:
                            gameFeatDict[genre['description']] = 'True'
                    except:
                        pass
                if record['tags'] <> []:
                    try:
                        for tag in record['tags']:
                            gameFeatDict[tag] == 'True'
                    except:
                        pass
                gameFeatDicts.append(gameFeatDict)
                gameCount += 1
            else:
                pass
                # gameFeatDict['steam_appid'] = record['appid']
                # gameFeatDicts.append(gameFeatDict)
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
        for id in gameFeaturesDF.index:
            gameNameList.append(gameNameDict[id])

        vectorizer = TfidfVectorizer(sublinear_tf = True, max_df = 1.0, lowercase=False,
                                 stop_words = 'english', ngram_range = (1,2), analyzer='word', norm = 'l1')
        X_train_TFIDFNames = vectorizer.fit_transform(gameNameList)
        gameFeaturesDF = gameFeaturesDF.drop(['steam_appid'], axis=1)
        Y_train = pandas.Series(np.zeros(len(gameFeaturesDF)), index = gameFeaturesDF.index)
        X_train_sparse = sparse.coo_matrix(gameFeaturesDF)
        X_train_sparse2 = hstack([X_train_sparse, X_train_TFIDFNames])
        X_train = pandas.DataFrame(X_train_sparse2.todense(), index=gameFeaturesDF.index)
        print "Finished Creating Game Profiles"
        return Y_train, X_train_sparse2, X_train

def getAppNames():
    with open(appID_path) as f:
        records = json.load(f)
        IDNameDict = {}
        for game in records['applist']['apps']['app']:
            IDNameDict[str(game['appid'])] = game['name']
    return IDNameDict

## User Information
def get_user_games(user_data_path, numToRetrieve):
    with open(user_data_path) as f:
        linecount = 0; userGamesList = {}; userIDList = []

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
                userIDList.append(record['user'])
                userGamesList[record['user']] = gamesPlayedList

            if linecount == 1:
                return userIDList, userGamesList


def CrossValUsingLinReg(userGames, Y_train, X_train, X_train_sparse):
    #
    IDNameDict = getAppNames()
    IDindexDict = {}
    # "Adding user playtime to Response"
    appIDs = []
    ErrorList = [2430]
    for game in userGames:
        if game['appid'] not in ErrorList:
            try:
                Y_train[game['appid']] = game['playtime_forever']
                appIDs.append(game['appid'])
            except:
                pass
    gameCount = len(appIDs)
    recAccuracyCount = 0

    for i, id in enumerate(Y_train.index):
        IDindexDict[id] = i

    print "Linear Regression Cross-Validation"
    for i in range(gameCount):
        # "Splitting Out Training and Test Y and X"
        tempAppIDs = copy.copy(appIDs)
        del tempAppIDs[i]

        Y_temp_train = copy.deepcopy(Y_train)
        Y_temp_train[appIDs[i]] = 0
        X_test = X_train.drop(tempAppIDs, axis=0)
        unPlayedGames = list(X_test.index)
        X_test_sparse = sparse.coo_matrix(X_test)

        # "Fitting Linear Model"
        regr = linear_model.LinearRegression()
        regr.fit(X_train_sparse,Y_train)

        # "Getting Predictions"
        predictions = list(regr.predict(X_test_sparse))
        gamePredictions = []
        for k in range(len(unPlayedGames)):
            gamePredictions.append([unPlayedGames[k],predictions[k]])

        print IDNameDict[str(appIDs[i])]
        topRecommendations = sorted(gamePredictions, key= lambda x: x[1], reverse = True)[0:50]

        if appIDs[i] in np.array(topRecommendations)[:,0]:
            print "Success"
            recAccuracyCount += 1
            print recAccuracyCount
            print ""
        else:
            print "Failure"
            RecList =[]
            for id in np.array(topRecommendations)[0:10,0]:
                RecList.append(IDNameDict[str(int(id))])
            print RecList
            print ""
    print ""
    print recAccuracyCount, "of the user's games out of", gameCount, "recommended."

if __name__ == '__main__':
    print "Getting User Games"
    userIDList, userGamesList = get_user_games(user_path, numToRetrieve = 2)
    PlayTimeZeros, GameDF_sparse, GameDF = create_game_profile_df(game_path)
    selectedUserGames = userGamesList[userIDList[0]]
    CrossValUsingLinReg(selectedUserGames, PlayTimeZeros, GameDF, GameDF_sparse)

    # IDNameDict = getAppNames()
    # count = 0
    # print "Users Current Game List:"
    # for game in selectedUserGames:
    #     print game
    # print ""
    # print "Recommended Games List"
    # for app in appIDRankings:
    #     print IDNameDict[str(int(app[0]))]
    #     count += 1
    #     if count == 20:
    #         sys.exit()




# allowedCategories = ['Includes Source SDK', 'Stats', 'Single-player','Cross-Platform Multiplayer',
#                                          'Captions available', 'Co-op', 'Steam Workshop', 'Partial Controller Support',
#                                          'Commentary available', 'Multi-player', 'Steam Leaderboards', 'Game demo',
#                                          'Downloadable Content', 'Valve Anti-Cheat enabled', 'Full controller support',
#                                          'Steam Trading Cards', 'Steam Achievements', 'MMO', 'Includes level editor',
#                                          'VR Support', 'Steam Cloud', 'Local Co-op']