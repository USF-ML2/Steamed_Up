__author__ = 'Paul Thompson'

import sys, json
from sklearn.feature_extraction import DictVectorizer
import pandas
from datetime import datetime

with open("/Users/paulthompson/Documents/MSAN_Files/Spr1_AdvML/Final_Proj/game_dataset.txt") as f:
    gameFeatDicts = []
    gameCount = 0
    for line in f:
        record = json.loads(line)
        # print record.keys()
        if record['details'][record['details'].keys()[0]]['success']:
            gameFeatDict = {}
            try:
                # print record['details'][record['details'].keys()[0]]['data'].keys()
                gameFeatDict['steam_appid'] = record['details'][record['details'].keys()[0]]['data']['steam_appid']
                gameFeatDict['mac'] = record['details'][record['details'].keys()[0]]['data']['platforms']['mac']
                gameFeatDict['windows'] = record['details'][record['details'].keys()[0]]['data']['platforms']['windows']
                gameFeatDict['linux'] = record['details'][record['details'].keys()[0]]['data']['platforms']['linux']
                gameFeatDict['gameType'] =  record['details'][record['details'].keys()[0]]['data']['type']
                gameFeatDict['releaseYear'] = int(record['details'][record['details'].keys()[0]]['data']['release_date']['date'][-4:])
                gameFeatDict['isFree'] = record['details'][record['details'].keys()[0]]['data']['is_free']
                gameFeatDict['metacriticScore'] = record['details'][record['details'].keys()[0]]['data']['metacritic']['score']
                gameFeatDict['developer'] = record['details'][record['details'].keys()[0]]['data']['developers'][0]
                gameFeatDict['requiredAge'] = int(record['details'][record['details'].keys()[0]]['data']['required_age'])
                gameFeatDict['genre'] = record['details'][record['details'].keys()[0]]['data']['genres'][0]['description']
                categories = record['details'][record['details'].keys()[0]]['data']['categories']
                for category in categories:
                    gameFeatDict[str(category['description'])] = 'True'
                gameFeatDict['fullPrice'] = \
                    record['details'][record['details'].keys()[0]]['data']['price_overview']['initial']
                gameFeatDict['discountedPrice'] = \
                    record['details'][record['details'].keys()[0]]['data']['price_overview']['final']
            except:
                continue
            gameFeatDicts.append(gameFeatDict)
        gameCount += 1
        # if gameCount > 2089:
        #     vec = DictVectorizer()
        #     gameFeatures = vec.fit_transform(gameFeatDicts).toarray()
        #     gameFeaturesNames = vec.get_feature_names()
        #     gameFeaturesDF = pandas.DataFrame(gameFeatures, columns = gameFeaturesNames)
        #     print gameFeaturesDF
        #     sys.exit()
    vec = DictVectorizer()
    gameFeatures = vec.fit_transform(gameFeatDicts).toarray()
    gameFeaturesNames = vec.get_feature_names()
    gameFeaturesDF = pandas.DataFrame(gameFeatures, columns = gameFeaturesNames)
    gameFeaturesDF.index = gameFeaturesDF['steam_appid']
    gameFeaturesDF = gameFeaturesDF.drop(['steam_appid'], axis=1)
    print gameFeaturesDF.head().T
