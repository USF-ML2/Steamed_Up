import sys, json

### Genre Counts and Proportions
with open("/Users/paulthompson/Documents/MSAN_Files/Spr1_AdvML/Final_Proj/game_dataset.txt") as f:
    genreCount = {}; genreProport = {}; gameCount = 0;  genreSuccessCount = 0
    categCount = {}; categProport = {}; gameDetailsCount = 0
    for line in f:
        record = json.loads(line)
        # print record.keys()
        if record['details'][record['details'].keys()[0]]['success']:
            try:
                gameDetailsCount += 1
                genre = record['details'][record['details'].keys()[0]]['data']['genres'][0]['description']
                categories = record['details'][record['details'].keys()[0]]['data']['categories']
                if genre in genreCount.keys():
                    genreSuccessCount += 1
                    genreCount[genre] += 1
                else:
                    genreCount[genre] = 1
                for category in categories:
                    if category['description'] in categCount.keys():
                        categCount[category['description']] += 1
                    else:
                        categCount[category['description']] = 1
            except:
                continue
        gameCount += 1
        # if gameCount == 100:
        #     sys.exit()
    for genreKey in genreCount.keys():
        genreProport[genreKey] = round(float(genreCount[genreKey]) / float(genreSuccessCount), 6)
    for categKey in categCount.keys():
        categProport[categKey] = round(float(categCount[categKey]) / float(gameDetailsCount), 6)
    print "Genre Success Count", genreSuccessCount, "Game Count:", gameCount
    print "Genre Counts:", genreCount
    print "Genre Proportions:", genreProport
    print "Number of Games with Category:", categCount
    print "Percent of Games with Category:", categProport
    keycount = 0
    for key in categCount.keys():
        keycount += 1
    print keycount


### Game Details Example
# with open("/Users/paulthompson/Documents/MSAN_Files/Spr1_AdvML/Final_Proj/game_dataset.txt") as f:
#     gameCount = 0
#     for line in f:
#         record = json.loads(line)
#         if record['details'][record['details'].keys()[0]]['success']:
#             try:
#                 if gameCount == 9:
#                     for i in range(len(record['details'][record['details'].keys()[0]]['data'].keys())):
#                         print record['details'][record['details'].keys()[0]]['data'].keys()[i], \
#                             record['details'][record['details'].keys()[0]]['data'].values()[i]
#             except:
#                 continue
#         gameCount += 1
#         if gameCount == 10:
#             sys.exit()

# ## User Information
# with open("/Users/paulthompson/Documents/MSAN_Files/Spr1_AdvML/Final_Proj/userData.txt") as f:
#     linecount = 0
#     for line in f:
#         record = json.loads(line)
#         if record['ownedGames']['response'].keys():
#             # print record['ownedGames']['response']
#             playedCount = 0
#             notPlayedCount = 0
#             for game in record['ownedGames']['response']['games']:
#                 if game['playtime_forever'] == 0:
#                     notPlayedCount += 1
#                 if game['playtime_forever'] <> 0:
#                     playedCount += 1
#             print "Owned but not played:", notPlayedCount, "Played:", playedCount
#             linecount += 1
#         if linecount == 10:
#             sys.exit()

#### Game App Ids
# with open("/Users/paulthompson/Documents/MSAN_Files/Spr1_AdvML/Final_Proj/appIDs.json") as f:
#     # 21,399 "games"
#     records = json.load(f)
#     for game in records['applist']['apps']['app']:
#         print game['name']
#         count += 1
#     print count
