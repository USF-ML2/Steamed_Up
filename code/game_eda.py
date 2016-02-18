__author__ = "Paul Thompson, Brynne Lycette"
import sys, json

### Genre Counts and Proportions
def genre_cat_info(filepath):
    genre_eda = dict()
    
    with open(filepath) as f:
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
            
        genre_eda["Genre Success Count"] = genreSuccessCount
        genre_eda["Game Count"] = gameCount
        genre_eda["Genre Counts"] = genreCount
        genre_eda["Genre Proportions"] = genreProport
        genre_eda["Number of Games with Category"] = categCount
        genre_eda["Percent of Games with Category"] = categProport

        return genre_eda

def count_achievements(filepath):
    count = 0
    game_count = 0
    
    with open(filepath, 'r') as inFile:
        game_jsons = inFile.readlines()

    for game in game_jsons:
        game_json = json.loads(game.strip())
        
        details = game_json['details']
        schema = game_json['schema']['game']
        gameID = str(game_json['appid'])
        name = game_json['name']

        if details[gameID]['success'] != False: #and len(schema) != 0:
            game_count += 1
            
            data = details[gameID]['data']
            keys = data.keys()


            if 'achievements' in keys:
                count += (0 if data['achievements']['total'] == 0 else 1)

    return game_count, count
            

if __name__ == '__main__':

    game_path = 'game_dataset.txt'
    user_path = 'userData.txt'
    appid_path = 'appIDs.json'

    #genre_eda = genre_cat_info(game_path)
    print count_achievements(game_path)
