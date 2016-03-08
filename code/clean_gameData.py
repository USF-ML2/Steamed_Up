import json

with open('game_dataset_w_tags_flawed.txt', 'r') as inFile:
    with open('gameData.txt', 'w') as outFile:
        stored = list()
        
        for line in inFile:
            j = json.loads(line.strip())

            gameID = str(j['appid'])
            if gameID not in stored:
                outFile.write(line)
                stored.append(gameID)


print len(stored)              
