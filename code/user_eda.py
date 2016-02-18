__author__ = "Brynne Lycette"

import sys, json

"""
avg game count
avg playtime
avg playtime by genre?

"""

def avg_game_count(filepath):
    """
    :param filepath: path to userData
    :return:

    Prints Total Users, Total Game Count, Avg Game per User
    """
    total_game_count = 0
    total_users = 0
    
    with open(filepath, 'r') as inFile:
        for line in inFile:
            info = json.loads(line.strip())

            # Check if public profile
            public = len(info['ownedGames']['response']) is not 0

            if public:
                total_users += 1
                total_game_count += info['ownedGames']['response']['game_count']

    print "Total Users:", total_users
    print "Total Game Count:", total_game_count
    print "Avg Game per User:", total_game_count/float(total_users)


def avg_playtime(filepath):
    """
    :param filepath: path to userData
    :return:

    Prints Total Users, Total Playtime, Avg playtime per user (hours : mins)
    """
    total_playtime = 0
    total_users = 0

    with open(filepath, 'r') as inFile:
        for line in inFile:
            info = json.loads(line.strip())

            # Check if public profile
            public = len(info['ownedGames']['response']) is not 0

            if public:
                total_users += 1
                count = info['ownedGames']['response']['game_count']
                if count != 0:
                    games = info['ownedGames']['response']['games']
                    for g in games:
                        total_playtime += g['playtime_forever']

    avg = total_playtime/float(total_users)
    avg_hours = int(avg)/60
    avg_mins = int(avg)%60

    print "Total Users:", total_users
    print "Total Playtime:", total_playtime
    print "Avg Playtime per User:", avg_hours, "hours and", avg_mins, "mins"


def get_game_genres(game_data):
    """
    :param game_data: path to game data
    :return: dictionary {appID : list of genres}
    """
    
    game_dict = dict()
    with open(game_data, 'r') as gameFile:
        for line in gameFile:
            game = json.loads(line)
            appid = str(game['appid'])

            # Check if successful
            if game['details'][appid]['success']:
                data = game['details'][appid]['data']
                keys = data.keys()

                genres_list = list()

                if 'genres' in keys:
                    genre_info = data['genres']
                    for i in genre_info:
                        genres_list.append(i['description'])

                game_dict[appid] = genres_list

    return game_dict

def avg_playtime_genre(user_data, game_data):
    """
    :params user_data, game_data: paths to user data and game data respect.
    :return: dictionary {genre : average playtime}
    """

    ### COLLECTING GENRES ###
    game_genres = get_game_genres(game_data)
    game_keys = game_genres.keys()

    genre_playtimes = dict()
    total_users = dict()

    with open(user_data, 'r') as inFile:
        for line in inFile:
            info = json.loads(line.strip())

            # Check if public profile
            public = len(info['ownedGames']['response']) is not 0

            if public:
                count = info['ownedGames']['response']['game_count']
                if count != 0:
                    games = info['ownedGames']['response']['games']
                    for g in games:
                        playtime = g['playtime_forever']
                        appid = str(g['appid'])
                        
                        if appid in game_keys:
                            genres = game_genres[appid]
                        else:
                            genres = []
                        
                        for genre in genres:
                            # Users with game of genre
                            if genre not in total_users.keys():
                                total_users[genre] = 1
                            elif genre in total_users.keys():
                                total_users[genre] += 1

                            # Playtime for game of genre
                            if genre not in genre_playtimes.keys():
                                genre_playtimes[genre] = playtime
                            elif genre in genre_playtimes.keys():
                                genre_playtimes[genre] += playtime

    ### AVERAGING ###
    avg_genre_playtimes = dict()
    for k, v in genre_playtimes.iteritems():
        avg_genre_playtimes[k] = v/float(total_users[k])

    return avg_genre_playtimes
                    

if __name__ == '__main__':

    avg_game_count('userData.txt')
    avg_playtime('userData.txt')
    #gd = avg_playtime_genre('userData.txt', 'game_dataset.txt')
