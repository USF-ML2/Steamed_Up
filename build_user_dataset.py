import requests
import json
import urllib2
import sys
import urllib
import psycopg2, psycopg2.extras
import feedparser
import BeautifulSoup as bs
import re

STEAM_API_KEY = '3751BF6704123565C3DB0BE1A8E03057'
STEAM_ID = '76561198013144096'

def get_ownedGames(user_id):
    """

    :return:
        if user's profile prviate, returns emtpy json object
    """
    
    url = 'http://api.steampowered.com/IPlayerService/GetOwnedGames/v0001/\
?key='+STEAM_API_KEY+'&steamid='+user_id+'&include_appinfo=1&format=json'
    request = urllib2.Request(url)

    try:
        page = urllib2.urlopen(request)
        result = page.read()
    except urllib2.URLError, e:
        if hasattr(e, 'reason'):
            if e.reason == 'Unauthorized':
                return {}
            else:
                print 'Failed to reach url'
                print url
                print 'Reason: ', e.reason
                sys.exit()
        elif hasattr(e, 'code'):
            if e.code == 404:
                print 'Error: ', e.code
                sys.exit()

    return json.loads(result)

if __name__ == '__main__':
    # Collecting all reachable users from GROUP_URL net
    '''
    seedlist_raw = get_memberIDs(GROUP_URL)
    seedlist = seedlist_raw[-10:]
    complete = build_friendNet(seedlist)
    '''
    inserted = list()
    with open('inserted.txt', 'r') as inFile:
        for line in inFile:
            inserted.append(line.strip())
    
##    with open('full_user_data.txt', 'r') as existing:
##        for line in existing:
##            data = json.loads(line.strip())
##            steamid = data['user']
##            inserted.append(steamid)
        
    with open('userIDs.txt', 'r') as inFile:
        with open('inserted.txt', 'a') as outFile:
            # Collect ownedGames json for each user
            for line in inFile:
                user = line.strip()
                if len(user) != 0:
                    if user not in inserted:
                        print "Getting owned games for",user
                        ownedGames = get_ownedGames(user)

                        with open('userData.txt', 'a') as full_data:
                            #print "writing to file."
                            if len(ownedGames) != 0:
                                inserted.append(user)
                            json.dump({"user":user, "ownedGames":ownedGames}, full_data)
                            full_data.write('\n')
                            outFile.write(user+'\n')
                        

    print len(inserted)
