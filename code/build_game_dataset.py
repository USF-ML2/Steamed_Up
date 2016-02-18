import requests
import json
import psycopg2, psycopg2.extras
import urllib2
import sys
from PIL import Image
import os
import feedparser
import BeautifulSoup as bs
import re

# Scrapes websites as well as Steam's API for game data in json format

STEAM_API_KEY = '38A258CB0B4EC6B310BF3E6B34A72604'
STEAM_ID = '76561198043891906'

incomplete = list()

def get_gameSchema(appID):
    """
    :param appid: game ID
    :return: json object of gameSchema as returned by ISteamUserStats API
    """
    
    url = 'http://api.steampowered.com/ISteamUserStats/GetSchemaForGame/v2/\
?key='+STEAM_API_KEY+'&steamid='+STEAM_ID+'&appid='+str(appID)
    request = urllib2.Request(url)

    try:
        page = urllib2.urlopen(request)
        result = page.read()
    except urllib2.URLError, e:
        incomplete.append((appID, "schema"))
        return {}

    return json.loads(result)

def get_gameDetails(appID):
    """
    :param appid: game ID
    :return: json object of gameDetails as returned by url
    """
    
    url = 'http://store.steampowered.com/api/appdetails?appids='+str(appID)
    request = urllib2.Request(url)

    try:
        page = urllib2.urlopen(request)
        result = page.read()
    except urllib2.URLError, e:
        incomplete.append((appID, "details"))
        return {}

    return json.loads(result)

def get_tags(appid):
    """
    :param appid: game ID
    :return: list of user identified tags for game
    """
    request = urllib2.Request("http://store.steampowered.com/app/"+str(appid))

    game_tags = list()
    
    try:
        page = urllib2.urlopen(request)
    except urllib2.URLError, e:
        if hasattr(e, 'reason'):
            print 'Failed to reach url'
            print request
            print 'Reason: ', e.reason
            sys.exit()
        elif hasattr(e, 'code'):
            if e.code == 404:
                print 'Error: ', e.code
                sys.exit()

    page = page.read()

    soup = bs.BeautifulSoup(page)

    tags = soup.find('div', attrs = {'class': 'glance_tags popular_tags'})
    if tags is not None:
        taglist = tags.findAll('a')
        for t in taglist:
            game_tags.append(t.text)

    return game_tags


if __name__ == "__main__":
    data = list()

    # For cleanup
    prev_complete = list()
    with open('complete.txt', 'r') as inFile2:
        for line in inFile2:
            prev_complete.append(line.strip())
    
    # appID_dataset.json is a json object with just the
    # names and IDs of every game in the steam library
    with open('appIDs.json', 'r') as inFile:
        j = json.load(inFile)

        apps = j['applist']['apps']['app']
        for item in apps:
            appID = item['appid']
            name = item['name']

            print 'checking', appID
            try:
                if appID not in prev_complete: # For cleanup
                    schema = get_gameSchema(appID)
                    details = get_gameDetails(appID)
                    tags = get_tags(appID)

                    if len(schema) != 0 or len(details) != 0:
                        success = details[str(appID)]['success'] # true/false
                        #print type(success)
                        
                        if success == True:
                            toAdd = {"appid": appID, "name": name, "schema" : schema, \
                                     "details" : details, "tags" : tags}

                            # writing to final app dataset file
                            with open('game_dataset_w_tags.txt', 'a') as app_data:
                                print "Writing for ID:", appID
                                json.dump(toAdd, app_data)
                                app_data.write('\n')
                                print 'Writing for', appID
                                
                        else:
                            prev_complete.append(appID)
            except:
                # for cleaning
                print "Complete AppIDs:"
                with open('complete.txt', 'w') as outFile:
                    for i in prev_complete:
                        outFile.write(str(i)+"\n")



