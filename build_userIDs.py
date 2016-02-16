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
#GROUP_URL = 'http://steamcommunity.com/groups/hellsgamers'

def web_iterate(base_url, pages, added):
    url_pages = list()
    with open(added, 'a') as outFile:
        for i in xrange(pages):
            url_pages.append(base_url+"?p="+str(i))
            outFile.write(base_url+"?p="+str(i)+'\n')

    return url_pages

def get_steamID(link):
    """
    :param link: url to steam profile
    :return: steamID
    """
    request = urllib2.Request(link+"?xml=1")

    try:
        page = urllib2.urlopen(request)
    except urllib2.URLError, e:
        if hasattr(e, 'reason'):
            print 'Failed to reach url'
            print link
            print 'Reason: ', e.reason
            sys.exit()
        elif hasattr(e, 'code'):
            if e.code == 404:
                print 'Error: ', e.code
                sys.exit()

    page = page.read()

    reMatch = re.search(r'<steamID64>\d+', page)
    if reMatch != None:
        steamID_raw = reMatch.group()
        steamID = steamID_raw[11:]

        return steamID
    else:
        return ''

def get_group_urls():
    """
    :param group_list_url:
    :return: list of group urls
    """
    group_urls = list()

    request = urllib2.Request("http://steamcommunity.com/actions/GroupList")
    
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

    # creating soup
    soup = bs.BeautifulSoup(page)

    memberRows = soup.findAll('div', attrs = {'class': 'memberRow'})
    for row in memberRows:
        linkStnd = str(row.findAll('a', attrs={'class':'linkStandard'}))
        link = re.search(r'href="([^"]|\\")*"', linkStnd).group()
        group_urls.append(link.split('"')[1])
    
    return group_urls

def get_pages(base_url, added):
    request = urllib2.Request(base_url)
    
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

    # creating soup
    soup = bs.BeautifulSoup(page)

    last_page_link = soup.findAll('a', attrs = {'class': 'pagelink'})[-1]
    last_page = str(last_page_link['href']).split('?p=')[1]

    # list of all member listing urls
    return web_iterate(base_url, int(last_page), added)
    

def get_memberIDs(group_url):
    """
    :param group_url: url of popular steam community group to seed from
    :return: list of members' steamIDs within this group
    """    
    request = urllib2.Request(group_url)

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

    #creating soup
    soup = bs.BeautifulSoup(page)

    memberIDs = list()
    for link in soup.findAll('a'):
        info = link.get('href')
        needed = re.match(r'http://steamcommunity.com/id/.*', info)
        have = re.match(r'http://steamcommunity.com/profiles/.*', info)
        
        if needed != None: #if member has personalized url
            memberLink = needed.group() #get actual link

            #Navigate to link to grab steam id
            memberID = get_steamID(memberLink)
            
            memberIDs.append(memberID)
        
        if have != None: #if member has general url
            memberID_raw = have.group().split("/")
            memberID = memberID_raw[4].encode('utf-8')

            memberIDs.append(memberID)
          
    return memberIDs

def build_userIDs(userIDfile, added):

    addedPages = list()
    with open(added, 'r') as inFile:
        for line in inFile:
            addedPages.append(line.strip())

    with open(userIDfile, 'a') as outFile:
        group_urls = get_group_urls()
        for url in group_urls:
            pages = get_pages(url, added) #list of urls
            for p in pages:
                if p not in addedPages:
                    members = get_memberIDs(p)
                    for m in members:
                        print 'Writing',m,'to file'
                        outFile.write(m+'\n')



##def get_friendsList(user_id):
##    """
##    :param user_id: steam id of user to get friends list of
##    :return: json object of friends list returned by steam API
##        If user profile private, returns empty json object
##    """
##    
##    url = 'http://api.steampowered.com/ISteamUser/GetFriendList/v0001/\
##?key='+STEAM_API_KEY+'&steamid='+user_id+'&format=json'
##    request = urllib2.Request(url)
##
##    try:
##        page = urllib2.urlopen(request)
##        result = page.read()
##    except urllib2.URLError, e:
##        if hasattr(e, 'reason'):
##            if e.reason == 'Unauthorized':
##                return {}
##            else:
##                print 'Failed to reach url'
##                print 'Reason: ', e.reason
##                sys.exit()
##        elif hasattr(e, 'code'):
##            if e.code == 404:
##                print 'Error: ', e.code
##                sys.exit()
##
##    return json.loads(result)
##
##
##def extract_friends(user_id):
##    """
##    :param user_id: steam id of user getting friends from
##    :return: list of user's friends' steam ids
##        If user's profile private, return empty list
##    """
##
##    user_friends = get_friendsList(user_id)
##    if len(user_friends) == 0:
##        return []
##    else:
##        friendList = list()
##        friends_json = user_friends['friendslist']['friends']
##
##        for item in friends_json:
##            friendList.append(item['steamid'].encode('utf-8'))
##
##        return friendList
##
##
##def build_friendNet_helper(seed_friends, complete):
##
##    with open('steamIDs.txt', 'a') as inFile:
##        # iterate through friend network
##        new = list()
##        for friend in seed_friends:
##            newList = extract_friends(friend)
##            for newFriend in newList:
##                if newFriend not in complete:
##                    print "Writing ID:",newFriend
##                    inFile.write(newFriend+"\n")
##                    new.append(newFriend)
##                    complete.append(newFriend)
##
##    if len(new) == 0:
##        return complete
##    else: #dig into 
##        complete = complete + build_friendNet_helper(new, complete)        
##            
##
##def build_friendNet(seedlist):
##    """
##    :param seed: seed list of steam ids to build net off of
##    :return: full list of all user steam ids reached
##    """
##    
##    # start just with seed users IDs
##    complete = seedlist
##
##    '''
##    # gather seed user's friends
##    seed_friends = extract_friends(seed)
##    # add initial group to complete list, no need to check for dups
##    complete = complete + seed_friends
##    '''
##    # begin recursion
##    toAdd = build_friendNet_helper(seedlist, complete)
##    complete = complete + toAdd
##
##    return complete
##      


if __name__ == "__main__":
    
    build_userIDs('userIDs.txt', 'addedPages.txt')


      
