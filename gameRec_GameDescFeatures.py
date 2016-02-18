import sys, json, pandas, numpy as np
from sklearn.feature_extraction import DictVectorizer
from time import time

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

n_features = 1000
n_topics = 50
n_top_words = 20
game_path = "/Users/paulthompson/Documents/MSAN_Files/Spr1_AdvML/Final_Proj/game_dataset.txt"

def create_game_profile_df(game_data_path):
    with open(game_data_path) as f:
        gameDescriptions = []
        gameNames = []
        gameCount = 0
        for line in f:
            record = json.loads(line)
            # print record.keys()
            if record['details'][record['details'].keys()[0]]['success']:
                try:
                    gameNames.append(record['details'][record['details'].keys()[0]]['data']['name'])
                    gameDescriptions.append(record['details'][record['details'].keys()[0]]['data']['detailed_description'])
                except:
                    continue
                gameCount += 1

    return gameDescriptions, gameNames

def produceLDATopics():
    data_samples,gameNames = create_game_profile_df(game_path)
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=n_features, stop_words='english')
    tf = tf_vectorizer.fit_transform(data_samples)
    lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=5,
                                    learning_method='online', learning_offset=50.,
                                    random_state=0)
    topics = lda.fit_transform(tf)
    # for i in range(50):
    #     gameTopics = []
    #     for j in range(len(topics[0])):
    #         if topics[i,j] > 1.0/float(n_topics):
    #             gameTopics.append(j)
    #     print gameNames[i], gameTopics
    topicsByGame = pandas.DataFrame(topics)
    topicsByGame.index = gameNames
    print topicsByGame

    tf_feature_names = tf_vectorizer.get_feature_names()
    for topic_idx, topic in enumerate(lda.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([tf_feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))

produceLDATopics()