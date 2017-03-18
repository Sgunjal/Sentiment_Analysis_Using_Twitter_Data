from TwitterAPI import TwitterAPI
from collections import defaultdict
import sys
import time
import os
import glob

consumer_key = 'EMBcTZDeaiQ6NoVTv51a3CEZm'
consumer_secret = '8lM7Vz7kIq0XtniLU715T14GsBx6411I5xeNbzdwDqFPylhPgE'
access_token = '64342841-f3LAynjrVCwtWWpF7jpsrhgwa8U2BYBNb9SZg7s4P'
access_token_secret = 'uFYVdx0pHXO96D4vz2Qs2CpNM1bwXiZdUkws2pmZKKqTl'

def get_twitter():
        
    return TwitterAPI(consumer_key, consumer_secret, access_token, access_token_secret)

def robust_request(twitter, resource, params, max_tries=5):
    for i in range(max_tries):
        request = twitter.request(resource, params)
        if request.status_code == 200:
            return request
        else:
            print('Got error %s \nsleeping for 15 minutes.' % request.text)
            sys.stderr.flush()
            time.sleep(61 * 15)
            
def get_friends_of_user(twitter, screen_name):
    resource='followers/ids'
    params={'screen_name':screen_name ,'count':2000}
    responseofRequest=robust_request(twitter, resource, params, max_tries=5)
    friend_ids=[r for r in responseofRequest]
    return sorted(friend_ids)  
    pass

def get_tweets(resource, twitter, params):
    responseofRequest=robust_request(twitter, resource, params, max_tries=5)
    friend_ids=[r for r in responseofRequest]
    return friend_ids  
    pass

def write_followers_to_file(friendlist,twitter,file_name):

    for x in friendlist:
        f=open(file_name, 'a')
        all_friends=get_friends_of_user(twitter, x['user']['screen_name'])
        f.write(x['user']['screen_name']+" $|$ " + str(all_friends)+"\n")
        f.close()


def write_tweets_to_file(unique_users,friendlist,twitter,file_name,id_list,counter):
    
    path=os.path.abspath("data/Test")
    tweets_data=defaultdict(lambda: 0)
    data=defaultdict(lambda: 0)
    
    for x in friendlist:
        fx=open(file_name, 'a')
        id_list.append(x['id'])
        unique_users.add(x['id'])
        f=open(path+'/pos/'+str(counter)+'.txt', 'w+')
        str1=x['text'].replace('\n',' ')
        str1=str1.strip("?")
        f.write(str1.encode(sys.stdout.encoding, errors='replace').decode("utf-8", "ignore")+'\n')
        fx.write(str1.encode(sys.stdout.encoding, errors='replace').decode("utf-8", "ignore")+'\n')
        f.close()
        counter=counter+1
        fx.close()
        
    return id_list,counter,unique_users
    
def main():
    path=os.path.abspath("data/Test")
    fnames = sorted([f for f in glob.glob(os.path.join(path, 'pos', '*.txt'))])
    for f in fnames:
        os.remove(f)
    twitter = get_twitter()
    #f3=open('summarize.txt', 'w+')
    no_of_iterations=5
    fn=open('tweets_Collected.txt', 'w+')
    fn.close()
    f2=open('user_tweets.txt', 'w+')
    f2.close()
    f1=open('followers_list.txt', 'w+')
    f1.close()
    no_of_tweets=100
    max_id=0
    id_list=[]
    friendlist = get_tweets('search/tweets',twitter,{'count':100,'q':'#DeMonetisation','lang':'en'})
    for x in friendlist:
        id_list.append(x['id'])
            
    counter=1
    no_of_iterations=20
    unique_users=set()
    for i in range(0,no_of_iterations):
        friendlist = get_tweets('search/tweets',twitter,{'count':no_of_tweets,'q':'#DeMonetisation','lang':'en','max_id':min(id_list)-1})
        id_list,counter,unique_users=write_tweets_to_file(unique_users,friendlist,twitter,'user_tweets.txt',id_list,counter)
        fn=open('tweets_Collected.txt', 'w+')
        fn.write(str((i+1)*no_of_tweets))
        fn.close()
        write_followers_to_file(friendlist,twitter,'followers_list.txt')

if __name__ == '__main__':
    main()
    
    