def create_file():
    f1=open('summary.txt', 'w+')
    f2=open('cluster_data.txt',"r")
    f3=open('tweets_Collected.txt', 'r')
    f4=open('followers_list.txt',"r")
    f5=open('classify_data.txt',"r")
    
    line=f3.readline()
    num_of_followers = sum(1 for followers_line in open('followers_list.txt'))
    
    f1.write("Number of users collected: "+str(num_of_followers)+"\n\n")
    f1.write("Number of messages collected:: "+line+"\n\n")
    
    line=f2.readline()
    while(line !=''):
        f1.write(line)
        line=f2.readline()
        
    line=f5.readline()
    while(line !=''):
        f1.write(line)
        line=f5.readline()
        
    f1.close()
    f2.close()
    f3.close()
    f4.close()
    f5.close()
    
def main():
    create_file()
    
if __name__ == '__main__':
    main()
    