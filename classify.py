from collections import defaultdict
from itertools import combinations
import glob
import numpy as np
import os
import re
from scipy.sparse import csr_matrix
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression
import string

def read_data(path):
    
    fnames = sorted([f for f in glob.glob(os.path.join(path, 'pos', '*.txt'))])
    data = [(1, open(f).readlines()[0]) for f in sorted(fnames)]
    fnames = sorted([f for f in glob.glob(os.path.join(path, 'neg', '*.txt'))])
    data += [(0, open(f).readlines()[0]) for f in sorted(fnames)]
    data = sorted(data, key=lambda x: x[1])
    return np.array([d[1] for d in data]), np.array([d[0] for d in data])

def tokenize(doc, keep_internal_punct=False):
   
    list1=[]                # Create token list
    lower_case_words=doc.lower().strip().split() # Convert it to lower case. 
    exclude_strings=string.punctuation           # Get the characters from result.
    doc = re.sub('http\S+', '', doc)
    if keep_internal_punct== False:              # if punctuation is False.
        return np.array(re.sub('\W+', ' ', doc.lower()).split())   #return tokens without punctuations. 
    else:                                        # if punctuation is True.   
        for word_in_doc in lower_case_words:
            word_in_doc=word_in_doc.lstrip(exclude_strings)  # Trim unwanted characters from left.    
            word_in_doc=word_in_doc.rstrip(exclude_strings)  # Trim unwanted characters from right.
            list1.append(word_in_doc)                        # add that token to list.  
        return np.array(list1)                               # Convert list to np array.

    pass
    
def token_features(tokens, feats):    
    
    for token in tokens:
        feats['token='+token]=feats['token='+token]+1   # return how many time word appears in document.
    pass

def featurize(tokens, feature_fns):
   
    feats=defaultdict(lambda: 0)
    for method in feature_fns:
        method(tokens,feats)
    return sorted(feats.items())
    pass

def vectorize(tokens_list, feature_fns, min_freq, vocab=None):
    
    row_counter=0
    feature_dict=defaultdict(lambda: 0)
    final_dict=defaultdict(lambda: 0)
    row=[]
    column=[]
    data = []
    temp_dict=defaultdict(lambda: 0)
    column_dictionary=defaultdict(lambda: 0)
    for list1 in tokens_list:
        feature_dict[row_counter]=featurize(list1,feature_fns)
        for x in feature_dict[row_counter]:
            temp_dict[x[0]]=temp_dict[x[0]]+1
        row_counter+=1
        
    for key,val in temp_dict.items():
        if(val>=min_freq):
            final_dict[key]=val
        
    final_dict_list=sorted(final_dict.items(), key=lambda x:(x[0]))
    counter=0
    len1=len(final_dict_list)
    
    while counter<len1:
        column_dictionary[final_dict_list[counter][0]]=counter
        counter+=1
    
    def add_data(column_number,word_name,row_number): 
        column.append(column_dictionary[x[0]])
        data.append(x[1])
        row.append(key)
     
    if(vocab==None):
        for key,val in feature_dict.items():
            for x in val:
                if(temp_dict[x[0]]>=min_freq):
                    add_data(column_dictionary[x[0]],x[1],key)
    else:
        column_dictionary=vocab
        for key,val in feature_dict.items():
            for x in val:
                if(x[0] in column_dictionary ):
                    add_data(column_dictionary[x[0]],x[1],key)

    return (csr_matrix((data, (row,column)),shape=(len(tokens_list), len(column_dictionary)), dtype='int64'),column_dictionary)
    pass

def accuracy_score(truth, predicted):
    
    return len(np.where(truth==predicted)[0]) / len(truth)

def cross_validation_accuracy(clf, X, labels, k):
    
    label_length=len(labels)
    number_of_flods = KFold(label_length, k)
    accuracies_of_data=[]
    for t_ind, p_ind in number_of_flods:
        clf.fit(X[t_ind], labels[t_ind])
        predictions = clf.predict(X[p_ind])
        accuracies_of_data.append(accuracy_score(labels[p_ind], predictions))
    return np.mean(accuracies_of_data)
    pass

def eval_all_combinations(docs, labels, punct_vals,
                          feature_fns, min_freqs):
    
    feature_fns_list=[]
    for val in range(1, len(feature_fns)+1):
        for item in combinations(feature_fns, val):
            feature_fns_list.append(item)
    dictionary_list=[]
    token_list_true=[tokenize(d,True) for d in docs]
    token_list_false=[tokenize(d) for d in docs]
    
    for freq in min_freqs:
        for feature in feature_fns_list:
            for punct in punct_vals:
                data_settings={}
                data_settings['punct']=punct
                data_settings['features']=feature
                data_settings['min_freq']=freq
                if (punct==True):
                    CSR,vocab=vectorize(token_list_true, feature, freq,None)
                else:
                    CSR,vocab=vectorize(token_list_false, feature, freq,None)                    
                data_settings['accuracy']=cross_validation_accuracy(LogisticRegression(), CSR, labels, 5)
                dictionary_list.append(data_settings)
    dictionary_list=sorted(dictionary_list, key=lambda x:x['accuracy'],reverse=True)
    
    return dictionary_list
    pass

def fit_best_classifier(docs, labels, best_result):
   
    clf = LogisticRegression()
    X,vocab=vectorize([tokenize(d,best_result["punct"]) for d in docs], best_result['features'], best_result['min_freq'])
    clf.fit(X, labels) 
    return clf,vocab
    pass


def parse_test_data(best_result, vocab):
   
    testing_docs,testing_labels=read_data(os.path.join('data', 'Test'))
    X_test,vocab=vectorize([tokenize(d,best_result["punct"]) for d in testing_docs], best_result['features'], best_result['min_freq'],vocab=vocab)
    return testing_docs,testing_labels,X_test
    pass

def main():
    f3=open('classify_data.txt', 'w+')
    feature_fns = [token_features]
    #print(os.path.join('data', 'Train'))
    docs, labels = read_data(os.path.join('data', 'Train'))
    
    results = eval_all_combinations(docs, labels,
                                    [True,False],
                                    feature_fns,
                                    [2,5,10])
    best_result = results[0]
    worst_result = results[-1]
    clf, vocab = fit_best_classifier(docs, labels, results[0])
    test_docs, test_labels, X_test = parse_test_data(best_result, vocab)
    #print(test_docs[0])
    predictions = clf.predict(X_test)
    probablities_label = clf.predict_proba(X_test)

    index_counter=0
    
    pos_class_counter=0
    neg_class_counter=0
    positive_max_probability=0
    positive_max_probability_counter=0
    negative_max_probability=0
    negative_max_probability_counter=0

    for index in range(len(probablities_label)):
        val=probablities_label[index]
        if np.all(val[1] > positive_max_probability):
            
            positive_max_probability_counter=index
            positive_max_probability=probablities_label[index]
        elif np.all(val[0] > negative_max_probability):
            negative_max_probability_counter=index
            negative_max_probability=probablities_label[index]
    #print(len(predictions))
    for index in range(len(predictions)):
        if predictions[index] == 1:
            pos_class_counter+=1
        elif predictions[index] == 0:
            neg_class_counter+=1
    
    f3.write("Number of instances per class found: "+"\n")
    f3.write("Number of instances of Positive class found: "+str(pos_class_counter)+"\n")
    f3.write("Number of instances of Negative class found: "+str(neg_class_counter)+"\n\n")

    f3.write("One example from each class: "+"\n")        
    
    #print(test_docs[positive_max_probability_counter])
    f3.write("Example of Positive Class: "+"\n")
    f3.write(test_docs[positive_max_probability_counter]+"\n")
    
    #print("Negative Document")        
    #print(test_docs[negative_max_probability_counter])
    f3.write("Example of Negative Class: "+"\n")
    f3.write(test_docs[negative_max_probability_counter]+"\n")
    
    f3.close()
   
if __name__ == '__main__':
    main()