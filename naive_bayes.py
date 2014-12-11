# -*- coding: utf-8 -*-
"""
Filename    :
Input Data  : email files, split into spam and ham (non-spam)

build naive bayes spam classifier with nltk

references: bird, steven and et.al., nlp with python
perkins, jacob, python text processing with nltk 2.0 cookbook
"""

from pandas import *
import numpy as np
import os
import re
from nltk import NaiveBayesClassifier
import nltk.classify
from nltk.tokenize import wordpunct_tokenize
#from nltk.corpus import stopwords
from collections import defaultdict

# load email messages into lists
# -use spam and ham email to train classifier
data_path = os.path.abspath(os.path.join('.','data'))
spam_path = os.path.join(data_path,'spam')
spam2_path = os.path.join(data_path,'spam_2')
easyham_path = os.path.join(data_path, 'easy_ham')
easyham2_path = os.path.join(data_path, 'easy_ham_2')
hardham_path = os.path.join(data_path, 'hard_ham')
hardham2_path = os.path.join(data_path, 'hard_ham_2')


# load all email files into a directory, extracts content and return as a list

def get_msgdir(path):
    '''
    Read all messages from files in a directory into
    a list where each item is the text of a message.
    
    Simply gets a list of email files in a directory,
    and iteractions get_msg() over them.
    
    Returns a list of strings.
    '''
    filelist = os.listdir(path)
    filelist = filter(lambda x:x != 'cmds', filelist)
    all_msgs = [get_msg(os.path.join(path, f)) for f in filelist]
    return all_msgs
    
def get_msg(path):
    '''
    Read in the message portion of an email, given its file path.
    The message text begins after the first blank line
    above is header information.
    
    Returns a string.
    '''
    with open(path, 'rU') as con:
        msg = con.readlines()
        first_blank_index = msg.index('\n')
        msg = msg[(first_blank_index + 1):]
        return ''.join(msg)
        
# get lists containing training and testing message for each type
        
# training sets
train_spam_messages = get_msgdir(spam_path)
train_easyham_messages = get_msgdir(easyham_path)
# only keep the first 500 to balance with number of spams
train_easyhamn_messages = train_easyham_messages[:500]
train_hardham_messages = get_msgdir(hardham_path)

# test sets
test_spam_messages = get_msgdir(spam2_path)
test_easyham_messages = get_msgdir(easyham2_path)
test_hardham_messages = get_msgdir(hardham2_path)

# extract word features from emails

# parse and tokenize the emails func 
def get_msg_words(msg, stopwords = [], strip_html = False):
    ''' 
    Returns the set of unique words contained in an email message.
    Excludes any that are in an optionally-provided list.
    
    NLTK's 'wordpunct' tokenizer is used, and this will break contractions.
    For example, don't -> (don, ', t). Therefoe, it's advisable
    to supply a stopwords list that includes contraction parts, like 'don' and 't'
    '''
    # strip out weird 3D artefacts
    msg = re.sub('3D','', msg)
    
    # strip out html tags and attributes and html character codes, like &nbsp; and &lt;
    if strip_html:
        msg = re.sub('<(.|\n)*?>',' ',msg)
        msg = re.sub('&\w+;',' ', msg)
        
    # tokens with long underscore strings replace with a single one
    msg = re.sub('_+','_',msg)
    
    #remove '=' symbols before tokenizing, since these are sometimes occur within words, e.g. line-wrapper
    msg_words = set(nltk.wordpunct_tokenize(msg.replace('=\n','').lower()))
    
    # get rid of stopwords
    msg_words = msg_words.difference(stopwords)
    
    # get rid of punctuation tokens, numbers, and single letters
    msg_words = [w for w in msg_words if re.search('[a-zA-Z]', w) and len(w) > 1]
    
    return msg_words

# get stop words (NLTK)
stopwords_path = os.path.join(data_path, 'english')
sw = open(stopwords_path).read().splitlines()
sw.remove('')
print sw

#with open(stopwords_path, 'rU') as con:
#    sw = con.readlines(stopwords_path)
'''
sw = nltk.corpus.stopwords.words('english')
nltk.download()
sw.extend(['ll', 've'])

# stopwords exported from the 'tm' library in R
stopwords_path = os.path.join(data_path, 'english')
stopwords_path = os.path.join(data_path, 'r_stopwords.csv')
rsw = read_csv(stopwords_path)['x'].values.tolist()
rsw = read_csv(stopwords_path).values.tolist()
sw =rsw
'''
# make a features and label list
# naive bayes trains data on the form:
    #[(feature1, label1), (feature2, label2), ..., (featureN, labelN)]
    #where feature is a dictionary of features for email i and label i is label for email (spam/ham)

# features_from _messages: iterates through the messages creating this list and create features for each email
# this function is useful for extracting features from emails besides the set of words. combines features to email's label in a tuple and adds tuple to the list
def features_from_messages(messages, label, feature_extractor, **kwargs):
    ''' 
    Make a (feature, label) tuple for each message in a list of a certain,
    label of emails ('spam,'ham'') and return a list of these tuples
    
    Note every email should have the same label
    '''
    features_labels = [ ]
    for msg in messages:
        features = feature_extractor(msg, **kwargs)
        features_labels.append((features, label))
        return features_labels
    
# word-indicator: get an email words as a set, then creates a dictionary with entries {word: True}
# for each word in the set. 

def word_indicator(msg, **kwargs):
    '''
    Create a dictionary of entries {word: True} for every unique word in a message.
    Note **kwargs are options to the word-set creator, get_msg_words()
    '''
    features = defaultdict(list)
    msg_words = get_msg_words(msg, **kwargs)
    for w in msg_words:
        features[w] = True
    return features

# train and evaluate classifier
# note training set is a single list with train_spam and train_ham

def make_train_test_sets(feature_extractor, **kwargs):
    '''
    Make (feature, label) lists for each of the training and testing lists
    '''
    train_spam = features_from_messages(train_spam_messages, 'spam', feature_extractor, **kwargs)
    train_ham = features_from_messages(train_easyham_messages, 'ham', feature_extractor, **kwargs)
    train_set = train_spam + train_ham
    
    test_spam = features_from_messages(test_spam_messages, 'spam', feature_extractor, **kwargs)
    test_ham = features_from_messages(test_easyham_messages, 'ham', feature_extractor, **kwargs)
    test_hardham = features_from_messages(test_hardham_messages, 'ham', feature_extractor, **kwargs)
    
    return train_set, test_spam, test_ham, test_hardham
    
# function prints out results by NaiveBayesClassifier's show_most_informative_features method.
# which to show which features are most unique to one label or another. 
# e.g. 'viagra' shows up in 500 of spams only 2 in hams, then it is one of the most informative features with a spam:ham ratio 250:1
    
def check_classifier(feature_extractor, **kwargs):
    '''
    Train classifier on the training spam and ham, then check its accuracy on the test data,
    and show classifier's most informative features.
    '''
    
    # make train and test sets of (feature, label) data
    train_set, test_spam, test_ham, test_hardham = make_train_test_sets(feature_extractor, **kwargs)
    
    # train classifier on the trainning set
    classifier = NaiveBayesClassifier.train(train_set)
    
    # How accurate is the classifier on the test set?
    print ('Test Spam accuracy: {0:.2f}%'.format(100 * nltk.classify.accuracy(classifier, test_spam)))
    print ('Test Ham accuracy: {0:.2f}%'.format(100 * nltk.classify.accuracy(classifier, test_ham)))
    print ('Test Hard Ham accuracy: {0:.2f}%'.format(100 * nltk.classify.accuracy(classifier, test_hardham)))
    
    # show the top 20 informative features
    print classifier.show_most_informative_features(20)
    
    # Run classifier keeping all html information in the feature set. 
    # Accuracy at identifying spam and ham is very high.
    # but, lousy job at hard spam may due to train set too much rely on html_tags to identify spam

check_classifier(word_indicator, stopwords = sw)

# without html-tags, loose a bit on accuracy in identifying spam but improve with hard ham
check_classifier(word_indicator, stopwords = sw, strip_html = True)
    
    
  
    


