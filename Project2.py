#!/usr/bin/env python
# coding: utf-8

# In[1]:

#Name: Sanket Manik Salunke Id: 1001764897
import os  # to access the current working directory
import math #using for sum calculation
import random #using to generate random number in preprocessing
from sklearn.metrics import confusion_matrix   # Just to print the confusion matrix
import copy # copy the data 
import re # to implement regex in preprocessing- eliminate unwanted data


# In[2]:


data_path = "20_newsgroups"
#get current working directory
data_path = os.getcwd()
#getting exact path
data_path = data_path + "/20_newsgroups/"


# In[3]:


#grabbing all the folder names from the directory
Folder_name = os.listdir(data_path)
len(Folder_name)


# In[4]:


data_train = 500


# In[5]:


filename ={}


# In[6]:


#List of stopwords to remove from the text to clearn the data
list_stopwords = ['a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', "aren't", 'as', 'at',
 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 
 'can', "can't", 'cannot', 'could', "couldn't", 'did', "didn't", 'do', 'does', "doesn't", 'doing', "don't", 'down', 'during',
 'each', 'few', 'for', 'from', 'further', 
 'had', "hadn't", 'has', "hasn't", 'have', "haven't", 'having', 'he', "he'd", "he'll", "he's", 'her', 'here', "here's",
 'hers', 'herself', 'him', 'himself', 'his', 'how', "how's",
 'i', "i'd", "i'll", "i'm", "i've", 'if', 'in', 'into', 'is', "isn't", 'it', "it's", 'its', 'itself',
 "let's", 'me', 'more', 'most', "mustn't", 'my', 'myself',
 'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought', 'our', 'ours' 'ourselves', 'out', 'over', 'own',
 'same', "shan't", 'she', "she'd", "she'll", "she's", 'should', "shouldn't", 'so', 'some', 'such', 
 'than', 'that',"that's", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', "there's", 'these', 'they', "they'd", 
 "they'll", "they're", "they've", 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 
 'was', "wasn't", 'we', "we'd", "we'll", "we're", "we've", 'were', "weren't", 'what', "what's", 'when', "when's", 'where',
 "where's", 'which', 'while', 'who', "who's", 'whom', 'why', "why's",'will', 'with', "won't", 'would', "wouldn't", 
 'you', "you'd", "you'll", "you're", "you've", 'your', 'yours', 'yourself', 'yourselves', 
 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'hundred', 'thousand', '1st', '2nd', '3rd',
 '4th', '5th', '6th', '7th', '8th', '9th', '10th',"'",'!','/','\\','=',',',':', '<','>','?','.','"',')','(','|','-','#','*','+','$']


# In[7]:


stop_words_1 = len(list_stopwords)
print("Total number of stop words",stop_words_1)


# In[8]:


def remove_stopwords(data):
    #https://www.geeksforgeeks.org/python-string-replace/
    data = data.replace('\n', ' ')  
    data = data.replace('\t', ' ')  
    #https://www.programiz.com/python-programming/methods/string/lower
    data = data.lower()     
    for i in list_stopwords:
        data = data.replace(i,' ')  
    return data     


# In[9]:


#https://stackoverflow.com/questions/12851791/removing-numbers-from-string
#preprocessing of the data by removing unwanted digits and repeated symbols in the data
def preprocess(data):
    #removing digits
    data = ''.join([i for i in data if not i.isdigit()])

    
    #https://stackoverflow.com/questions/12628958/remove-small-words-using-python
    shortword = re.compile(r'\W*\b\w{1,3}\b')
    #remove symbols which are repeating
    #https://stackoverflow.com/questions/1660694/regular-expression-to-match-any-character-being-repeated-more-than-10-times
    test = re.compile(r'(.)\1{9,}')
    data = shortword.sub('', data)
    data = test.sub('', data)
    #https://www.w3schools.com/python/ref_string_split.asp
    words = data.split(' ')    
    return words


# In[10]:


#function to get the total words dictionary of bag of words
def totalcal(words,total_dic,dic):
    #continue for spcaces
    for word in words:
        if word == ' ':
            continue
        if word == '':
            continue
        #reference: https://www.programiz.com/python-programming/methods/dictionary/get
        #get the value of the word if present otherwise 0
        hold = dic.get(word, 0)
        hold_t = total_dic.get(word, 0)
        #if it is not present then set to 1
        if hold == 0:
            dic[word] = 1
        else:
            #if it is present then increment with 1 for count
            dic[word] = hold + 1
        if hold_t == 0:
            total_dic[word] = 1
        else:
            total_dic[word] = hold_t + 1
    return total_dic, dic
        


# In[11]:


#bag of words calculation by giving training data value and filename
#https://towardsdatascience.com/word-bags-vs-word-sequences-for-text-classification-e0222c21d2ec
def bagofwords(data_train, filename):
    folder_list = os.listdir(data_path)  
    total_dic,bag_dict = {}, {}

    print(" bag of words calculation")
    print("\n Reading all the folders one by one")
    for each_folder in folder_list:
        dic = {}
        folder_ = data_path + each_folder
        print(each_folder)
        files = os.listdir(folder_)     
        count = 0
        #iterate over the files to get the total count
        for file in files:
            count = count + 1
            #break once finished with training dataset which is 50% of the total data
            if count > data_train:     
                break
            check = folder_ + '/'+file
            #open the file to read 
            #https://www.w3schools.com/python/python_file_open.asp
            currentFile = open(check,'r')
            #read the file and remove stopwords from the data
            data = remove_stopwords(currentFile.read())    
            
            #preprocessing data
            words = preprocess(data)
            #dictionary for total number of words as a bag of words from the text
            total_dic, dic = totalcal(words,total_dic,dic)
            #remove file from list once done
            files.remove(file)
        filename[each_folder] = files
        bag_dict[each_folder] = dic
    print("\nTotal number of bag of words ", len(total_dic), "words" )
    return folder_list, bag_dict


# In[12]:


folder_list, bag_dict = bagofwords(data_train, filename)


# In[13]:


# folder_list


# In[14]:


print("\n dictionary of the words in one class:\n",bag_dict['comp.graphics'])


# In[15]:



#dictionary of all the words 
print("\n dictionary of all the words:\n",bag_dict)


# In[16]:


# total_dic


# In[17]:


folder_l = copy.copy(folder_list)


# In[18]:


# folder_l


# In[19]:


def getdata():
    
    global loc
    #iterate to get the data from the file from the particular dataset
    #https://docs.python.org/3/library/random.html
    while (len(folder_l)):
        temp = random.randint(0,len(folder_l)-1)
        curr_folder = folder_l[temp]
        #check if file exits randomly return null if it doesnt
        if len(filename[curr_folder])== 0:
            folder_l.remove(curr_folder)
        else:
            #read data from the file inside the folder 
#             https://docs.python.org/3/library/random.html
            tempi = random.randint(0, len(filename[curr_folder])-1)
            file = filename[curr_folder][tempi]
            filename[curr_folder].remove(file)
            loc = curr_folder
            #open and read the file and then return the data
            data = open(data_path + curr_folder + '/'+ file,'r')
            return data.read()
    loc = 'null'
    return 'null'


# In[20]:


def probability_calculation(words, dic):
    #list of dictionary values
    Total_dic_values = dic.values()
    #sum of all the values in the dictionary
    Sum_dic_values = sum(Total_dic_values)
    #Initialising probability for prob calculation
    probability = 0.0
    #iterate total words to calculate probability by one by one dictionary words to get the total probability
    for word in words:
        p_cal = dic.get(word, 0.0) + 0.0001
        probability = probability + math.log(float(p_cal)/float(Sum_dic_values))
    return probability


# In[21]:


#https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
#function to calculate the accuracy using probability calculation of total number of hits
from sklearn.metrics import classification_report
def Accuracy_cal(hit,count,y_true,y_pred):
    print('\nAccuracy = %.1f'% (float(hit)/float(count - 1)*100))
    print("\nConfusion Martix : \n")
    print(confusion_matrix(y_true, y_pred))
    print("\n .......Classification Report.....")
    print(classification_report(y_true, y_pred))

#     global t1  
    t1 = float(hit)/float(count - 1)*100
    return t1


# In[22]:


def bayes(folder_l):
    print("\nCalculating Accuracy using Naive Bayes")
 
    y_pred = []
    y_true = []
    #initializing the values along with the list declaration
    hit, count, data = 0, 0, 1
    #calculate the probability using bayes theorem
    while (data):
        data = getdata()
        count = count + 1
        if data =='null':
            break
        data = remove_stopwords(data)
        
        #preprocessing data
        words = preprocess(data)
        #https://www.programiz.com/python-programming/methods/list/remove
        #remove all the unnecessary spaces from the data
        if words == ' ':
            words = words.remove(' ')
        if words == '':
            words = words.remove('')
        

        prob_list = []
        #calculate the probability by finding out maximum value
        for each_folder in folder_list:
            prob = probability_calculation(words,bag_dict[each_folder])
            prob_list.append(prob)
            maxval_index = prob_list.index(max(prob_list))
            y_pred.append(folder_list[maxval_index])
            y_true.append(loc)
        if loc == folder_list[maxval_index]:
            hit = hit + 1
    global t1
    t1 = Accuracy_cal(hit,count,y_true,y_pred)


# In[23]:


bayes(folder_l)


# <h1> Implementation with different stop words </h1>

# In[24]:


from nltk.corpus import stopwords #generate stopwords for comparison
stop_words = list(stopwords.words('english'))
    


# In[25]:


stop_words_2 = len(stop_words)
print("Total number of stop words",stop_words_2)


# In[26]:


def remove_stopwords1(data1):
    #remove unnecessary data from the data
    #remove extra spaces at end of the line
    data1 = data1.replace('\n', ' ')  
    data1 = data1.replace('\t', ' ')      
    data1 = data1.lower() 
    #remove stop words from the data
    for i in stop_words:
        data1 = data1.replace(i,' ')  
    return data1 


# In[27]:


def bagofwords1(data_train, filename):
    folder_list1 = os.listdir(data_path)  
    total_dic1,bag_dict1 = {}, {}
#     bag_dict1 = {}
    print("bag of words calculation")
    print("\n Reading all the folders one by one")

    for fo in folder_list1:
        dic1 = {}
        folder_ = data_path + fo
        print(fo)
        files = os.listdir(folder_)     
        count = 0
        for fi in files:
            count = count + 1
            #break once reach end of the training data
            if count > data_train:     
                break
            check = folder_ + '/'+fi
            currentFile = open(check,'r')   
            #remove stopwords from the currentdata from the current file
            data1 = remove_stopwords1(currentFile.read())    
            
            #preprocessing data
            words1 = preprocess(data1)
            
            total_dic1, dic1 = totalcal(words1,total_dic1,dic1)
            
                
            files.remove(fi)
        filename[fo] = files
        bag_dict1[fo] = dic1
    print("\nTotal number of bag of words", len(total_dic1), "words" )
    return folder_list1, bag_dict1


# In[28]:


folder_list1, bag_dict1 = bagofwords1(data_train, filename)


# In[29]:


#https://docs.python.org/3/library/copy.html
#copy the value
folder_l1 = copy.copy(folder_list1)


# In[30]:


def getdata1():
    
    global loc
    #iterate to get the data from the file from the particular dataset
    #https://docs.python.org/3/library/random.html
    while (len(folder_l1)):
        r_fo = random.randint(0,len(folder_l1)-1)
        curr_folder = folder_l1[r_fo]
        if len(filename[curr_folder])== 0:
            folder_l1.remove(curr_folder)
        else:
            #read data from the file inside the folder
            r_fi = random.randint(0, len(filename[curr_folder])-1)
            fil = filename[curr_folder][r_fi]
            filename[curr_folder].remove(fil)
            loc = curr_folder
            data = open(data_path + curr_folder + '/'+ fil,'r')
            return data.read()
    loc = 'null'
    return 'null'


# In[31]:


def bayes1(folder_l1):
    print("\nCalculating Accuracy")

    y_pred1 = []
    y_true1 = []
    #intialisation and declaration
    hit1, count1, data1 = 0, 0, 1
    #get the data after preprocessing and removing extra spaces then calculate the probability using bayes theorem to 
    # calculate the maximum probability to get the exact value which can be further use for accuracy calculation
    while (data1):
        data1 = getdata1()
        count1 = count1 + 1
        if data1 =='null':
            break
        data1 = remove_stopwords1(data1)
        
        #preprocessing data
        words1 = preprocess(data1)
            
        if words1 == ' ':
            words1 = words1.remove(' ')
        if words1 == '':
            words1 = words1.remove('')
        
        #list to add all the probabilities
        prob_list1 = []
        for each_folder1 in folder_list1:
            prob1 = probability_calculation(words1,bag_dict1[each_folder1])
            prob_list1.append(prob1)
            #taking maximum value of probability
            maxval_index1 = prob_list1.index(max(prob_list1))
            y_pred1.append(folder_list1[maxval_index1])
            y_true1.append(loc)
        if loc == folder_list1[maxval_index1]:
            hit1 = hit1 + 1
    global t2
    #accuracy and confusion matrix calculation
    t2 = Accuracy_cal(hit1,count1,y_true1,y_pred1)


# In[32]:


bayes1(folder_l1)


# In[33]:


print("Accuracy from stopword1:",t1)


# In[34]:


print("Accuracy from stopword2:",t2)


# In[35]:


#Reference: https://www.tutorialspoint.com/matplotlib/matplotlib_bar_plot.htm
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
#plot the values on x axis
stop_words_ = [stop_words_1, stop_words_2]
#plot the values on y-axis
Accuracy = [t1,t2]
ax.bar(stop_words_,Accuracy)
ax.set_ylabel('Accuracy')
ax.set_xlabel('No of StopWords')
plt.show()


# In[ ]:




