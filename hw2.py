
# coding: utf-8

# In[217]:


import sklearn
import numpy
import os


class Classifier(object):

    def __init__(self, algorithm, x_train, y_train, iterations=1, averaged = False, eta = 1, alpha = 1.005):

        # Get features from examples; this line figures out what features are present in
        # the training data, such as 'w-1=dog' or 'w+1=cat'
        features = {feature for xi in x_train for feature in xi.keys()}

        if algorithm == 'Perceptron':
            if averaged:
                v_past, v_past['bias'] = {feature:0.0 for feature in features},0.0 ; c_current = 0;
                self.w, self.w['bias'] = {feature:0.0 for feature in features}, 0.0
                for i in range(iterations):
                    for i in range(len(x_train)):
                        xi, yi = x_train[i], y_train[i]
                        y_hat = self.predict(xi)
                        if yi == y_hat:
                            c_current = c_current + 1
                        else:
                            for x in v_past:
                                v_past[x] = v_past[x] + self.w[x]*c_current
                            for feature, value in xi.items():
                                self.w[feature] = self.w[feature] + yi*eta*value
                            self.w['bias'] = self.w['bias'] + yi*eta
                            c_current = 0
                for x in v_past:
                    v_past[x] = v_past[x]/(iterations*len(x_train))
                self.w = v_past
                
            else:
                #Initialize w, bias
                self.w, self.w['bias'] = {feature:0.0 for feature in features}, 0.0
                #Iterate over the training data n times
                for i in range(iterations):
                    #Check each training example
                    for i in range(len(x_train)):
                        xi, yi = x_train[i], y_train[i]
                        y_hat = self.predict(xi)
                        #Update weights if there is a misclassification
                        if yi != y_hat:
                            for feature, value in xi.items():
                                self.w[feature] = self.w[feature] + yi*eta*value
                            self.w['bias'] = self.w['bias'] + yi*eta
                        
        if algorithm == 'Winnow':
            if averaged:
                v_past, v_past['bias'] = {feature:1.0 for feature in features},-len(features) ; c_current = 0;
                self.w, self.w['bias'] = {feature:1.0 for feature in features}, -len(features)
                for i in range(iterations):
                    for i in range(len(x_train)):
                        xi, yi = x_train[i], y_train[i]
                        y_hat = self.predict(xi)
                        if yi == y_hat:
                            c_current = c_current + 1
                        else:
                            for x in v_past:
                                v_past[x] = v_past[x] + self.w[x]*c_current
                            for feature, value in xi.items():
                                self.w[feature] = self.w[feature] * alpha**(yi*value)
                            c_current = 0
                for x in v_past:
                    v_past[x] = v_past[x]/(iterations*len(x_train))
                self.w = v_past
     
            else:
                #Initialize w, bias
                self.w, self.w['bias'] = {feature:1.0 for feature in features}, -len(features)
                #Iterate over the training data n times
                for i in range(iterations):
                    #Check each training example
                    for i in range(len(x_train)):
                        xi, yi = x_train[i], y_train[i]
                        y_hat = self.predict(xi)
                        #Update weights if there is a misclassification
                        if yi != y_hat:
                            for feature, value in xi.items():
                                self.w[feature] = self.w[feature] * alpha**(yi*value)
                            
        if algorithm == 'AdaGrad':
            if averaged:
                v_past, v_past['bias'] = {feature:0.0 for feature in features},0.0 ; c_current = 0;
                self.w, self.w['bias'] = {feature:0.0 for feature in features}, 0.0
                grad, grad['bias'] = {feature:0.0 for feature in features}, 0.0
                for i in range(iterations):
                    for i in range(len(x_train)):
                        xi, yi = x_train[i], y_train[i]
                        y_hat = yi * self.predictAdaGrad(xi)
                        if y_hat > 1:
                            c_current = c_current + 1
                        else:
                            for x in v_past:
                                v_past[x] = v_past[x] + self.w[x]*c_current
                            for feature, value in xi.items():
                                grad[feature] = grad[feature] + (yi*value)**2
                                self.w[feature] = self.w[feature] + (yi*eta*value)/grad[feature]**(1.0/2.0)
                            grad['bias'] = grad['bias'] + yi**2
                            self.w['bias'] = self.w['bias'] + (yi*eta)/grad['bias']**(1.0/2.0)
                            c_current = 0
                for x in v_past:
                    v_past[x] = v_past[x]/(iterations*len(x_train))
                self.w = v_past
     
            else:
               #Initialize w, bias, gradient
                    self.w, self.w['bias'] = {feature:0.0 for feature in features}, 0.0
                    grad, grad['bias'] = {feature:0.0 for feature in features}, 0.0
                    #Iterate over the training data n times
                    for i in range(iterations):
                        #Check each training example
                        for i in range(len(x_train)):
                            xi, yi = x_train[i], y_train[i]
                            y_hat = yi * self.predictAdaGrad(xi)
                            #Update weights if there is a misclassification
                            if y_hat <= 1:
                                grad['bias'] = grad['bias'] + yi**2
                                self.w['bias'] = self.w['bias'] + (yi*eta)/grad['bias']**(1.0/2.0)
                                for feature, value in xi.items():
                                    grad[feature] = grad[feature] + (yi*value)**2
                                    self.w[feature] = self.w[feature] + (yi*eta*value)/grad[feature]**(1.0/2.0)

    def predict(self, x):
        s = sum([self.w[feature]*value for feature, value in x.items()]) + self.w['bias']
        return 1 if s > 0 else -1
    
    def predictAdaGrad(self, x):
        s = sum([self.w[feature]*value for feature, value in x.items()]) + self.w['bias']
        return s

#Parse the real-world data to generate features, 
#Returns a list of tuple lists
def parse_real_data(path):
    #List of tuples for each sentence
    data = []
    for filename in os.listdir(path):
        with open(path+filename, 'r', encoding='ascii') as file:
            sentence = []
            for line in file:
                if line == '\n':
                    data.append(sentence)
                    sentence = []
                else:
                    sentence.append(tuple(line.split()))
    return data

#Returns a list of labels
def parse_synthetic_labels(path):
    #List of tuples for each sentence
    labels = []
    with open(path+'y.txt', 'rb') as file:
        for line in file:
            labels.append(int(line.strip()))
    return labels

#Returns a list of features
def parse_synthetic_data(path):
    #List of tuples for each sentence
    data = []
    with open(path+'x.txt') as file:
        features = []
        for line in file:
            #print('Line:', line)
            for ch in line:
                if ch == '[' or ch.isspace():
                    continue
                elif ch == ']':
                    data.append(features)
                    features = []
                else:
                    features.append(int(ch))
    return data

if __name__ == '__main__':
    print('Loading data...')
    #Load data from folders.
    #Real world data - lists of tuple lists
    news_train_data = parse_real_data('Data/Real-World/CoNLL/train/')
    news_dev_data = parse_real_data('Data/Real-World/CoNLL/dev/')
    news_test_data = parse_real_data('Data/Real-World/CoNLL/test/')
    email_dev_data = parse_real_data('Data/Real-World/Enron/dev/')
    email_test_data = parse_real_data('Data/Real-World/Enron/test/')
    
    # #Load dense synthetic data
    syn_dense_train_data = parse_synthetic_data('Data/Synthetic/Dense/train/')
    syn_dense_train_labels = parse_synthetic_labels('Data/Synthetic/Dense/train/')
    syn_dense_dev_data = parse_synthetic_data('Data/Synthetic/Dense/dev/')
    syn_dense_dev_labels = parse_synthetic_labels('Data/Synthetic/Dense/dev/')
    syn_dense_test_data = parse_synthetic_data('Data/Synthetic/Dense/test/')

    
    #Load sparse synthetic data
    syn_sparse_train_data = parse_synthetic_data('Data/Synthetic/Sparse/train/')
    syn_sparse_train_labels = parse_synthetic_labels('Data/Synthetic/Sparse/train/')
    syn_sparse_dev_data = parse_synthetic_data('Data/Synthetic/Sparse/dev/')
    syn_sparse_dev_labels = parse_synthetic_labels('Data/Synthetic/Sparse/dev/')
    syn_sparse_test_data = parse_synthetic_data('Data/Synthetic/Sparse/test/')


    # Convert to sparse dictionary representations.
    # Examples are a list of tuples, where each tuple consists of a dictionary
    # and a lable. Each dictionary contains a list of features and their values,
    # i.e a feature is included in the dictionary only if it provides information. 

    # You can use sklearn.feature_extraction.DictVectorizer() to convert these into
    # scipy.sparse format to train SVM, or for your Perceptron implementation.
    print('Converting Synthetic data...')
    syn_dense_train = zip(*[({'x'+str(i): syn_dense_train_data[j][i]
        for i in range(len(syn_dense_train_data[j])) if syn_dense_train_data[j][i] == 1}, syn_dense_train_labels[j]) 
            for j in range(len(syn_dense_train_data))])
    syn_dense_train_x, syn_dense_train_y = syn_dense_train
    syn_dense_dev = zip(*[({'x'+str(i): syn_dense_dev_data[j][i]
        for i in range(len(syn_dense_dev_data[j])) if syn_dense_dev_data[j][i] == 1}, syn_dense_dev_labels[j]) 
            for j in range(len(syn_dense_dev_data))])
    syn_dense_dev_x, syn_dense_dev_y = syn_dense_dev
    syn_dense_test = [{'x' + str(i): syn_dense_test_data[j][i]
        for i in range(len(syn_dense_test_data[j])) if syn_dense_test_data[j][i] == 1}
            for j in range(len(syn_dense_test_data))]
    syn_dense_test_x = syn_dense_test
    
    syn_sparse_train = zip(*[({'x'+str(i): syn_sparse_train_data[j][i]
        for i in range(len(syn_sparse_train_data[j])) if syn_sparse_train_data[j][i] == 1}, syn_sparse_train_labels[j]) 
            for j in range(len(syn_sparse_train_data))])
    syn_sparse_train_x, syn_sparse_train_y = syn_sparse_train
    syn_sparse_dev = zip(*[({'x'+str(i): syn_sparse_dev_data[j][i]
        for i in range(len(syn_sparse_dev_data[j])) if syn_sparse_dev_data[j][i] == 1}, syn_sparse_dev_labels[j]) 
            for j in range(len(syn_sparse_dev_data))])
    syn_sparse_dev_x, syn_sparse_dev_y = syn_sparse_dev
    syn_sparse_test = [{'x' + str(i): syn_sparse_test_data[j][i]
        for i in range(len(syn_sparse_test_data[j])) if syn_sparse_test_data[j][i] == 1}
            for j in range(len(syn_sparse_test_data))]
    syn_sparse_test_x = syn_sparse_test

     # Feature extraction
    print('Extracting features from real-world data...')
    news_train_y = []
    news_train_x = []
    train_features = set([])
    for sentence in news_train_data:
        padded = sentence[:]
        padded.insert(0, ('SSS', None))
        padded.insert(0, ('SSS', None))
        padded.append(('EEE', None))
        padded.append(('EEE', None))
        for i in range(2,len(padded)-2):
            news_train_y.append(1 if padded[i][1]=='I' else -1)
            feat1 = 'w-2='+str(padded[i-2][0])
            feat2 = 'w-1='+str(padded[i-1][0])
            feat3 = 'w+1='+str(padded[i+1][0])
            feat4 = 'w+2='+str(padded[i+2][0])
            feat5 = 'w-2&w-1='+str(padded[i-2][0])+' '+str(padded[i-1][0])
            feat6 = 'w+1&w+2='+str(padded[i+1][0])+' '+str(padded[i+2][0])
            feat7 = 'w-1&w+1='+str(padded[i-1][0])+' '+str(padded[i+1][0])
            feats = [feat1, feat2, feat3, feat4, feat5, feat6, feat7]
            train_features.update(feats)
            feats = {feature:1 for feature in feats}
            news_train_x.append(feats)
    news_dev_y = []
    news_dev_x = []
    for sentence in news_dev_data:
        padded = sentence[:]
        padded.insert(0, ('SSS', None))
        padded.insert(0, ('SSS', None))
        padded.append(('EEE', None))
        padded.append(('EEE', None))
        for i in range(2,len(padded)-2):
            news_dev_y.append(1 if padded[i][1]=='I' else -1)
            feat1 = 'w-2='+str(padded[i-2][0])
            feat2 = 'w-1='+str(padded[i-1][0])
            feat3 = 'w+1='+str(padded[i+1][0])
            feat4 = 'w+2='+str(padded[i+2][0])
            feat5 = 'w-2&w-1='+str(padded[i-2][0])+' '+str(padded[i-1][0])
            feat6 = 'w+1&w+2='+str(padded[i+1][0])+' '+str(padded[i+2][0])
            feat7 = 'w-1&w+1='+str(padded[i-1][0])+' '+str(padded[i+1][0])
            feats = [feat1, feat2, feat3, feat4, feat5, feat6, feat7]
            feats = {feature:1 for feature in feats if feature in train_features}
            news_dev_x.append(feats)
    news_test_x = []
    for sentence in news_test_data:
        padded = sentence[:]
        padded.insert(0, ('SSS', None))
        padded.insert(0, ('SSS', None))
        padded.append(('EEE', None))
        padded.append(('EEE', None))
        for i in range(2,len(padded)-2):
            feat1 = 'w-2='+str(padded[i-2][0])
            feat2 = 'w-1='+str(padded[i-1][0])
            feat3 = 'w+1='+str(padded[i+1][0])
            feat4 = 'w+2='+str(padded[i+2][0])
            feat5 = 'w-2&w-1='+str(padded[i-2][0])+' '+str(padded[i-1][0])
            feat6 = 'w+1&w+2='+str(padded[i+1][0])+' '+str(padded[i+2][0])
            feat7 = 'w-1&w+1='+str(padded[i-1][0])+' '+str(padded[i+1][0])
            feats = [feat1, feat2, feat3, feat4, feat5, feat6, feat7]
            feats = {feature:1 for feature in feats if feature in train_features}
            news_test_x.append(feats)
    email_dev_y = []
    email_dev_x = []
    for sentence in email_dev_data:
        padded = sentence[:]
        padded.insert(0, ('SSS', None))
        padded.insert(0, ('SSS', None))
        padded.append(('EEE', None))
        padded.append(('EEE', None))
        for i in range(2,len(padded)-2):
            email_dev_y.append(1 if padded[i][1]=='I' else -1)
            feat1 = 'w-2='+str(padded[i-2][0])
            feat2 = 'w-1='+str(padded[i-1][0])
            feat3 = 'w+1='+str(padded[i+1][0])
            feat4 = 'w+2='+str(padded[i+2][0])
            feat5 = 'w-2&w-1='+str(padded[i-2][0])+' '+str(padded[i-1][0])
            feat6 = 'w+1&w+2='+str(padded[i+1][0])+' '+str(padded[i+2][0])
            feat7 = 'w-1&w+1='+str(padded[i-1][0])+' '+str(padded[i+1][0])
            feats = [feat1, feat2, feat3, feat4, feat5, feat6, feat7]
            feats = {feature:1 for feature in feats if feature in train_features}
            email_dev_x.append(feats)
    email_test_x = []
    for sentence in email_test_data:
        padded = sentence[:]
        padded.insert(0, ('SSS', None))
        padded.insert(0, ('SSS', None))
        padded.append(('EEE', None))
        padded.append(('EEE', None))
        for i in range(2,len(padded)-2):
            feat1 = 'w-2='+str(padded[i-2][0])
            feat2 = 'w-1='+str(padded[i-1][0])
            feat3 = 'w+1='+str(padded[i+1][0])
            feat4 = 'w+2='+str(padded[i+2][0])
            feat5 = 'w-2&w-1='+str(padded[i-2][0])+' '+str(padded[i-1][0])
            feat6 = 'w+1&w+2='+str(padded[i+1][0])+' '+str(padded[i+2][0])
            feat7 = 'w-1&w+1='+str(padded[i-1][0])+' '+str(padded[i+1][0])
            feats = [feat1, feat2, feat3, feat4, feat5, feat6, feat7]
            feats = {feature:1 for feature in feats if feature in train_features}
            email_test_x.append(feats)

    #Print results
    print('\nPerceptron Accuracy')

    # Test Perceptron on Dense Synthetic
    p = Classifier('Perceptron', syn_dense_train_x, syn_dense_train_y)
    accuracy = sum([1 for i in range(len(syn_dense_dev_y)) if p.predict(syn_dense_dev_x[i]) == syn_dense_dev_y[i]])/len(syn_dense_dev_y)*100
    print('Syn Dense Dev Accuracy:', accuracy)
    
    # Test Perceptron on Sparse Synthetic
    p = Classifier('Perceptron', syn_sparse_train_x, syn_sparse_train_y)
    accuracy = sum([1 for i in range(len(syn_sparse_dev_y)) if p.predict(syn_sparse_dev_x[i]) == syn_sparse_dev_y[i]])/len(syn_sparse_dev_y)*100
    print('Syn Sparse Dev Accuracy:', accuracy)

    # Test Perceptron on Real World Data
    p = Classifier('Perceptron', news_train_x, news_train_y, iterations=10)
    accuracy = sum([1 for i in range(len(news_dev_y)) if p.predict(news_dev_x[i]) == news_dev_y[i]])/len(news_dev_y)*100
    print('News Dev Accuracy:', accuracy)


# In[210]:


print('\nWinnow Accuracy')

# Test Winnow on Dense Synthetic
p = Classifier('Winnow', syn_dense_train_x, syn_dense_train_y)
accuracy = sum([1 for i in range(len(syn_dense_dev_y)) if p.predict(syn_dense_dev_x[i]) == syn_dense_dev_y[i]])/len(syn_dense_dev_y)*100
print('Syn Dense Dev Accuracy:', accuracy)

 # Test Winnow on Sparse Synthetic
p = Classifier('Winnow', syn_sparse_train_x, syn_sparse_train_y)
accuracy = sum([1 for i in range(len(syn_sparse_dev_y)) if p.predict(syn_sparse_dev_x[i]) == syn_sparse_dev_y[i]])/len(syn_sparse_dev_y)*100
print('Syn Sparse Dev Accuracy:', accuracy)

# Test Winnow on Real World Data
p = Classifier('Winnow', news_train_x, news_train_y, iterations=10)
accuracy = sum([1 for i in range(len(news_dev_y)) if p.predict(news_dev_x[i]) == news_dev_y[i]])/len(news_dev_y)*100
print('News Dev Accuracy:', accuracy)


# In[211]:


print('\nAdaGrad Accuracy')
   
# Test AdaGrad on Dense Synthetic
p = Classifier('AdaGrad', syn_dense_train_x, syn_dense_train_y,eta = 1.5)
accuracy = sum([1 for i in range(len(syn_dense_dev_y)) if p.predict(syn_dense_dev_x[i]) == syn_dense_dev_y[i]])/len(syn_dense_dev_y)*100
print('Syn Dense Dev Accuracy:', accuracy)
   
# Test AdaGrad on Sparse Synthetic
p = Classifier('AdaGrad', syn_sparse_train_x, syn_sparse_train_y,eta = 1.5)
accuracy = sum([1 for i in range(len(syn_sparse_dev_y)) if p.predict(syn_sparse_dev_x[i]) == syn_sparse_dev_y[i]])/len(syn_sparse_dev_y)*100
print('Syn Sparse Dev Accuracy:', accuracy)

# Test AdaGrad on Real World Data
p = Classifier('AdaGrad', news_train_x, news_train_y, iterations=10, eta = 1.5)
accuracy = sum([1 for i in range(len(news_dev_y)) if p.predict(news_dev_x[i]) == news_dev_y[i]])/len(news_dev_y)*100
print('News Dev Accuracy:', accuracy)


# In[214]:


#Print results
print('\nAveraged Perceptron Accuracy')

# Test Averaged Perceptron on Dense Synthetic
p = Classifier('Perceptron', syn_dense_train_x, syn_dense_train_y, 1,averaged = True)
accuracy = sum([1 for i in range(len(syn_dense_dev_y)) if p.predict(syn_dense_dev_x[i]) == syn_dense_dev_y[i]])/len(syn_dense_dev_y)*100
print('Syn Dense Dev Accuracy:', accuracy)

# Test Averaged Perceptron on Sparse Synthetic
p = Classifier('Perceptron', syn_sparse_train_x, syn_sparse_train_y, 1, averaged = True)
accuracy = sum([1 for i in range(len(syn_sparse_dev_y)) if p.predict(syn_sparse_dev_x[i]) == syn_sparse_dev_y[i]])/len(syn_sparse_dev_y)*100
print('Syn Sparse Dev Accuracy:', accuracy)


# In[120]:


print('\nAveraged Winnow Accuracy')

# Test Averaged Winnow on Dense Synthetic
p = Classifier('Winnow', syn_dense_train_x, syn_dense_train_y, True)
accuracy = sum([1 for i in range(len(syn_dense_dev_y)) if p.predict(syn_dense_dev_x[i]) == syn_dense_dev_y[i]])/len(syn_dense_dev_y)*100
print('Syn Dense Dev Accuracy:', accuracy)

 # Test Averaged Winnow on Sparse Synthetic
p = Classifier('Winnow', syn_sparse_train_x, syn_sparse_train_y, True)
accuracy = sum([1 for i in range(len(syn_sparse_dev_y)) if p.predict(syn_sparse_dev_x[i]) == syn_sparse_dev_y[i]])/len(syn_sparse_dev_y)*100
print('Syn Sparse Dev Accuracy:', accuracy)


# In[124]:


print('\nAveraged AdaGrad Accuracy')
   
# Test Averaged AdaGrad on Dense Synthetic
p = Classifier('AdaGrad', syn_dense_train_x, syn_dense_train_y, True, eta = 1.5)
accuracy = sum([1 for i in range(len(syn_dense_dev_y)) if p.predict(syn_dense_dev_x[i]) == syn_dense_dev_y[i]])/len(syn_dense_dev_y)*100
print('Syn Dense Dev Accuracy:', accuracy)
   
# Test Averaged AdaGrad on Sparse Synthetic
p = Classifier('AdaGrad', syn_sparse_train_x, syn_sparse_train_y, True, eta = 1.5)
accuracy = sum([1 for i in range(len(syn_sparse_dev_y)) if p.predict(syn_sparse_dev_x[i]) == syn_sparse_dev_y[i]])/len(syn_sparse_dev_y)*100
print('Syn Sparse Dev Accuracy:', accuracy)


# In[77]:


from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import LinearSVC
# Test SVM on Dense Synthetic
vect = DictVectorizer(sparse = False)
vect_train_x = vect.fit_transform(syn_dense_train_x)
vect_dev_x = vect.transform(syn_dense_dev_x)
dense = LinearSVC()
dense.fit(vect_train_x, syn_dense_train_y)
y_predict = dense.predict(vect_dev_x)
accuracy = sum([1 for i in range(len(syn_dense_dev_y)) if y_predict[i] == syn_dense_dev_y[i]])/len(syn_dense_dev_y)*100
print('SVM Syn Dense Dev Accuracy:', accuracy)


# In[79]:


# Test SVM on Sparse Synthetic
vect = DictVectorizer()
vect_train_x = vect.fit_transform(syn_sparse_train_x)
vect_dev_x = vect.transform(syn_sparse_dev_x)
sparse = LinearSVC()
sparse.fit(vect_train_x, syn_sparse_train_y)
y_predict = sparse.predict(vect_dev_x)
accuracy = sum([1 for i in range(len(syn_sparse_dev_y)) if y_predict[i] == syn_sparse_dev_y[i]])/len(syn_sparse_dev_y)*100
print('SVM Syn Sparse Dev Accuracy:', accuracy)


# In[138]:


# Test SVM on News Data
vect = DictVectorizer()
vect_train_x = vect.fit_transform(news_train_x)
vect_dev_x = vect.transform(news_dev_x)
news = LinearSVC()
news.fit(vect_train_x, news_train_y)
y_predict = news.predict(vect_dev_x)
accuracy = sum([1 for i in range(len(news_dev_y)) if y_predict[i] == news_dev_y[i]])/len(news_dev_y)*100
print('SVM News Dev Accuracy:', accuracy)


# In[106]:


#EXPERIMENTS
#Perceptron 
#Print results
# Test Perceptron on Dense Synthetic
print('\nPerceptron Dense Accuracy')
num = 0;
for x in range(11):
    if(x < 10):
        num = num + 500
    else:
        num = 50000
    p = Classifier('Perceptron', syn_dense_train_x[:num], syn_dense_train_y[:num], iterations= 20)
    accuracy = sum([1 for i in range(len(syn_dense_dev_y)) if p.predict(syn_dense_dev_x[i]) == syn_dense_dev_y[i]])/len(syn_dense_dev_y)*100
    print('Syn Dense Dev Accuracy:', accuracy)
    
# Test Perceptron on Sparse Synthetic
print('\nPerceptron Sparse Accuracy')
num = 0;
for x in range(11):
    if(x < 10):
        num = num + 500
    else:
        num = 50000
    p = Classifier('Perceptron', syn_sparse_train_x[:num], syn_sparse_train_y[:num], iterations= 20)
    accuracy = sum([1 for i in range(len(syn_sparse_dev_y)) if p.predict(syn_sparse_dev_x[i]) == syn_sparse_dev_y[i]])/len(syn_sparse_dev_y)*100
    print('Syn Sparse Dev Accuracy:', accuracy)
    


# In[218]:


#Winnow 
#Print results
# Test Winnow on Dense Synthetic
print('\nWinnow Dense Accuracy')
num = 0;
for x in range(11):
    if(x < 10):
        num = num + 500
    else:
        num = 50000
    p = Classifier('Winnow', syn_dense_train_x[:num], syn_dense_train_y[:num], iterations= 20)
    accuracy = sum([1 for i in range(len(syn_dense_dev_y)) if p.predict(syn_dense_dev_x[i]) == syn_dense_dev_y[i]])/len(syn_dense_dev_y)*100
    print('Syn Dense Dev Accuracy:', accuracy)
    
# Test Winnow on Sparse Synthetic
print('\nWinnow Sparse Accuracy')
num = 0;
for x in range(11):
    if(x < 10):
        num = num + 500
    else:
        num = 50000
    p = Classifier('Winnow', syn_sparse_train_x[:num], syn_sparse_train_y[:num], iterations= 20)
    accuracy = sum([1 for i in range(len(syn_sparse_dev_y)) if p.predict(syn_sparse_dev_x[i]) == syn_sparse_dev_y[i]])/len(syn_sparse_dev_y)*100
    print('Syn Sparse Dev Accuracy:', accuracy)



# In[111]:


#AdaGrad
#Print results
# Test AdaGrad on Dense Synthetic
print('\nAdaGrad Dense Accuracy')
num = 0;
for x in range(11):
    if(x < 10):
        num = num + 500
    else:
        num = 50000
    p = Classifier('AdaGrad', syn_dense_train_x[:num], syn_dense_train_y[:num], iterations= 20)
    accuracy = sum([1 for i in range(len(syn_dense_dev_y)) if p.predict(syn_dense_dev_x[i]) == syn_dense_dev_y[i]])/len(syn_dense_dev_y)*100
    print('Syn Dense Dev Accuracy:', accuracy)
    
# Test AdaGrad on Sparse Synthetic
print('\nAdaGrad Sparse Accuracy')
num = 0;
for x in range(11):
    if(x < 10):
        num = num + 500
    else:
        num = 50000
    p = Classifier('AdaGrad', syn_sparse_train_x[:num], syn_sparse_train_y[:num], iterations= 20)
    accuracy = sum([1 for i in range(len(syn_sparse_dev_y)) if p.predict(syn_sparse_dev_x[i]) == syn_sparse_dev_y[i]])/len(syn_sparse_dev_y)*100
    print('Syn Sparse Dev Accuracy:', accuracy)


# In[109]:


#EXPERIMENTS
#Averaged Perceptron 
#Print results
# Test Avearged Perceptron on Dense Synthetic
print('\nAveraged Perceptron Dense Accuracy')
num = 0;
for x in range(11):
    if(x < 10):
        num = num + 500
    else:
        num = 50000
    p = Classifier('Perceptron', syn_dense_train_x[:num], syn_dense_train_y[:num], iterations= 20, averaged = True)
    accuracy = sum([1 for i in range(len(syn_dense_dev_y)) if p.predict(syn_dense_dev_x[i]) == syn_dense_dev_y[i]])/len(syn_dense_dev_y)*100
    print('Syn Dense Dev Accuracy:', accuracy)
    
# Test Averaged Perceptron on Sparse Synthetic
print('\nAveraged Perceptron Sparse Accuracy')
num = 0;
for x in range(11):
    if(x < 10):
        num = num + 500
    else:
        num = 50000
    p = Classifier('Perceptron', syn_sparse_train_x[:num], syn_sparse_train_y[:num], iterations= 20, averaged = True)
    accuracy = sum([1 for i in range(len(syn_sparse_dev_y)) if p.predict(syn_sparse_dev_x[i]) == syn_sparse_dev_y[i]])/len(syn_sparse_dev_y)*100
    print('Syn Sparse Dev Accuracy:', accuracy)


# In[219]:


# Averaged Winnow 
#Print results
# Test Averaged Winnow on Dense Synthetic
print('\nAveraged Winnow Dense Accuracy')
num = 0;
for x in range(11):
    if(x < 10):
        num = num + 500
    else:
        num = 50000
    p = Classifier('Winnow', syn_dense_train_x[:num], syn_dense_train_y[:num], iterations= 20, averaged = True)
    accuracy = sum([1 for i in range(len(syn_dense_dev_y)) if p.predict(syn_dense_dev_x[i]) == syn_dense_dev_y[i]])/len(syn_dense_dev_y)*100
    print('Syn Dense Dev Accuracy:', accuracy)
    
# Test Averaged Winnow on Sparse Synthetic
print('\nAveraged Winnow Sparse Accuracy')
num = 0;
for x in range(11):
    if(x < 10):
        num = num + 500
    else:
        num = 50000
    p = Classifier('Winnow', syn_sparse_train_x[:num], syn_sparse_train_y[:num], iterations= 20, averaged = True)
    accuracy = sum([1 for i in range(len(syn_sparse_dev_y)) if p.predict(syn_sparse_dev_x[i]) == syn_sparse_dev_y[i]])/len(syn_sparse_dev_y)*100
    print('Syn Sparse Dev Accuracy:', accuracy)


# In[220]:


#Averaged AdaGrad
#Print results
# Test Averaged AdaGrad on Dense Synthetic
print('\nAveraged AdaGrad Dense Accuracy')
num = 0;
for x in range(11):
    if(x < 10):
        num = num + 500
    else:
        num = 50000
    p = Classifier('AdaGrad', syn_dense_train_x[:num], syn_dense_train_y[:num], iterations= 20, averaged = True)
    accuracy = sum([1 for i in range(len(syn_dense_dev_y)) if p.predict(syn_dense_dev_x[i]) == syn_dense_dev_y[i]])/len(syn_dense_dev_y)*100
    print('Syn Dense Dev Accuracy:', accuracy)
    
# Test Averaged AdaGrad on Sparse Synthetic
print('\nAveraged AdaGrad Sparse Accuracy')
num = 0;
for x in range(11):
    if(x < 10):
        num = num + 500
    else:
        num = 50000
    p = Classifier('AdaGrad', syn_sparse_train_x[:num], syn_sparse_train_y[:num], iterations= 20, averaged = True)
    accuracy = sum([1 for i in range(len(syn_sparse_dev_y)) if p.predict(syn_sparse_dev_x[i]) == syn_sparse_dev_y[i]])/len(syn_sparse_dev_y)*100
    print('Syn Sparse Dev Accuracy:', accuracy)


# In[133]:


from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import LinearSVC
# Test SVM on Dense Synthetic
print('\nSVM Dense Accuracy')
vect = DictVectorizer(sparse = False)
num = 0
for x in range(11):
    num = num + 500
    if(x < 10):
        vect_train_x = vect.fit_transform(syn_dense_train_x[:num])
        vect_dev_x = vect.transform(syn_dense_dev_x)
        dense = LinearSVC()
        dense.fit(vect_train_x, syn_dense_train_y[:num])
        y_predict = dense.predict(vect_dev_x)
    else:
        num = 50000
    accuracy = sum([1 for i in range(len(syn_dense_dev_y)) if y_predict[i] == syn_dense_dev_y[i]])/len(syn_dense_dev_y)*100
    print('SVM Syn Dense Dev Accuracy:', accuracy)


# Test SVM on Sparse Synthetic
print('\nSVM Sparse Accuracy')

vect = DictVectorizer()
num = 0 
for x in range(11):
    num = num + 500
    if(x < 10):
        vect_train_x = vect.fit_transform(syn_sparse_train_x[:num])
        vect_dev_x = vect.transform(syn_sparse_dev_x)
        sparse = LinearSVC()
        sparse.fit(vect_train_x, syn_sparse_train_y[:num])
        y_predict = sparse.predict(vect_dev_x)
    else:
        num = 50000
    accuracy = sum([1 for i in range(len(syn_sparse_dev_y)) if y_predict[i] == syn_sparse_dev_y[i]])/len(syn_sparse_dev_y)*100
    print('SVM Syn Sparse Dev Accuracy:', accuracy)


# In[144]:


# Test SVM on News Data
vect = DictVectorizer()
vect_train_x = vect.fit_transform(news_train_x)
vect_dev_x = vect.transform(news_dev_x)
news = LinearSVC()
news.fit(vect_train_x, news_train_y)
y_predict = news.predict(vect_dev_x)
accuracy = sum([1 for i in range(len(news_dev_y)) if y_predict[i] == news_dev_y[i]])/len(news_dev_y)*100
print('SVM News Dev Accuracy:', accuracy)

# Test SVM on Email Data
vect2_dev_x = vect.transform(email_dev_x)
y_predict = news.predict(vect2_dev_x)
accuracy = sum([1 for i in range(len(email_dev_y)) if y_predict[i] == email_dev_y[i]])/len(email_dev_y)*100
print('SVM Email Dev Accuracy:', accuracy)


# In[221]:


#Test Labels for Synthetic Data
#Averaged Basic Perceptron
#Prediction of Dense Test Labels
p = Classifier('Perceptron', syn_dense_train_x, syn_dense_train_y, 20, averaged = True)
yTestLabels = []
for i in range(len(syn_dense_test_x)):
    yTestLabels.append(p.predict(syn_dense_test_x[i]))

#Printing labels to file
f = open("p-dense.txt", "w")
for i in range(len(yTestLabels)):
    f.write(str(yTestLabels[i]) + '\n')
f.close() 


# In[222]:


#Prediction of Sparse Test Labels
p = Classifier('Perceptron', syn_sparse_train_x, syn_sparse_train_y, 20, averaged = True)
yTestLabels = []
for i in range(len(syn_sparse_test_x)):
    yTestLabels.append(p.predict(syn_sparse_test_x[i]))

print(len(yTestLabels))
#Printing labels to file
f = open("p-sparse.txt", "w")
for i in range(len(yTestLabels)):
    f.write(str(yTestLabels[i]) + '\n')
f.close() 


# In[223]:


#Averaged Perceptron on Real World News & Email
p = Classifier('Perceptron', news_train_x, news_train_y, iterations = 3, averaged = True)
yTestLabelsNews = []
for i in range(len(news_test_x)):
    yTestLabelsNews.append(p.predict(news_test_x[i]))

print(len(yTestLabelsNews))

yTestLabelsEmail = []
for i in range(len(email_test_x)):
    yTestLabelsEmail.append(p.predict(email_test_x[i]))
    
print(len(yTestLabelsEmail))

yFinalLabelsNews = []
for x in range(len(yTestLabelsNews)):
    if(yTestLabelsNews[x] == -1):
        yFinalLabelsNews.append('O')
    else: yFinalLabelsNews.append('I')

yFinalLabelsEmail = []
for x in range(len(yTestLabelsEmail)):
    if(yTestLabelsEmail[x] == -1):
        yFinalLabelsEmail.append('O')
    else: yFinalLabelsEmail.append('I')
        
#Printing labels to file
f = open("p-conll.txt", "w")
for i in range(len(yFinalLabelsNews)):
    f.write(str(yFinalLabelsNews[i]) + '\n')
f.close()

g = open("p-enron.txt", "w")
for i in range(len(yFinalLabelsEmail)):
    g.write(str(yFinalLabelsEmail[i]) + '\n')
g.close()


# In[226]:


#SVM
#Prediction of Dense Test Labels
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import LinearSVC
# Test SVM on Dense Synthetic
vect = DictVectorizer(sparse = False)
vect_train_x = vect.fit_transform(syn_dense_train_x)
vect_test_x = vect.transform(syn_dense_test_x)
dense = LinearSVC()
dense.fit(vect_train_x, syn_dense_train_y)
y_predict = dense.predict(vect_test_x)

#Printing labels to file
f = open("svm-dense.txt", "w")
for i in range(len(y_predict)):
    f.write(str(y_predict[i]) + '\n')
f.close() 


# In[227]:


# Test SVM on Sparse Synthetic
vect = DictVectorizer()
vect_train_x = vect.fit_transform(syn_sparse_train_x)
vect_test_x = vect.transform(syn_sparse_test_x)
sparse = LinearSVC()
sparse.fit(vect_train_x, syn_sparse_train_y)
y_predict = sparse.predict(vect_test_x)

#Printing labels to file
f = open("svm-sparse.txt", "w")
for i in range(len(y_predict)):
    f.write(str(y_predict[i]) + '\n')
f.close() 


# In[228]:


#Test SVM on CoNLL
vect = DictVectorizer()
vect_train_x = vect.fit_transform(news_train_x)
vect_test_x = vect.transform(news_test_x)
news = LinearSVC()
news.fit(vect_train_x, news_train_y)
y_predict = news.predict(vect_test_x)

yFinalLabels = []
for x in range(len(y_predict)):
    if(y_predict[x] == -1):
        yFinalLabels.append('O')
    else: yFinalLabels.append('I')

#Printing labels to file
f = open("svm-conll.txt", "w")
for i in range(len(yFinalLabels)):
    f.write(str(yFinalLabels[i]) + '\n')
f.close() 


# In[229]:


#Test SVM on EMAIL
vect = DictVectorizer()
vect_train_x = vect.fit_transform(news_train_x)
vect_test_x = vect.transform(email_test_x)
news = LinearSVC()
news.fit(vect_train_x, news_train_y)
y_predict = news.predict(vect_test_x)

yFinalLabels = []
for x in range(len(y_predict)):
    if(y_predict[x] == -1):
        yFinalLabels.append('O')
    else: yFinalLabels.append('I')

#Printing labels to file
f = open("svm-enron.txt", "w")
for i in range(len(yFinalLabels)):
    f.write(str(yFinalLabels[i]) + '\n')
f.close() 

