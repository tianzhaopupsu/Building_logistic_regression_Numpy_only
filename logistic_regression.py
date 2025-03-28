import numpy as np
import matplotlib.pyplot as plt

class log_reg:

    def __init__(self,train,test):
        self.train=np.array(train)
        self.test=np.array(test)
        return 
    def logistic(self,lrate=0.01,epochs=10000):   #########default is 10000 epochs, this value can be changed based on the data quality
      
        xtrain=self.train[:,0:5]
        ytrain=self.train[:,5].reshape(-1,1)
        m,n=xtrain.shape
        weight=np.random.normal(loc=0,scale=1,size=(n,1))
        bias=0
        for ep in range(epochs):
          
            lres=np.dot(xtrain,weight)+bias
            y_pred=self.sigmoid(lres)
          #####doing gradient descent
            dweight=(1/m)*np.dot(xtrain.T,(y_pred-ytrain))
            dbias=(1/m)*np.sum(y_pred-ytrain)
            
            weight-=dweight*lrate
            bias-=dbias*lrate
          #####some printing lines, could be very annoying if epochs is a huge number, delete or live with it
            if ep%100==0:
                los=self.log_loss(ytrain,y_pred)
                print('Epoch'+str(ep)+':')
                print('loss:  '+str(los))
        return weight, bias

    def log_test(self):
        weight,bias=self.logistic()
        xtest=self.test[:,0:5]
        ytest=self.test[:,5].reshape(-1,1)
        y_pred=self.sigmoid(np.dot(xtest,weight)+bias)
        tp=0
        tn=0
        fp=0
        fn=0
        for i in range(len(y_pred)):
            if ytest[i]==1:
                if y_pred[i]>0.5:
                    tp+=1
                elif y_pred[i]<0.5:
                    fn+=1
            if ytest[i]==0:
                if y_pred[i]<0.5:
                    tn+=1
                elif y_pred[i]>0.5:
                    fp+=1
        print('Precision: '+str(tp/(tp+fp)))
        print('Recall: '+str(tp/(tp+fn)))
        return y_pred

    def log_loss(self,y,y_pred):   ############log -loss we need to make sure that log(0) wont happen
       y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
       return -(1/len(y))*np.sum(y*np.log(y_pred)+(1-y)*np.log(1-y_pred))
      
    def sigmoid(self,x):   #### this is sigmoid, like what it is named
        return 1/(1+np.exp(-x))
    

###this will be the sythetic data, 200 data points with 5 dimension for training,50% positive, 50% negative
####  and 100 data points with 65 positive case as the testing data
###make type1 distribution 100 points
feature=[3,20,59,30,5]

train=np.empty((200,6))
test=np.empty((100,6))
for i in range(5):
    train[0:100,i]=np.random.normal(loc=feature[i],scale=3,size=(100,1)).squeeze()
    train[100:200,i]=np.random.normal(loc=feature[i]*5,scale=30,size=(100,1)).squeeze()
    test[0:65,i]=np.random.normal(loc=feature[i],scale=3,size=(65,1)).squeeze()
    test[65:100,i]=np.random.normal(loc=feature[i]*5,scale=30,size=(35,1)).squeeze()
train[0:100,5]=1
train[100:200,5]=0
test[0:65,5]=1
test[65:100,5]=0
###############   we fed the training and test data here. If nothing is changed, this demo will give you an incrediably 100% precision and 100% recall. Well done.
ans=log_reg(train,test)
y_pre=ans.log_test()
plt.plot(y_pre,marker='o',linestyle='none')
plt.show()





