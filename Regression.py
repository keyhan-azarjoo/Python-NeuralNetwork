import numpy as np
import pandas as pd

"""
    Prepairing Data
"""

#Loading data
data = pd.read_excel('C:/Users/HP/OneDrive - Teesside University/Uni/Semester 2/Neural Network/Faradars/New folder/FVPHT9810-S02/Files and Codes/daily-mean-temperatures.xlsx')
data = data.to_numpy()

# Normalize Data
mn = np.min(data)
mx = np.max(data)
data_norm = (data - mn) / (mx - mn)

# use of previous four steps
step = 4
X = np.zeros(shape=(data_norm.shape[0]-step, step))
Y = []
for i in range(len(data_norm) - step):
    X[i,:] = data_norm[i:i+step,0]
    Y.append(data_norm[i+step,0])
Y = np.array(Y)

# splitting data to train and test
trn_tst_split = int(0.7*len(X))
X_train = X[0:trn_tst_split]
Y_train = Y[0:trn_tst_split]
X_test = X[trn_tst_split:]
Y_test = Y[trn_tst_split:]



"""
    Design of Neural Network
"""
from sklearn.neural_network import MLPRegressor

mlp = MLPRegressor(hidden_layer_sizes=(8, 4), activation='tanh',
                   solver='sgd', batch_size=5, learning_rate='adaptive'
                   ,learning_rate_init=0.01, max_iter=300, shuffle=True,
                   tol=0.000000000001, verbose=True, momentum=0.9)
mlp.fit(X_train, Y_train)

# hidden_layer_sizes=(8, 4) تعداد لایه های مخفی را نشان میدهد که ما در اینجا میگوییم شبکه ی ما 100 تا نورون لایه اول داشته باشد و 20 تا نرن لایه دوم و در کل 2 لایه داشته باشد
# در MLPClassifier نیازی نیست که تعداد لایه های خروجی را مشخص کنیم. خودش با توجه به X , Y ترین متوجه میشود لایه خروجی نیاز به چند تا نورون دارد

# activation برای تعیین اکتیویشن فانکشن میباشد که توابع (relu, tanh, logistic سیگموید تک قطبی, )

# solver روش بهینه سازی میباشد و میتواند (adam, sgd) باشد

# مقدار batch_size را 5 تا سمپل میدهیم

# learning_rate میتواند (adaptive, constant) باشد
# اگر ادپتیو باشد با توجه به نتیجه ها مقدار بالا پایین میشود و اگر کانستنت باشد مقدار ثابت است

# learning_rate_init مفدار لرنینگ ریت است

# max_iter تعداد epacs ها را نشان میدهد

# shuffle داده ها را قرو قاطی میکند که ما در اینجا انجام داده بودیم خودمان

#  tol یعنی اگر بعد ار 10 ایپاکس مقدار لاسفانکشن از این مقدار کمتر نشد از ترین بیرون بیا و ترین شبکه را ادامه نده

# verbose اطلاعات ترین شبکه را برای ما چاپ میکند



train_pred = mlp.predict(X_train)
test_pred = mlp.predict(X_test)
all_pred = mlp.predict(X)


#%%


import matplotlib.pyplot as plt
plt.plot(Y, label= 'real')
plt.plot(all_pred, label='predict')
plt.legend()
plt.xlabel('day')
plt.ylabel('tempreture')
plt.show()


loss = mlp.loss_curve_
plt.plot(loss)
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()


from sklearn.metrics import r2_score
R2_train = r2_score(Y_train, train_pred)
R_train = np.sqrt(R2_train)

R2_test = r2_score(Y_test, test_pred)
R_test = np.sqrt(R2_test)
# R2 score معیاری برای دقت است و هرچقدر به یک نزدیکتر باشد بهتر است. بین -1 و 1 است ولی میتواند بین 0 و 1 هم باشد با یک سری توابع
