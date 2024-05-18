import numpy as np
import glob as gb
import cv2 

"""
    ---- Preparing data ---- 
"""

image_paths = gb.glob(r'C:/Users/HP/OneDrive - Teesside University/Uni/Semester 2/Neural Network/Faradars/New folder/FVPHT9810-S02/Files and Codes/Hoda 0-9/' + '*.bmp')
# بجای \ باید / گزاشته شود

#%%
X = []
Y = []
for i in range(len(image_paths)):
    path = image_paths[i]
    idx = path.rfind('_') - 1
    Y.append(int(path[idx]))
    img = cv2.imread(path, 0) # این دستور تصویر را میخواند. و 0 هم تصویر تک کاناله ی خاکستری را به ما برمیگرداند
    #cv2.imshow('image',img)
    #ch = cv2.waitKey(0)   #0 میزان زمانیست که منتظر کاربر میماند تا کلیک کند. اگر 1000 باسد همینقدر میلیثانیه صبر میکند بعد میرود بعدی
    #if ch == ord('q'):
    #    break
    img = cv2.resize(img, dsize = (10,10))
    ftrs = np.reshape(img, newshape=(100,))
    X.append(ftrs) 
#cv2.destroyAllWindows()

X = np.array(X)
Y = np.array(Y)

# Shuffle data
list_per = np.random.permutation(len(X))
X_per = []
Y_per = []
for i in range(len(list_per)):
    idx_per = list_per[i]
    ftr = X[idx_per]
    lbl = Y[idx_per]
    X_per.append(ftr)
    Y_per.append(lbl)
X_per = np.array(X_per)
Y_per = np.array(Y_per)


# Splitting data to train and validation
trn_tst_split = int(0.7 * len(X_per))
X_train = X_per[0:trn_tst_split, :]
Y_train = Y_per[0:trn_tst_split]

X_val = X_per[trn_tst_split:, :]
Y_val = Y_per[trn_tst_split: ]


# Normalize data
#در اینجا چون تعداداعداد در هر پیگسل بین 0 تا 255 است بنابر این برای نرمال کردن داده ها آنهارا بر 255 تقسیم میکنیم
X_train = X_train/255
X_val = X_val / 255
 

"""
    ---- Design of Neural network ----
"""
from sklearn.neural_network import MLPClassifier # MLPRegressor هم برای داده های خطی میباشد

# create neural network
mlp = MLPClassifier(hidden_layer_sizes=(50, 20), activation='logistic',
                    solver='adam', batch_size=50, learning_rate='adaptive',
                    learning_rate_init=0.001, max_iter=200, shuffle=True,
                    tol=0.0000001, verbose=True, momentum=0.95)
# hidden_layer_sizes=(100, 20) تعداد لایه های مخفی را نشان میدهد که ما در اینجا میگوییم شبکه ی ما 100 تا نورون لایه اول داشته باشد و 20 تا نرن لایه دوم و در کل 2 لایه داشته باشد
# در MLPClassifier نیازی نیست که تعداد لایه های خروجی را مشخص کنیم. خودش با توجه به X , Y ترین متوجه میشود لایه خروجی نیاز به چند تا نورون دارد

# activation برای تعیین اکتیویشن فانکشن میباشد که توابع (relu, tanh, logistic سیگموید تک قطبی, )

# solver روش بهینه سازی میباشد و میتواند (adam, sgd) باشد

# learning_rate میتواند (adaptive, constant) باشد 

# max_iter تعداد epacs ها را نشان میدهد

# shuffle داده ها را قرو قاطی میکند که ما در اینجا انجام داده بودیم خودمان

#  tol یعنی اگر بعد ار 10 ایپاکس مقدار لاسفانکشن از این مقدار کمتر نشد از ترین بیرون بیا و ترین شبکه را ادامه نده

# verbose اطلاعات ترین شبکه را برای ما چاپ میکند

# train neural network
mlp.fit(X_train, Y_train)


print('train accuracy : ', mlp.score(X_train, Y_train))
print('val accuracy : ', mlp.score(X_val, Y_val))
# نکته
# در اینجا ممکن است داده های ترین 100% یا 99.9% دقت برسد که اورفیت میشود. برای جولوگیری از اورفیت شدن باید با پارامتر ها بازی کنیم تا از اور فیت بودن دراوریم و همجنین دقت بالایی داشته باشیم
#

# برای بدست آوردن لاست فانکشن در ایپاک های مخطلف از mlp.loss_curve_ استفاده میکنیم
# لاس فانکشن اخطلاف بین نتیجه پردیکت و لیبل واقعی ایت
#یک جورایی همان مین اسکور اررور است ولی با فرمول های دیگر

loss = mlp.loss_curve_

import matplotlib.pyplot as plt
plt.plot(loss, label= 'loss-Train')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()


#confusion_matrix : 
# در اینجا نتیجه ی واقعی و پیشبینی را با هم مقایسه میکنیم و کانفیوژن ماتریکسش را بدست می آوریم
from sklearn.metrics import confusion_matrix as cm 
Y_val_pred = mlp.predict(X_val)
mlp_cm = cm(Y_val , Y_val_pred)

#%%
# در اینجا F1 Recall  و ... را به ما میدهد
from sklearn.metrics import classification_report
mlp_report = classification_report(Y_val , Y_val_pred)



# برای این که مدل را دخیره کنیم از کد زیر استفاده میکنیم
#مثلا اگر خواستیم بر روی سخت افزاری یا جای دیگری از این مدل استفاده کنیم کافیست فایل را فراخوانی کنیم و آن را ران کنیم
import joblib 
joblib.dump(mlp , 'mlp-network.joblib')
new_model = joblib.load('mlp-network.joblib')
Y_val_pred_newModel = new_model.predict(X_val)

# در اینجا مدل را دخیره کردیم و سپس لود کردیم و آنرا با داده های قبل خودمان تست کردیم و در مقایسه ی جواب های Y_val_pred_newModel و Y_val_pred میبینیم که مدل درست عمل میکند



