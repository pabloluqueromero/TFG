# pazzani_bsej

# scores = []
# path = "../input/iris-flower-dataset/IRIS.csv"
# df = pd.read_csv(path)
# y = df["species"]
# X = df.drop("species",axis=1)
# for i in range(24):
#     X_train,X_test,y_train,y_test = train_test_split(
#                                             X, y, 
#                                             test_size=test_size, 
#                                             random_state=i,
#                                             stratify=y)
#     clf_encoding.fit(X_train,y_train)
#     scores.append(clf_encoding.score(X_test,y_test))
    
# print(np.mean(scores))
# scores = []
# cv= None#LeaveOneOut()
# for i in range(24):
#     X_train,X_test,y_train,y_test = train_test_split(
#                                             X, y, 
#                                             test_size=test_size, 
#                                             random_state=i,
#                                             stratify=y)
#     transformer,features,model = PazzaniWrapperNB(strategy="FSSJ",verbose=0).search(X_train,y_train)
#     scores.append(model.score(transformer(X_test),y_test))
    
# print(np.mean(scores))