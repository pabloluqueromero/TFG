import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.naive_bayes import GaussianNB


class NaiveBayes:
    def __init__(self, attributes: list, class_to_predict: str):
        self.attributes = attributes
        self.class_to_predict = class_to_predict
        self.probability_tables = dict()

    def fit(self, X_train: np.ndarray, Y_train: np.ndarray):
        # Añadir checks
        data_df = pd.DataFrame(data=X_train,columns=attributes)
        data_df[self.class_to_predict]=Y_train
        data_df['P']=np.array(X_train.shape[0])
        joint_probability_distribution = data_df.groupby(attributes+[self.class_to_predict],as_index=False,sort=True).agg({'P': lambda x: np.size(x)/data_df.shape[0]})

        self.probability_tables[self.class_to_predict] = joint_probability_distribution.groupby([self.class_to_predict],as_index=False,sort=True).agg({'P': np.sum})
        #si no hay ningun caso entonces tengo que añadir manueealmente todos los casos y ponerle 0
        class_values = np.unique(Y_train)
        for attribute,cases in zip(self.attributes,X_train.T):
            sum_on_condition = joint_probability_distribution.groupby([self.class_to_predict,attribute],as_index=False,sort=True).agg({'P': np.sum})
            condition_prob = pd.merge(sum_on_condition,self.probability_tables[self.class_to_predict], on= [self.class_to_predict])
            condition_prob['P'] = condition_prob['P_x']/condition_prob['P_y']
            condition_prob.drop(['P_x','P_y'], axis='columns', inplace=True)
            
            #si algún caso suma 1 hayq eu añadir 0 para los otros
            #Esta parte se tiene que mejorar
            #Se podría crear primero la tabla y luego hacer el merge con el condition_prob.drop
            attribute_values = np.unique(cases,return_counts=False)
            if attribute_values.shape[0]*class_values.shape[0]!= condition_prob.shape[0]:
                cartesian = np.transpose([np.tile(attribute_values, len(class_values)), np.repeat(class_values, len(attribute_values))])
                temp_df=pd.DataFrame(data=cartesian,columns=[attribute,self.class_to_predict])
                temp_df['P']=np.zeros(temp_df.shape[0])
                condition_prob = pd.merge(temp_df,condition_prob, on= [attribute,self.class_to_predict],how='left')
                condition_prob['P'] = np.nan_to_num(condition_prob['P_x']+condition_prob['P_y'])
                condition_prob.drop(['P_x','P_y'], axis='columns', inplace=True)
            self.probability_tables[attribute]=condition_prob
            # print(condition_prob)

    def predict(self,X):          
        df =  self.probability_tables[self.class_to_predict]
        values_X = pd.DataFrame(data=X,columns=attributes)
        values_X['data_index'] = np.arange(X.shape[0])                                 #Insertamos el indice de la evidencia
        df = values_X.assign(key=1).merge(df.assign(key=1), on='key').drop('key', 1)  #Producto cartesiano
        # print(df)
        #Multiplicamos para todas las evidencias a la vez
        for attribute in self.attributes:                                  
            conditional_probability = self.probability_tables[attribute]
            aux= pd.merge(df,conditional_probability,on=[attribute,self.class_to_predict],how='left')
            # print('aux\n',aux)
            df['P']= aux['P_x']*aux['P_y']
            # print('df\n',df)

        #Ordenamos por P y eliminamos evidencias duplicadas (nos quedamos con el  último - mayor)
        
        df = df.sort_values('P', na_position = 'first' ).drop_duplicates(['data_index'],keep='last') 
        return df.sort_values('data_index')[self.class_to_predict].values.tolist()
            
    def score(self, X, y):
        return np.sum(self.predict(X)==y)/y.shape[0]


if __name__ == "__main__":
    attributes = [
        'X1',
        'X2'
    ]
    class_to_predict = 'C'

    data = np.array([
            [1,1,1],
            [1,1,1],
            [1,1,1],
            [1,1,1],
            [1,1,1],
            [1,1,1],
            [1,1,1],
            [1,1,1],
            [1,1,1],
            [1,1,1],
            [1,1,1],
            [1,1,1],
            [1,1,1],
            [1,1,1],
            [1,1,1],
            [1,1,1],
            [1,1,1],
            [1,1,1],
            [1,1,1],
            [1,1,1],
            [1,1,1],
            [1,1,1],
            [1,1,1],
            [1,1,1],
            [1,1,1],
            [1,1,1],
            [0,1,0],
            [1,0,1],
            [0,1,1],
            [0,0,0],
            [1,1,1],
            [0,1,0]
        ])



    nb_classifier = NaiveBayes(attributes, class_to_predict)

    x,y = np.hsplit(data,np.array([2]))
    nb_classifier.fit(x,y)
    print("--------------------Fitted----------------------")
    
    
    
    
    gnb = GaussianNB()
    gnb.fit(x,y)

    # print(all((np.equal(nb_classifier.predict(np.array([ejemplo])),gnb.predict(np.array([ejemplo]))) for ejemplo in x)))
 
    # print(gnb.predict(np.array([[1,1]])))
    print(gnb.score(x,y.T))
    print(nb_classifier.score(x,y.T))
