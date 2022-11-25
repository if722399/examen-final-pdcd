"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: Proyecto de aplicación IV                                                                                        -- #
# -- script: functions.py : python script with general functions                                         -- #
# -- author: José Tonatiuh Navarro Silva                                                                                  -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot as plt
import os
from typing import List

plt.style.use('ggplot')
seed = 28
path = '.'

def dqr(df):
    return pd.DataFrame(
    {
    '% of nulls':[round(df[i].isna().sum()/len(df),4) for i in df.columns],
    'unique_values':[df[i].nunique() for i in df.columns],
    'type':df.dtypes.tolist()
    },
    index=df.columns
    )


def cat_proportion(df,cat_df,target_v):
    categorical_temp = df[list(cat_df.columns) + [target_v]]
    a = pd.DataFrame()
    for predictor in cat_df.columns:
        b = categorical_temp.groupby(predictor).mean(target_v)
        a = pd.concat([a,b])
        display(b)
        print('\n')
    a['diff'] = np.abs(a-df['client_stayed'].mean())
    return a.sort_values('diff',ascending=False).iloc[0:10,:]



class encode:

    
    def __init__(self, df:pd.DataFrame, target:str):
        self.df = df
        self.target = target
        self.categoric_columns = df.select_dtypes(include=['object'])
        
    def woe_encoder(self,train):

        ''' 
        # --------- Ventajas --------- #
        - Signo y magnitud están directamente relacionados con el problema.
        - Es rápido de calcular.
        - Es directamente interpretable.
        - No generar columnas extras.



        # --------- Desventajas --------- #

        - Hay un ligero leak al incluir la respuesta de manera directa.
        - Se debe de tener mucho cuidado al calcularse con datos únicamente de entrenamiento.
        - Se deben de cuidar categorías "pequeñas" pues al existir una división, puede haber problemas por cantidades pequeñas.
        - Debe de tenerse cuidado con categorías que no tengan "ceros", pues pueden llegar a existir divisiones entre cero.
        '''
        woes = {}
        for col in self.categoric_columns.columns:
            #Primero obtenemos los 1s y los counts de las categorías individuales.
            primer_paso = train.groupby(col).agg({self.target:['sum','count']})[self.target]

            # La diferencia de estos dos nos dará la cantidad de la categoría de ceros 
            primer_paso.loc[:,'ceros'] = primer_paso['count'] - primer_paso['sum']

            # Obtenemos la proporción de la categoría de unos
            props_1s = primer_paso['sum'] / primer_paso['sum'].sum()

            # Obtenemos la proporción de la categoría de ceros
            props_0s = primer_paso['ceros'] / primer_paso['ceros'].sum()

            # Calculamos las proporciones finales
            props_finales = props_1s/props_0s

            # Sacamos el logarítmos y guardamos como último paso
            woes[col] = np.log(props_finales).to_dict()


        return woes
    
    def frequency_encoder(self,train):

        #fig,axes = plt.subplots(figsize=(14,8),nrows=1, ncols=len(self.categoric_columns.columns), sharey=True)

        results = {}
        for index, col in enumerate(self.categoric_columns.columns):
         #   ax = axes[index]

            # Agrupacion
            agrupacion = train.groupby(col).agg({self.target: ['mean','std','count']})[self.target]

            # Obtener proporciones
            agrupacion['prop'] = agrupacion['count'] / agrupacion['count'].sum()

            # Ordenar
            agrupacion = agrupacion.sort_values('prop')
            
            results[col] = agrupacion['prop'].to_dict()

        return results
    
    def mean_encoder(self,train):
        
        # df deber ser un set de entrenamiento que tenga la variable de respuesta
        mean_encoder_dict = {}

        for col in self.categoric_columns.columns:
            mean_encoder_dict[col] = train.groupby(col)[self.target].mean().to_dict()
        return mean_encoder_dict


class get_model(encode):
    
    def __init__(self, df:pd.DataFrame, target: str, encoding_type: str):
        super().__init__(df,target)
        self.df = df
        self.target = target
        self.encoding_type = encoding_type
        self.seed = 17
        self.predictores = [c for c in df.columns if c!=target]
        
        
    def split_set_with_encoding(self):
        
        predictores = [c for c in self.df.columns if c!=self.target]
        train_x,test_x,train_y,test_y = train_test_split(self.df[predictores],
                                                        self.df[self.target],
                                                        test_size=.2,
                                                        random_state=self.seed)
        
    # Generamos un set de entrenamiento que tenga la variable de respuesta
        train = pd.merge(train_x, train_y, left_index=True, right_index=True)

        if self.encoding_type == 'WOE':
            encoder_dict = self.woe_encoder(train)
        elif self.encoding_type == 'Frecuency':
            encoder_dict = self.frequency_encoder(train)
        elif self.encoding_type == 'Mean':
            encoder_dict = self.mean_encoder(train)
        else:
            raise ValueError('Enter a valid encoding type')

        train_x.replace(encoder_dict, inplace=True)
        test_x.replace(encoder_dict, inplace=True)
    
    
        
        return {'train_x':train_x, 'test_x':test_x, 'train_y':train_y, 'test_y':test_y} 
    
    def create_model(self):
        
        model = LogisticRegression()
        model.fit(self.split_set_with_encoding()['train_x'][self.predictores], self.split_set_with_encoding()['train_y'])
        
        train_scores = model.predict_proba(self.split_set_with_encoding()['train_x'][self.predictores])[:,1]
        test_scores = model.predict_proba(self.split_set_with_encoding()['test_x'][self.predictores])[:,1]
        
        return {'train_scores':train_scores, 'test_scores':test_scores}
    
    def get_performance(self):
        
        train_auc = roc_auc_score(y_true=self.split_set_with_encoding()['train_y'], y_score=self.create_model()['train_scores'])
        test_auc = roc_auc_score(y_true=self.split_set_with_encoding()['test_y'], y_score=self.create_model()['test_scores'])
        
        return {'train_auc':train_auc, 'test_auc':test_auc}