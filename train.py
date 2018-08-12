# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import pymc3 as pm
from scipy import stats
import json

# Jobの登録時に設定したhyperparameterの読み取り
with open('/opt/ml/input/config/hyperparameters.json') as f:
    hyperparameters = json.load(f)
sample_size = int(hyperparameters['sample_size'])

# 広告アイテムと広告枠の効果を考慮して推定
csv_file = f"/opt/ml/input/data/sagemaker_sample/train_data.csv"
train_data = pd.read_csv(csv_file)
ad_id_data = pd.Categorical(train_data['ad_id'].values).codes
adspot_id_data = pd.Categorical(train_data['adspot_id'].values).codes
is_clicked_data = train_data['is_clicked'].values

with pm.Model() as logistic_model:
    alpha = pm.Uniform('alpha', lower=-4, upper=1)
    ad_coefs = pm.Normal('ad_coefs', mu=0, sd=1, shape=len(set(train_data['ad_id'].values)))
    adspot_coefs = pm.Normal('adspot_coefs', mu=0, sd=1, shape=len(set(train_data['adspot_id'].values)))
    p = pm.Deterministic('p', pm.math.sigmoid(alpha + ad_coefs[ad_id_data] + adspot_coefs[adspot_id_data]))
    yl = pm.Bernoulli('yl', p=p, observed=is_clicked_data)
    trace = pm.sample(sample_size, chains=2)

summary = pm.summary(trace, varnames={'alpha', 'ad_coefs', 'adspot_coefs'})
summary.to_csv("/opt/ml/model/result.csv")
