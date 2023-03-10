import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
st.set_page_config(
    page_title="cpanlp的机器学习",
    page_icon="🐱",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.cpanlp.com/',
        'Report a bug': "https://www.cpanlp.com/",
        'About': "很高兴您使用cpanlp的机器学习项目"
    }
)
html_temp = """
    <div style="background-color:#000080 ;padding:2ßpx">
    <a href="https://cpanlp.com/example/">
    <h4 style="color:white;text-align:center;">
    车辆价格估算 </h4>
    </a>
    </div>
    """
st.markdown(html_temp, unsafe_allow_html=True)
#训练模型
# df = pd.read_csv("/Users/dengxinkaiacca163.com/Desktop/语言学理论/react/streamlit/app5/cardata.csv")
# final_dataset=df[['Year','Selling_Price','Present_Price','Kms_Driven','Fuel_Type','Seller_Type','Transmission',
# 'Owner']]
# final_dataset['Current Year']=2020
# final_dataset['no_year']=final_dataset['Current Year'] - final_dataset['Year']
# final_dataset.drop(['Year'],axis=1,inplace=True)
# le = LabelEncoder()
# final = final_dataset[['Fuel_Type', 'Seller_Type','Transmission']].apply(le.fit_transform)
# final_dataset.drop(['Fuel_Type', 'Seller_Type', 'Transmission'], inplace=True,axis=1)
# Fuel = final['Fuel_Type']
# Seller = final['Seller_Type']
# Transmission = final['Transmission']
# final_dataset=final_dataset.join(Fuel)
# final_dataset=final_dataset.join(Seller)
# final_dataset=final_dataset.join(Transmission)
# final_dataset=final_dataset.drop(['Current Year'],axis=1)
# X=final_dataset.iloc[:,1:]
# y=final_dataset.iloc[:,0]
# X_train, X_test, y_train, y_test = train_test_split(X, 
# y, test_size=0.3, random_state=0)
# n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
# max_features = ['auto', 'sqrt']
# max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
# min_samples_split = [2, 5, 10, 15, 100]
# min_samples_leaf = [1, 2, 5, 10]
# random_grid = {'n_estimators': n_estimators,
            #    'max_features': max_features,
            #    'max_depth': max_depth,
            #    'min_samples_split': min_samples_split,
            #    'min_samples_leaf': min_samples_leaf}
# rf = RandomForestRegressor()
# rf_random = RandomizedSearchCV(estimator = rf, 
                            #    param_distributions = random_grid,
                            #    scoring='neg_mean_squared_error', 
                            #    n_iter = 10, 
                            #    cv = 5,
                            #    verbose=2, 
                            #    random_state=42,
                            #    n_jobs = -1)
# rf_random.fit(X_train,y_train)
# predictions=rf_random.predict(X_test)
# # file = open('/Users/dengxinkaiacca163.com/Desktop/语言学理论/react/streamlit/app5/random_forest_regression_model.pkl', 'wb')
# pickle.dump(rf_random, file)
def Fuel_info(x):
    a={0: "97号油", 1: "汽油",2: '柴油'}
    return "选择了"+a[x]
def Seller_info(x):
    a={0: "个人", 1: "中介"}
    return "选择了"+a[x]
def Owner_info(x):
    a={1: "新手", 2: "2手",3:"3手"}
    return "选择了"+a[x]
def Transmission_info(x):
    a={0: "自动", 1: "手动"}
    return "选择了"+a[x]
def load_model():
    return pickle.load(open('./app5/random_forest_regression_model.pkl','rb'))
model = load_model()
def predict_price(Present_Price, Kms_Driven, Fuel_Type, Seller_Type,
                  Transmission, Owner, no_year):
    input=np.array([[Present_Price, Kms_Driven, Fuel_Type,
                     Seller_Type, Transmission, Owner,
                     no_year]]).astype(np.float64)
    prediction=model.predict(input)
    return float(prediction)
Present_Price = st.number_input("车子目前市场价值?(万）",min_value=0.0, max_value=200.0,value=3.0,step=0.5)
Kms_Driven = st.number_input("车子开了多少公里了?",min_value=0, max_value=100000,value=5000,step=500)
Fuel_Type = st.radio("油的型号?",[0, 1, 2],format_func=lambda x: Fuel_info(x))
Seller_Type = st.radio("销售商?",[0, 1],format_func=lambda x: Seller_info(x))
Transmission = st.radio("换挡方式?",[0, 1],format_func=lambda x: Transmission_info(x))
Owner = st.radio("几手?",[1, 2,3],format_func=lambda x: Owner_info(x))
No_year = st.number_input("How many years old?",min_value=0, max_value=15,value=5,step=1)
if st.button("Predict"):
    output=predict_price(Present_Price, Kms_Driven,Fuel_Type, Seller_Type,Transmission, Owner, No_year)
    st.success('这辆车的售价预估为{} 元'.format(round(output, 2)))
