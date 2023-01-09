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
@st.cache
def Fuel_info(x):
    a={0: "a", 1: "b",2: 'c'}
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
Present_Price = st.number_input("车子目前市场价值?",value=3000,step=50)
Kms_Driven = st.number_input("车子开了多少公里了?",value=5000,step=100)
Fuel_Type = st.radio("油的型号?",(0, 1, 2),format_func=lambda x: Fuel_info(x))
Seller_Type = st.text_input("What is the type of seller?","Please Type 0 for Dealer/ 1 for Individual")
Transmission = st.text_input("What is the type of Transmission?","Please type 0 for Automatic/ 1 for manual")
Owner = st.text_input("What is the no. of owners?", "Please type 0/1/3")
no_year = st.number_input("How many years old?",value=10,step=1)
if st.button("Predict"):
    output=predict_price(Present_Price, Kms_Driven,Fuel_Type, Seller_Type,Transmission, Owner, no_year)
    st.success('这辆车的售价预估为{} 元'.format(round(output, 2)))
