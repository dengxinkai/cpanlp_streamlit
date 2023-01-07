import streamlit as st 
import pandas as pd

data = {'name':['Tom', 'nick', 'krish', 'jack','Tom'],
        'nickname':['jack','krish','karim','joe','joe'],
        'age':[20, 18, 19, 18,22]}
 
df = pd.DataFrame(data)
df_result_search = pd.DataFrame() 


searchcheckbox_name_nickname = st.checkbox("Name or Nickname ",value = False,key=1)
searchcheckbox_age = st.checkbox("age",value = False,key=2)

if searchcheckbox_name_nickname:
    name_search = st.text_input("name")
    nickname_search = st.text_input("nickname")
if searchcheckbox_age:   
    age_search = st.number_input("age",min_value=0)
if st.button("search"):
    df_result_search = df[df['name'].str.contains(name_search,case=False, na=False)]
    df_result_search = df[df['nickname'].str.contains(nickname_search,case=False, na=False)]
    
    df_result_search = df[df['age']==(age_search)]
                    
    st.write("{} Records ".format(str(df_result_search.shape[0])))
    st.dataframe(df_result_search)
st.write(df)
