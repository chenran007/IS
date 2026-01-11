#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt


# In[3]:


model = joblib.load('xgb3.pkl')


# In[5]:


X_test = pd.read_csv('X_test_IS.csv')


# In[7]:


#定义特征名称，对应数据集中的列名
feature_names = ["QDC", "QSC", "Age", "Gender", "WHtR", "Saltytaste", "Depression","Socialparticipation", "CHD", "Hypertension","Diabetes","Hyperlipidemia"]


# In[11]:


#Streamlit 用户界面
st.title("IS Risk Prediction")
QDC = st.selectbox("Qi-deficiency constitution (QDC):", options=[0, 1])
QSC = st.selectbox("qi stagnation constitution (QSC):", options=[0, 1])
Age = st.selectbox("Age:", options=[0, 1,2])
Gender = st.selectbox("Gender:", options=[0, 1])
WHtR = st.selectbox("WHtR:", options=[0, 1,2])
Saltytaste = st.selectbox("Saltytaste:", options=[0, 1])
Depression = st.selectbox("Depression:", options=[0, 1])
Socialparticipation = st.selectbox("Socialparticipation:", options=[0, 1])
Hyperlipidemia = st.selectbox("Hyperlipidemia:", options=[0, 1])
Hypertension = st.selectbox("Hypertension:", options=[0, 1])
CHD = st.selectbox("Coronary heart disease (CHD):", options=[0, 1])
Diabetes = st.selectbox("Diabetes:", options=[0, 1])


# In[13]:


# 实现输入数据并进行预测
feature_values = [QDC, QDC, Age, Gender, WHtR, Saltytaste, Depression, Socialparticipation, Hyperlipidemia, Hypertension, CHD, Diabetes]  # 将用户输入的特征值存入列表
features = np.array([feature_values])  # 将特征转换为 NumPy 数组，适用于模型输入
# 当用户点击 "Predict" 按钮时执行以下代码
if st.button("Predict"):
    # 预测类别（0: 无脑梗塞，1: 有脑梗塞）
    predicted_class = model.predict(features)[0]
    # 预测类别的概率
    predicted_proba = model.predict_proba(features)[0]

    # 创建 SHAP 解释器，基于树模型（如随机森林）
    explainer_shap = shap.TreeExplainer(model)
    # 计算 SHAP 值，用于解释模型的预测
    shap_values = explainer_shap.shap_values(pd.DataFrame([feature_values], columns=feature_names))

    # 显示预测结果
    st.write(f"**Predicted Class:** {predicted_class} (1: Disease, 0: No Disease)")
    st.write(f"**Prediction Probabilities:** {predicted_proba}")

    # 根据预测结果生成建议
    # 如果预测类别为 1（高风险）
    if predicted_class==1:
        probability = predicted_proba[1] * 100
        advice = (
            f"According to our model, you have a high risk of IS. "
            f"The model predicts that your probability of having IS is {probability:.1f}%. "
            "It's advised to consult with your healthcare provider for further evaluation and possible intervention."
        )
    # 如果预测类别为 0（低风险）
    else:
        probability = predicted_proba[0] * 100
        advice = (
            f"According to our model, you have a low risk of IS. "
            f"The model predicts that your probability of not having IS is {probability:.1f}%. "
            "However, maintaining a healthy lifestyle is important. Please continue regular check-ups with your healthcare provider."
        )
    st.write(advice)
    # SHAP 解释
    st.subheader("SHAP Force Plot Explanation")
    
    # 根据预测类别显示 SHAP 强制图
    # 期望值（基线值）
    # 解释类别 1（患病）的 SHAP 值
    # 特征值数据
    # 使用 Matplotlib 绘图
    shap.force_plot(explainer_shap.expected_value, shap_values, pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)
    # 期望值（基线值）
    # 解释类别 0（未患病）的 SHAP 值
    # 特征值数据
    # 使用 Matplotlib 绘图 
    #plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.pyplot(plt.gcf(), use_container_width=True)


# In[ ]:




