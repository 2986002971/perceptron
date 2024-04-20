import streamlit as st
import numpy as np
from matplotlib import pyplot as plt

import perceptron

# 在终端输入 streamlit run ./main.py 启动
st.title("感知机实验可视化")

# 初始化或获取session_state中的状态
if 'data_generated' not in st.session_state:
    st.session_state.data_generated = False
if 'data' not in st.session_state:
    st.session_state.data = None
if 'center1' not in st.session_state:
    st.session_state.center1 = None
if 'center2' not in st.session_state:
    st.session_state.center2 = None

with st.sidebar:
    st.markdown("## 模型参数设置")
    learning_rate = st.slider("学习率", 0.01, 1.0, 0.01, 0.01)
    n_iters = st.slider("迭代次数", 1, 1000, 100, 100)

    st.markdown("## 数据设置")
    n_samples = st.number_input("训练数据量", 10, 10000, 100, 10)
    n_inference = st.number_input("测试数据量", 1, 1000, 10, 1)
    n_features = st.number_input("特征维度数", 2, 10000, value=2, step=1)
    position = st.number_input("两组数据中心距离调节", 1, 10, 1, 1)
    sigma = st.number_input("方差", 0.1, 2.0, 0.5, 0.1)

    generate_data = st.button("生成数据")
    if generate_data:
        data = perceptron.DataSet(dimension=n_features)
        center1 = np.zeros(n_features)
        center2 = np.zeros(n_features)
        for i in range(n_features):
            if np.random.random() < 0.5:
                center1[i] = position
            else:
                center2[i] = position

        data.gaussian_generate(center1, center2, sigma, n_samples)
        # 保存生成的数据和状态
        st.session_state.data = data
        st.session_state.data_generated = True
        st.session_state.center1 = center1
        st.session_state.center2 = center2

# 如果数据已经生成，显示数据的散点图
if st.session_state.data_generated and n_features == 2:
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.scatter(st.session_state.data.x[:, 0], st.session_state.data.x[:, 1], c=st.session_state.data.y.flatten(), cmap='coolwarm')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('DataSet')
    st.pyplot(fig)

    start_train_plot = st.button("开始训练")

    if start_train_plot:
        model = perceptron.Perceptron(learning_rate=learning_rate, n_iters=n_iters)
        model.fit(st.session_state.data.x, st.session_state.data.y)

        st.session_state.model = model

        weights = model.weights
        bias = model.bias

        slope = -weights[0] / weights[1]
        intercept = -bias / weights[1]

        x_line = np.linspace(np.min(st.session_state.data.x[:, 0]), np.max(st.session_state.data.x[:, 0]), 100)
        y_line = slope * x_line + intercept

        fig, ax = plt.subplots(figsize=(8, 6))
        plt.scatter(st.session_state.data.x[:, 0], st.session_state.data.x[:, 1], c=st.session_state.data.y, cmap='coolwarm')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('Classification')
        plt.plot(x_line, y_line, color='black', linestyle='--')

        st.pyplot(fig)

if st.session_state.data_generated:
    test = st.button('开始测试')
    if test:
        data_test = perceptron.DataSet(dimension=n_features)
        data_test.gaussian_generate(st.session_state.center1, st.session_state.center2, sigma, n_inference)

        model = perceptron.Perceptron(learning_rate=learning_rate, n_iters=n_iters)
        model.fit(st.session_state.data.x, st.session_state.data.y)
        predictions = model.predict(data_test.x)
        accuracy = np.sum(predictions == data_test.y) / n_inference
        st.write(f"测试准确率为: {accuracy}")

st.write("感谢streamlit提供的可视化工具！")
