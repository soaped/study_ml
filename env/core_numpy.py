# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 1.0.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import numpy as np
my_array = np.array([1,2,3,4,5])
print(my_array)
#shape 指定长度，可以是二维， dtype 默认为float64,order存储顺序C按行存，F按列存 默认C
zero_array = np.zeros((5,2),dtype=int, order='F')
print(zero_array)

np.ones(3)

#生成随机数0~1之间
random_arr = np.random.random(5)
print(random_arr)


# -
my_array = np.array([[1,5],[4,6]])
#数组截取
col2 = my_array[:,1]
print(col2)


#数组运算
a = np.array([[1,2],[2.1,4]])
b = np.array([[3.2,2],[1.1,3]])
sum = a + b
print(sum)
diff = a-b
print(diff)
#不是矩阵相乘，而是将所有元素相乘
product= a*b
print("product = \n",product)
matrix_product = a.dot(b)
print("matrix_product=",matrix_product)

#定义多维数组
a = np.array([[11, 12, 13, 14, 15],
              [16, 17, 18, 19, 20],
              [21, 22, 23, 24, 25],
              [26, 27, 28 ,29, 30],
              [31, 32, 33, 34, 35]])
# >>>25
print(a[2,4]) 


