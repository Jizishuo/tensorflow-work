import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#创建一个Series通过传递值的列表，让pandas创建一个默认的整数索引：
s = pd.Series([1,3,5,np.nan,6,8])
print(s)
print("_________________________________________")
#DataFrame通过传递带有日期时间索引和标记列的NumPy数组来创建
datas = pd.date_range("20180801", periods=6)
print(datas)
print("_____________________________")
df = pd.DataFrame(np.random.rand(6, 4), index=datas, columns=list("ABCD"))
print(df)
print("_________________________________________")
#DataFrame通过传递可以转换为类似系列的对象的dict来创建
df2 = pd.DataFrame({"A": 1.,
                    "B": pd.Timestamp("20180801"),
                    "C": pd.Series(1,index=list(range(4)), dtype='float32'),
                    'D': np.array([3] * 4, dtype='int32'),
                    'E': pd.Categorical(["test", "train", "test", "train"]),
                    'F': 'foo'})
print(df2)
print(df2.dtypes)
print("________________________________________________")

#以下是查看框架的顶行和底行的方法：
#查看数据
print(df.head())
print(df.tail(3))
print("________________________________________________")

#显示索引，列和基础NumPy数据：
print(df.index)
print(df.columns)
print(df.values)
print("________________________________________________")

#describe() 显示您的数据的快速统计摘要：
print(df.describe())
#转置您的数据：
print(df.T)
print("________________________________________________")

#按轴排序：
print(df.sort_index(axis=1,ascending=False))
print("________________________________________________")
#按值排序
print(df.sort_values(by="B"))

print("___________________选择_____________________________")
#选择
print(df["A"])
print(df[0:3])
print(df['20180802':'20180804'])
print("________________________________________________")

#使用标签获取横截面：
print(df.loc[datas[0]])
print(df.loc[:, ["A","B"]])
#显示标签切片，两个端点包括：
print(df.loc["20180802":"20180804", ["A", "B"]])
#减少返回对象的尺寸：
print(df.loc["20180802", ["A","B"]])
print("________________________________________________")

#获取标量值：
print(df.loc[datas[0], "A"])
#为了快速访问标量（相当于以前的方法）：
print(df.at[datas[0], "A"])
print("________________________________________________")

#按位置选择
print(df.iloc[3])
#通过整数切片，类似于numpy / python：
print(df.iloc[3:5, 0:2])
#通过整数位置位置列表，类似于numpy / python样式：
print(df.iloc[[1,2,4], [0,2]])
#对于明确切片行：
print(df.iloc[1:3, :])
#对于明确切片列：
print(df.iloc[:, 1:3])
#为了明确获取值：
print(df.iloc[1,1]) #df.iat[1,1]
print("________________________________________________")

#布尔引索
print(df[df.A > 0])
print(df[df > 0])
print("_______________________过滤方法_________________________")
#使用isin()过滤方法：
df2 = df.copy()
df2["E"] = ['one', 'one','two','three','four','three']
print(df2)
print(df2[df2["E"].isin(['two', 'four'])])
print("________________________________________________")

#设定
#设置新列会自动根据索引对齐数据
s1 = pd.Series([1,2,3,4,5,6], index=pd.date_range("20180802", periods=6))
print(s1)
