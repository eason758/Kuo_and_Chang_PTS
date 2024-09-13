import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

'''******************************* Do Your control below *******************************'''

test_result_csv = r"C:\Users\eason\Desktop\PTDQN-main\allen-kuo-DL\tmp7\2024-08-07-18-49-25-2015_350_None_None\Test\testing_result.csv"

'''******************************* Do Your control above *******************************'''


df = pd.read_csv(test_result_csv)

record = df["record"]
profit = df["half_tax_profit"]
capital = df["capital"]

open_count = (record!=0).sum()
WR = (profit>0).sum()/open_count
LR = (profit<0).sum()/open_count
NCR = (record==666).sum()/open_count
EXR = (record==-4).sum()/open_count
SLR = (record==-2).sum()/open_count
total_profit = profit.sum()
ppo = total_profit/open_count

print("open count ",open_count)
print("win rate ",WR)
print("loss rate ",LR)
print("Normal Close Rate",NCR)
print("Exit Rate",EXR)
print("Stop Loss Rate",SLR)
print("total profit",total_profit)
print("profit per open",ppo)
print("open count",open_count)

df = df.groupby("date").sum()
cumprof = df["half_tax_profit"].cumsum()

plt.plot(range(len(cumprof)), cumprof)
plt.show()

df["return"] = df["half_tax_profit"]/df["capital"]

# calculate MDD
max_profit = -10000000
min_profit = 10000000
MDD = 0
for p in cumprof:
    if p > max_profit:
        max_profit = p
        min_profit = 10000000
    elif p < min_profit:
        min_profit = p
    if min_profit - max_profit < MDD:
        MDD = min_profit - max_profit
MDD_percent = 0
for i in range(len(cumprof)):
    for j in range(i):
        if (cumprof.iloc[j] - (cumprof).iloc[i])/(cumprof).iloc[j] > MDD_percent:
            MDD_percent = ((cumprof).iloc[j] - (cumprof).iloc[i])/(cumprof).iloc[j]
print("MDD", MDD)
print("MDD percent", MDD_percent)

negative_returns = np.array([ret for ret in df["return"] if ret < 0])
print("Sharpe ratio:",np.mean(df["return"]) / np.std(df["return"]) * np.sqrt(252))
print("Sortino ratio:",np.mean(df["return"]) / np.sqrt(np.sum(np.square(negative_returns))/len(df["return"])) * np.sqrt(252))
