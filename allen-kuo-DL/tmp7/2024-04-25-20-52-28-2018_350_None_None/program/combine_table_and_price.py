import pandas as pd
import os
import numpy as np
import pickle
save_dir = r"C:\Users\eason\Desktop\PTDQN-main\Data\HsinHuaChang_reverse_random_300_new_std_price_pickle\\"
data_root = r'C:\Users\eason\Desktop\PTDQN-main\Data\HsinHuaChang_reverse_random_300_new_std_pickle\\'
stock_price_root = r'C:\Users\eason\Desktop\PT_code\stock_data\min_data\\'

for f in os.listdir(data_root):
    if "Pairs2015" not in f and "Pairs2016" not in f and "Pairs2017" not in f and "Pairs2018" not in f and \
        "Pairs2019" not in f :
        continue
    df = pd.DataFrame(pickle.load(open(data_root +f, 'rb')))
    print(f)
    stock1_price = []
    stock2_price = []
    for i in range(len(df)):
        if i%1000 == 0:
            print(f"  {i}/{len(df)}")
        date = str(df.iloc[i].date)
        s1 = df.iloc[i].stock1
        s2 = df.iloc[i].stock2
        y = date[:4]
        m = date[4:6]
        d = date[6:8]
        stock_path = stock_price_root+rf"{y}\minprice\\{date}_min_stock.csv"
        stocks = pd.read_csv(stock_path)
        p1 = np.array(stocks[str(s1)])
        p2 = np.array(stocks[str(s2)])
        stock1_price.append(p1)
        stock2_price.append(p2)
    df["stock1_price"] = stock1_price
    df["stock2_price"] = stock2_price
    df.to_pickle(save_dir+f)