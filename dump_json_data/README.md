## 生成配對交易績效檔案

重點為`dump_json_data.py`裡的`pair_trading_stationary_reward_json_input()`此function，此function會對每個配對，產生每個action的績效，並存成json or pickle檔，以便訓練時直接直接讀檔使用，不用每次都須重新計算。接著介紹function內部的參數：

- `method`: 選擇使用何種方式計算配對交易績效，共有4種選擇
    - None：使用co-integration方式計算配對交易績效
    - 'soft_computing'：採用Fallahpour et al. (2016)提出的方法計算配對交易績效
    - 'eg_ols'：採用Kim and Kim (2019)使用的eg_ols來計算配對交易績效
    - 'eg_tls'：採用Kim and Kim (2019)使用的eg_tls來計算配對交易績效
- `reward_type`: 選擇使用何種reward function，共有2種選擇
    - None：直接使用配對交易的報酬當作reward
    - 'complexity_reward'：採用Kim and Kim (2019)提供的reward function
- `output_stock_price`: json檔裡面是否需要包含股價
    - True：即會output股價到json檔中
    - False：即否
- `dump_spread_plot`: 是否需要產生各組配對交易情形的圖表
    - True：會在指定資料夾裡面，生成jpg資料夾，並在裡面生成各組配對的交易情形
    - False：即否
- 以下參數為配對交易的基本設定，若非必要，並不需動到
    - Formation period的長度
        - `formation_time = 150`
    - Trading period的長度
        - `trade_time = 100`
    - 交易稅
        - `transaction_cost = 0.0015`
    - 交易成本門檻
        - `transaction_cost_threshold = 0.0015`
    - 單檔股票最大持有張數
        - `maxi= 5`
    - 可使用的資金量
        - `capital = 50000000`
    - adf test，見先前學長論文(昭憲，喬登)，False表示不使用 (並未驗證開啟此功能的程式碼是否正確)
        - `adf = False`
    - fore lag 5 ，見先前學長論文(昭憲，喬登)，False表示不使用 (並未驗證開啟此功能的程式碼是否正確)
        - `fore_lag5 = False`
    - 使用舊均值(取sample mean)or新均值(迭代得到，見喬登論文)，True表示使用新均值
        - `new_std = True`
    - 不同交易成本門檻的設計，0表示使用上(下)開倉門檻減去均值要大於交易成本門檻才會開倉，1表示使用spread減去均值要大於交易成本門檻才會開倉
        - `cost_gate_type = 0`

若參數都未更動，直接跑此function，即會得到`D:\ptdqn_data\Data\HsinHuaChang_reverse_random_300_new_std`類似名稱的資料夾。

在績效資料夾中，會產生：

- program：保存當下跑的程式
- agent_action_info.json：儲存action的資訊
- Pairs20XXXX.json：儲存每個月中，每個配對對於每個action的績效
- jpg：當`dump_spread_plot`為True，才會產生各配對的交易情況

## 輸出配對交易固定門檻績效

當產生績效資料後，可使用`get_json_data_performance.py`得到特定action的績效。此檔案裡面有兩個重點function，分別為：

- `get_optimal_action`：使用訓練資料，來找出績效最佳的action
- `print_specific_action_performance`：使用測試資料，並給定特定action來產生最終的績效