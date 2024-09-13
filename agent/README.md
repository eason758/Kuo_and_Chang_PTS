## dqn_agent

整個流程為：

1. 若沒有給定action，則讀取 training data 並使用 HighFreq 的方式來找尋最佳25個action
2. 讀取績效資料檔案
3. 使用 training data 訓練 DQN
4. 使用 validation data 找尋勝率最高的epoch參數

以下為一些參數介紹：

- 配對交易基本設定：
    - `FORMATION_PERIOD`：formation period 長度，預設為150
    - `TRADING_PERIOD`：trading period 長度，預設為100
    - `COST_GATE`：交易成本門檻，預設為0.0015
    - `TAX_COST`：交易稅，預設為0.0015
- 配對交易績效資料讀取：
    - `TRAIN_MONTHS`：training data 的長度 (單位：月)
    - `VALIDATE_MONTHS`：validation data 的長度 (單位：月)
    - `TEST_YEAR`：testing data 從哪一年開始
    - `TEST_MONTH`：testing data 從哪個月開始
    - `TEST_MONTHS`：testing data 的長度 (單位：月)
    - `VALIDATE_YEAR`：validation data 從哪一年開始
    - `VALIDATE_MONTH`：validation data 從哪一個月開始
    - `SPECIFIC_MODEL_TYPE`：是否需要只讀取特定cluster的配對，若需要則直接給定 cluster index (為了與Johansen test對齊，此index從1開始)，若否則給定0
    - `MODEL_TYPE_JSON_FILE`：research/clustering.py 產生的檔案名稱。當`SPECIFIC_MODEL_TYPE`不為0且`MODEL_TYPE_JSON_FILE`為`None`，則使用Johansen test來進行clustering
    - `STOCK1`：所需要配對中的stock 1，格式可為 str 或 list，若要 load 所有配對則使用`None`
    - `STOCK2`：所需要配對中的stock 2，格式可為 str 或 list，若要 load 所有配對則使用`None`
      - 目前`STOCK1`與`STOCK2`由`../stock_info/`裡面的檔案來給定
    - `SOURCE_TRAINED_MODEL_FOLDER` and `DATA_FOLDER`：
        - 當 `DATA_FOLDER == None`：
            - 當 `SOURCE_TRAINED_MODEL_FOLDER != None`：
                - 直接使用`SOURCE_TRAINED_MODEL_FOLDER`此模型資料夾的action與`DATA_FOLDER`，不需再重新計算最佳的action
            - 當 `SOURCE_TRAINED_MODEL_FOLDER == None`：
                - 讀取 training data 並使用 HighFreq 的方式來找尋最佳25個action
        - 當 `DATA_FOLDER != None`：
            - 使用此績效資料夾內部的action
- State 設定：

  定義 DQN agent 的 state 為何，應用在`ptdqn_env.py` 中的`input_s()`

    - `STOCK_PRICE_RETURN`：state 是否需要加入兩檔股票的 return
    - `ONLY_STOCK_PRICE`：state 是否**只**需要兩檔股票價格
    - `ONLY_STOCK_PRICE_LN_RETURN`：state 是否**只**需要取 ln 的兩檔股票報酬率
    - `STOCK_PRICE_LN_RETURN`：state 是否需要加入取 ln 的 return
    - `STATE_LENGTH`：state 的長度
    - `STANDARDIZATION`：state input 需不需要進行標準化
    - `ONLY_TWSE`
    - `INDEX_DATA`
    - `STOCK_DAILY_RETURN`
- 模型設定：
    - `RESNET`：是否需要使用 ResNet，若否則使用simple DNN
    - `PRE_TRAINED_MODEL_FOLDER`：預訓練模型的資料夾

## trained_dqn_agent

當模型訓練好，可直接使用以下指令來看 testing 的績效

```
python trained_dqn_agent.py MODEL_FOLDER_NAME
```

績效的輸出格式為：

```
Profit
Win Rate/Normal Close Rate
(Normal close number, Stop loss close number, Force close number, Total number)
Sharpe ratio (daily-based)
Sharpe ratio (pair-based)
Profit of per open (thousand)/Trade capital (thousand)/MDD
```

如同前面所述，若程式發現到所需的績效資料夾不存在，則會使用以下兩個資料夾進行代替
* `HsinHuaChang_reverse_random_300_new_std_pickle`
* `HsinHuaChang_reverse_random_300_trend_stationary_new_std_pickle`

目前會先判斷模型資料夾裡是否已存在之前跑完的結果
* 如果已存在，則直接使用此csv檔輸出最終績效
* 若不存在，則會去讀取資料來針對各個testing pair選取其開倉停損門檻
* 如果需要整個測試流程重跑一遍，要注意到目前程式並不會直接蓋過原始的績效csv檔，而會**建立一個新的檔案**(詳情請見`dump_testing_result_csv()`此function)，因此程式碼需改成：

```python
if __name__ == '__main__':
    # if not os.path.isfile(f'{FOLDER}/Test/testing_result.csv'):
    main()
    print(TRAINED_FOLDER)
    print(f'COST_GATE:{COST_GATE}')
    result_df = pd.read_csv(f'{FOLDER}/Test/testing_result-new.csv')
    dump_final_result(result_df)
```