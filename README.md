# PTDQN doc

## 資料與模型下載
- FTP (目前僅限交大校網可連)

```
IP: 140.113.87.72
username: 72-FTP-USER
password: 72-FTP-USER
```

- OneDrive (密碼：wcchang1019)

```
股票原始資料：https://nycu1-my.sharepoint.com/:f:/g/personal/wcchang_cs09_m365_nycu_edu_tw/ErdECYF-mfVKsyn_9YLnPFMBwvJxVlg9f9R8Eh_nCRk6iA

績效資料夾：https://nycu1-my.sharepoint.com/:f:/g/personal/wcchang_cs09_m365_nycu_edu_tw/ErLQnvXc7elIsoVntHsY5_wBG64GiAgk_lca8fzJqksYIg?e=7x2iFc

模型資料夾：https://nycu1-my.sharepoint.com/:f:/g/personal/wcchang_cs09_m365_nycu_edu_tw/EofwOV_fFWxLuDpfK79DMH0BRsJ7H6kyQcXVy-OxIJ6Vlg?e=z3rvJm
```

績效資料夾並不是每個都需要，優先下載下述兩個資料夾
* `HsinHuaChang_reverse_random_300_new_std_pickle`
* `HsinHuaChang_reverse_random_300_trend_stationary_new_std_pickle`

其餘資料夾都是此兩資料夾的子集合，單純只是加速後續訓練，使得程式不須讀入大量的無用資料

## 檔案位置
以下為目前預設的檔案路徑，可在`config.ini`進行更改

```
root_folder = D:/PTDQN # 本專案程式碼放置路徑
train_model_root_folder = D:/ptdqn_model # 模型資料夾生成路徑
data_root_folder = D:/ptdqn_data/Data # 績效資料夾路徑
```

## 環境設定
使用Anaconda來創建新環境
* 目前以使用CPU為主
* 如要使用GPU，則改用`requirements-gpu.txt`，並自行額外安裝CUDA

```
conda create --name TF_1.14_ENV python=3.7
activate TF_1.14_ENV
pip install -r requirements.txt
```

## 程式簡介
分成四大部分：
- dump_json_data：負責生成各個配對每組action的績效，並保存下來，以便訓練時不需重新計算
- clustering：使用unsupervised learning來將配對進行分類，並產生json檔以便後續訓練使用，主程式為`research/clustering.py`
- agent：使用`dqn_agent.py`進行訓練，每次訓練都會產生一個資料夾，內包含log檔、weight檔、reward圖與使用的程式等等相關的資料。接著即可使用`trained_dqn_agent.py`進行測試。
- 過往文獻實作：目前實作了過往三篇paper的作法
  1. `allen-kuo-DL`：實作郭偉倫 (2021)，把門檻選擇問題變成分類問題，並提出 two-stage model 來提升績效。第一階段先把各個配對進行 label 再訓練 multi-scale ResNet，而在第二階段時訓練另一個模型來預測此配對是會賺錢還是賠錢。
  2. `OPTICS`：實作 [S. M. Sarmento and N. Hort (2020)](https://doi.org/10.1016/j.eswa.2020.113490) ，先使用 PCA 將股價降維，再使用 OPTICS 進行分群，分在同群的配對才能進行交易。
  3. `n-arm-bandit`：實作 [Fallahpour et al (2016)](https://doi.org/10.1007/s00500-016-2298-4) ，將問題簡化成單一個state，訓練N-armed bandit選取開倉停損門檻

而`stock_info`此資料夾裡，每個檔案都包含出現次數前N多的配對(目前是使用`research/get_pair_count.py`來手動產出)，接著介紹命名規則：

* HsinHuaChang_TOP_N_STOCK1_IN_2014_2015.txt
* HsinHuaChang_TOP_N_STOCK2_IN_2014_2015.txt

上述兩個檔案表示2014~2015年中，**model 1 ~ model 3** 中出現次數前N的配對
* HsinHuaChang_with_trend_TOP_N_STOCK1_IN_2014_2015.txt
* HsinHuaChang_with_trend_TOP_N_STOCK2_IN_2014_2015.txt

上述兩個檔案表示2014~2015年中，**model 1 ~ model 5** 中出現次數前N的配對，其餘以此類推
## 論文績效對照
在每個資料夾內，都會有`result.txt`的檔案，裡面會記錄所有**Top n**配對的績效。
(當初輸出績效時有發現小bug，因此若資料夾內有`result-new.txt`的檔案，則以此檔案為準)
<table>
    <thead>
        <tr>
            <th>論文表格</th>
            <th>Type</th>
            <th>模型資料夾名稱</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=5>Table 4.1 ~ Table 4.2</td>
            <td>pair-based local learning</td>
            <td>20220314-P114C1、20220314-P118C1</td>
        </tr>
        <tr>
            <td>global learning</td>
            <td>20220314-P114C2、20220314-P118C2</td>
        </tr>
        <tr>
            <td>pair-based transfer learning</td>
            <td>20220314-P114C3、20220314-P118C3</td>
        </tr>
        <tr>
            <td>cluster-based local learning</td>
            <td>20220314-P114C4、20220314-P118C4</td>
        </tr>
        <tr>
            <td>cluster-based transfer learning</td>
            <td>20220314-P114C5、20220314-P118C5</td>
        </tr>
        <tr>
            <td rowspan=5>Table 4.3</td>
            <td>pair-based local learning</td>
            <td>20220314-P118C1</td>
        </tr>
        <tr>
            <td>global learning</td>
            <td>20220314-P118C2</td>
        </tr>
        <tr>
            <td>cluster-based local learning (Johansen test)</td>
            <td>20220314-P118C4</td>
        </tr>
        <tr>
            <td>cluster-based local learning (K-means)</td>
            <td>20220314-P118C24</td>
        </tr>
        <tr>
            <td>cluster-based local learning (GMM)</td>
            <td>20220314-P118C20</td>
        </tr>
        <tr>
            <td rowspan=6>Table 4.4</td>
            <td>pair-based local learning</td>
            <td>20220314-P118C1</td>
        </tr>
        <tr>
            <td>global learning</td>
            <td>20220314-P118C2</td>
        </tr>
        <tr>
            <td>cluster-based local learning (GMM 2 clusters)</td>
            <td>20220314-P118C22</td>
        </tr>
        <tr>
            <td>cluster-based local learning (GMM 3 clusters)</td>
            <td>20220314-P118C20</td>
        </tr>
        <tr>
            <td>cluster-based local learning (GMM 4 clusters)</td>
            <td>20220314-P118C18</td>
        </tr>
        <tr>
            <td>cluster-based local learning (GMM 5 clusters)</td>
            <td>20220314-P118C16</td>
        </tr>
        <tr>
            <td rowspan=3>Table 4.5</td>
            <td>pair-based local learning</td>
            <td>20220425-P149C1</td>
        </tr>
        <tr>
            <td>global learning</td>
            <td>20220425-P149C2</td>
        </tr>
        <tr>
            <td>pair-based transfer learning</td>
            <td>20220425-P149C3</td>
        </tr>
        <tr>
            <td rowspan=5>Table 4.6</td>
            <td>pair-based local learning</td>
            <td>20220425-P149C1</td>
        </tr>
        <tr>
            <td>global learning</td>
            <td>20220425-P149C2</td>
        </tr>
        <tr>
            <td>cluster-based local learning (Johansen test)</td>
            <td>20220425-P149C4</td>
        </tr>
        <tr>
            <td>cluster-based local learning (K-means) </td>
            <td>20220425-P149C16</td>
        </tr>
        <tr>
            <td>cluster-based local learning (GMM) </td>
            <td>20220425-P149C22</td>
        </tr>
        <tr>
            <td rowspan=6>Table 4.7</td>
            <td>pair-based local learning</td>
            <td>20220425-P149C1</td>
        </tr>
        <tr>
            <td>global learning</td>
            <td>20220425-P149C2</td>
        </tr>
        <tr>
            <td>cluster-based local learning (GMM 2 clusters)</td>
            <td>20220425-P149C24</td>
        </tr>
        <tr>
            <td>cluster-based local learning (GMM 3 clusters)</td>
            <td>20220425-P149C18</td>
        </tr>
        <tr>
            <td>cluster-based local learning (GMM 4 clusters) </td>
            <td>20220425-P149C20</td>
        </tr>
        <tr>
            <td>cluster-based local learning (GMM 5 clusters) </td>
            <td>20220425-P149C22</td>
        </tr>
        <tr>
            <td rowspan=3>Table 4.8 (2015~2016)</td>
            <td>pair-based local learning</td>
            <td>20220620-P53C1</td>
        </tr>
        <tr>
            <td>global learning</td>
            <td>20220620-P53C2</td>
        </tr>
        <tr>
            <td>pair-based transfer learning</td>
            <td>20220620-P53C3</td>
        </tr>
        <tr>
            <td rowspan=3>Table 4.8 (2016~2017)</td>
            <td>pair-based local learning</td>
            <td>20220606-P46C1</td>
        </tr>
        <tr>
            <td>global learning</td>
            <td>20220606-P46C2</td>
        </tr>
        <tr>
            <td>pair-based transfer learning</td>
            <td>20220606-P46C3</td>
        </tr>
        <tr>
            <td rowspan=6>Table 4.9</td>
            <td>pair-based local learning</td>
            <td>20220620-P53C1</td>
        </tr>
        <tr>
            <td>global learning</td>
            <td>20220620-P53C2</td>
        </tr>
        <tr>
            <td>cluster-based local learning (GMM 2 clusters)</td>
            <td>20220620-P53C4</td>
        </tr>
        <tr>
            <td>cluster-based local learning (GMM 3 clusters)</td>
            <td>20220620-P53C5</td>
        </tr>
        <tr>
            <td>cluster-based local learning (GMM 4 clusters) </td>
            <td>20220620-P53C6</td>
        </tr>
        <tr>
            <td>cluster-based local learning (GMM 5 clusters) </td>
            <td>20220620-P53C7</td>
        </tr>
        <tr>
            <td rowspan=6>Table 4.10</td>
            <td>pair-based local learning</td>
            <td>20220606-P46C1</td>
        </tr>
        <tr>
            <td>global learning</td>
            <td>20220606-P46C2</td>
        </tr>
        <tr>
            <td>cluster-based local learning (GMM 2 clusters)</td>
            <td>20220606-P46C4</td>
        </tr>
        <tr>
            <td>cluster-based local learning (GMM 3 clusters)</td>
            <td>20220606-P46C5</td>
        </tr>
        <tr>
            <td>cluster-based local learning (GMM 4 clusters) </td>
            <td>20220606-P46C6</td>
        </tr>
        <tr>
            <td>cluster-based local learning (GMM 5 clusters) </td>
            <td>20220606-P46C7</td>
        </tr>
    </tbody>
</table>