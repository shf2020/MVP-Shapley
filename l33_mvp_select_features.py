import numpy as np
import pandas as pd
import lightgbm
import shap
from lightgbm import log_evaluation, early_stopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, mean_squared_error, classification_report
from sklearn.preprocessing import LabelEncoder
from shap_plot import summary_bar_plot, summary_dot_plot, dependence_plot, waterfall_plot
import joblib
import matplotlib.pyplot as plt
import multiprocessing
from tqdm import tqdm
from itertools import combinations
import pandas as pd
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import logging
logging.getLogger('matplotlib').setLevel(logging.ERROR)
def flatten(group):
            return pd.Series(group.values.flatten())
def data_load(path):
    df = pd.read_csv(path, index_col=0)
    df = df.groupby('room_id').filter(lambda x: len(x) == 6)
    df = df.sort_values(by=["room_id", 'home_team'], ascending=(True, False))
    return df
path = "data/dianfengsai_home_result_20240315_14plus_goalin.csv"
df1 = data_load(path)
test = pd.read_csv("data/dianfengsai_home_result_20240715_500_goalin.csv")
test = test.groupby('room_id').filter(lambda x: len(x) == 6)
test = test.sort_values(by=["room_id", 'home_team'], ascending=(True, False))
df = pd.concat([df1, test], axis=0)   

# 全部特征重要性排序
# ['defensiverebound', 'allmorale', 'floorball', 'nointerthreeshoot', 
#  'assist', 'beblocked', 'role_player_id', 'rebound', 'steal', 'irrationalshootnum',
#  'nointershoot', 'block', 'afktime', 'disturbedthreepointsshootsnum', 'bestealed',
#  'passball', 'offensiverebound', 'offenceovertime', 'threepointsshoot', 'passfailed',
#  'bedisturbedthreepointsshootsnum', 'bestrongdisturbedshoot', 'disturbenemyshoot', 
#  'bedisturbednum', 'beblockedultraskill', 'blockultraskillhit', 'blockultraskill', 
#  'passiveintercept', 'totalmovedis', 'bignointerthreeshoot', 'blocktwodunk', 'ultraskill', 
#  'benicepass', 'pickandrollnum', 'beblockedtwodunk', 'bignointershoot', 'ballownertimeproportion', 
#  'nicepass', 'blocktwolayup', 'beblockedthreepointsshoot', 'shootproportion', 'bedisturbeddunknum', 
#  'allshoots', 'blockthreepointsshoot', 'bedisturbedshoot', 'nointertwoshoot', 'boxoutsuccessnum', 
#  'feintshoot', 'twopointsshoot', 'twoshots', 'shotnodribble', 'boxoutpushnum', 'defencecrossover', 
#  'beblockedtwolayup', 'boxoutnum', 'bedisturbedlayupnum', 'sorryblocks', 'disturbeddunknum', 
#  'bedisturbedtwoshotsnum', 'disturbedlayupnum', 'twodunk', 'beblockedtwoshots', 'shakedown', 
#  'nicepassshoot', 'screensucess', 'crossoverloseball', 'stealultraskillmiss', 'blockultraskillmiss', 
#  'bignointertwoshoot', 'pickandrollsuccessnum', 'twolayup', 'enemycrossoverloseball', 
#  'stealultraskillhit', 'disturbedtwoshotsnum', 'skillfloorball', 'buffultraskillsteal', 
#  'buffultraskillrebound', 'blocktwoshots', 'skillintercept', 'pickandrolldownnum', 
#  'buffultraskillblock', 'afknum', 'buffultraskillintercept', 'buffultraskilltwoshots', 
#  'defencecrossoverdeadlose', 'twobarelylayup', 'morale', 'buffultraskillfloorball']

features = ['twoshots', 'shotnodribble', 'boxoutpushnum', 'defencecrossover', 
 'beblockedtwolayup', 'boxoutnum', 'bedisturbedlayupnum', 'sorryblocks', 'disturbeddunknum', 
 'bedisturbedtwoshotsnum', 'disturbedlayupnum', 'twodunk', 'beblockedtwoshots', 'shakedown', 
 'nicepassshoot', 'screensucess', 'crossoverloseball', 'stealultraskillmiss', 'blockultraskillmiss', 
 'bignointertwoshoot', 'pickandrollsuccessnum', 'twolayup', 'enemycrossoverloseball', 
 'stealultraskillhit', 'disturbedtwoshotsnum', 'skillfloorball', 'buffultraskillsteal', 
 'buffultraskillrebound', 'blocktwoshots']
acc_best = 0.53
# 迭代不同数量的特征组合
for r in range(5, len(features)+1):
    for select_features in combinations(features, r):
        select_features = list(select_features) + ['threepointsshoot','twopointsshoot', 'skillintercept', 'pickandrolldownnum', 
 'buffultraskillblock', 'afknum', 'buffultraskillintercept', 'buffultraskilltwoshots', 
 'defencecrossoverdeadlose', 'twobarelylayup', 'morale', 'buffultraskillfloorball']
        print(select_features)
        
        def data_preprocess(df):
            df_flatten = df.groupby('room_id').apply(flatten)
            feature_names = ['room_id', 'role_id', 'home_team', 'result', 'bestrongdisturbedshoot', 'nicepass', 'buffultraskillsteal',
                'twoshots', 'buffultraskillrebound', 'block', 'twopointsgoalin', 'threepointsgoalin', 'disturbedlayupnum', 'beblockedthreepointsshoot', 'threepointsshoot',
                'ultraskill', 'benicepass', 'allmorale', 'beblockedtwolayup', 'blocktwoshots', 'bedisturbedtwoshotsnum', 'skillintercept',
                'bedisturbedthreepointsshootsnum', 'bedisturbedlayupnum', 'pickandrollnum', 'pickandrollsuccessnum', 'feintshoot',
                'pickandrolldownnum', 'boxoutsuccessnum', 'rebound', 'offenceovertime', 'beblockedultraskill', 'bignointerthreeshoot',
                'boxoutpushnum', 'crossoverloseball', 'shakedown', 'ballownertimeproportion', 'irrationalshootnum', 'offensiverebound',
                'twopointsshoot', 'buffultraskillblock', 'skillfloorball', 'mvppoint', 'defencecrossover', 'bedisturbedshoot', 'assist',
                'blockultraskillmiss', 'defensiverebound', 'screensucess', 'beblocked', 'beblockedtwodunk', 'bedisturbeddunknum',
                'nointertwoshoot', 'bedisturbednum', 'sorryblocks', 'bestealed', 'blocktwodunk', 'afknum', 'nointershoot', 'stealultraskillmiss',
                'passfailed', 'buffultraskillintercept', 'disturbedtwoshotsnum', 'buffultraskilltwoshots', 'defencecrossoverdeadlose',
                'bignointertwoshoot', 'steal', 'blockultraskill', 'twobarelylayup', 'boxoutnum', 'stealultraskillhit', 'shotnodribble',
                'afktime', 'disturbedthreepointsshootsnum', 'beblockedtwoshots', 'morale', 'role_player_id',  'bignointershoot',
                'blocktwolayup', 'passiveintercept', 'disturbenemyshoot', 'nointerthreeshoot', 'buffultraskillfloorball', 'twolayup',
                'enemycrossoverloseball', 'passball', 'disturbeddunknum', 'twodunk', 'nicepassshoot', 'blockultraskillhit', 'shootproportion',
                'floorball', 'blockthreepointsshoot', 'allshoots', 'totalmovedis']
            flatten_feature_names = [feature_names[i]+'_'+str(k) for k in range(6) for i in range(len(feature_names))]
            
            df_flatten.columns = flatten_feature_names

            train_feature_names = select_features  
            
            
            total_over_features = [train_feature_names[i]+'_'+str(k) for k in range(6) for i in range(len(train_feature_names))]
            X = df_flatten[[train_feature_names[i]+'_'+str(k) for k in range(6) for i in range(len(train_feature_names))]]
            # print(len(feature_names),len(train_feature_names))
            # print(X.info())
            Y = df_flatten['result_0']
            return X, Y, df_flatten, total_over_features

        def model_train(X, Y):
            # categorical_features = ['role_player_id'+'_'+str(k) for k in range(6)]
            categorical_features =[]
            encoders = []
            for feature in categorical_features:
                if X[feature].dtypes != object:
                    X[feature] = X[feature].astype(str)
                encoder = LabelEncoder()
                encoder.fit(X[feature])
                encoders.append(encoder)
                X[feature] = encoder.transform(X[feature])
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)
            model = lightgbm.LGBMClassifier(objective='binary', num_leaves=31, learning_rate=0.05, n_estimators=1000)
            callbacks = [log_evaluation(period=100), early_stopping(stopping_rounds=10)]
            model.fit(X_train, Y_train, eval_set=[(X_test, Y_test)], eval_metric='binary_logloss', callbacks=callbacks, categorical_feature=categorical_features)
            preds = model.predict(X, num_iteration=model.best_iteration_)
            report = classification_report(Y, preds)
            # print(report)
            return X, model

        
        X, Y, df_flatten, total_over_features = data_preprocess(df)
        X, model = model_train(X, Y)

        with open('replay.log', 'r') as file:
            lines = file.readlines()
        # 处理每一行，提取最后部分
        result = [line.strip().split('/')[-1] for line in lines]

        # 球员id_姓名字典
        player_dict = {
        101010001: '詹姆斯',
        101010002: '库里',
        101010003: '约基齐',
        101010004: '保罗',
        101010005: '杜兰特',
        101010007: '东契奇',
        101010008: '扬尼斯',
        101010009: '锡安',
        101010011: '利拉德',
        101010012: '保罗乔治',
        101010013: '韦斯特布鲁克',
        101010014: '傅值',
        101010015: '巴特勒',
        101010017: '胡里奥',
        101010019: '戈贝尔',
        101010020: '哈登',
        101010022: '唐斯',
        101010023: '卡佩拉',
        101010025: '欧文',
        101010026: '恩比德',
        101010027: '瓦兰',
        101010028: '现役罗斯',
        101010029: '阿德巴约',
        101010034: '西亚卡姆',
        10101006: '威金斯',
        101010037: '布克',
        101010038: '菲利普',
        101010040: '拉文',
        101010041: '德罗赞',
        101010047: '塔图姆',
        101010049: '安东尼',
        101010050: '塞斯库里',
        101010006: '罗罗',
        101010082: '汤普森',
        101010054: '克拉克森',
        101010056: '吕小峰',
        101010057: '陈浩宇',
        101010058: '洪寿',
        101010059: '戴维斯',
        101010053: '张阳',
        101010060: '伦纳德',
        101010061: '周长',
        101010062: '韩旭',
        101010064: '穆雷',
        101010066: '库明加',
        101010067: '施罗德',
        101010068: '亚当斯',
        101010069: '富尼耶',
        101010070: '阿隆戈登',
        101010071: '波尔津吉斯',
        101010088: '郭艾伦',
        101010087: '易建联',
        101010089: '林书豪',
        101010095: '李梦',
        101010097: '诺维茨基'
        }

        # print(test.info())
        X, Y, df_flatten, total_over_features = data_preprocess(test)
        # categorical_features = ['role_player_id'+'_'+str(k) for k in range(6)]
        categorical_features = []
        encoders = []
        for feature in categorical_features:
            if X[feature].dtypes != object:
                X[feature] = X[feature].astype(str)
            encoder = LabelEncoder()
            encoder.fit(X[feature])
            encoders.append(encoder)
            X[feature] = encoder.transform(X[feature])
            
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X)
        data = df_flatten
            
        # 保存球员贡献
        def compute_shap_values(shap_values):
            feat_num = len(shap_values[0]) // 6 #每个球员有33个特征
            mvp_values = []
            for i in tqdm(range(len(X))):
                shap_value = shap_values[i]
                mvp_value = 0
                for j in range(len(shap_value)):
                    mvp_value += shap_value[j]
                    if j % feat_num == feat_num - 1:
                        mvp_values.append(mvp_value)
                        mvp_value = 0
            abs_array = np.abs(np.array(mvp_values).reshape(-1,6))
            sum_first_three = np.sum(abs_array[:, :3], axis=1, keepdims=True)
            x = 0.5
            normalized_first_three = (abs_array[:, :3] / sum_first_three) * x

            # 后三列数值除以后三列的和并乘以系数（1-x）
            sum_last_three = np.sum(abs_array[:, 3:], axis=1, keepdims=True)
            normalized_last_three = (abs_array[:, 3:] / sum_last_three) * (1 - x)

            # 合并处理后的数组
            result_array = np.hstack((normalized_first_three, normalized_last_three))
            return result_array, np.array(mvp_values).reshape(-1,6), shap_values

        # 队内排名
        def rank(values, order = 1):
            if order == 1:
                sorted_indices = np.argsort(values)[::-1]
            else:
                sorted_indices = np.argsort(values)
            ranks = np.empty_like(sorted_indices)
            ranks[sorted_indices] = np.arange(1, len(sorted_indices) + 1)

            return ranks, sorted_indices[0]

        def process_data(args):
            df, feat_values, shap_values, id = args
        
            used_features = select_features
            
            
            feature_value = []
            room_id_feature = 'room_id'
            room_id_features = [room_id_feature + '_' + str(i) for i in range(6)]
            player_feature = 'role_player_id'
            player_features = [player_feature + '_' + str(i) for i in range(6)]
            home_team_feature = 'home_team'
            home_team_features = [home_team_feature + '_' + str(i) for i in range(6)]
            result_feature = 'result'
            result_features = [result_feature + '_' + str(i) for i in range(6)]

            score_features = ['score_ourside', 'score_opposite']
            total_over_features = []
            feat_num = len(used_features)
            
            for feature in used_features:
                feature_value.append(feature)
                feature_value.append(feature + '_shap_value')
            value_name = [feature_value[i] for i in range(1, len(feature_value), 2)]

            for i in range(6):
                for feature in used_features:
                    feature = feature + '_' + str(i)
                    total_over_features.append(feature)
                    
            #ds, role_id
            new_data = pd.DataFrame(columns = ['room_id', 'player', 'home_team', 'result'] + feature_value + ['shap_score', 'shap_order_team'])
            for i in tqdm(range(len(df))):
            
                self_shap_values = shap_values[i][:3]
                opp_shap_values = shap_values[i][3:]
                
                for j in range(3):
                    #ds, role_id, feat_shap_value
                    new_data.loc[6 * i + j, ['room_id', 'player', 'home_team', 'result'] + used_features] = df.loc[i, [ room_id_features[j], player_features[j], home_team_features[j], result_features[j] ] + total_over_features[j * feat_num : (j + 1) * feat_num]].tolist()
                    new_data.loc[6 * i + j, value_name] = feat_values[i][j * feat_num : (j + 1) * feat_num] #n * feat_num * 6
                    
                rows = [6 * i + n for n in range(3)]
                shap_ranks, shap_index1 = rank(self_shap_values)
                new_data.loc[rows, 'shap_order_team'] = shap_ranks
                
                
                for j in range(3, 6):
                    new_data.loc[6 * i + j, ['room_id', 'player', 'home_team', 'result'] + used_features] = df.loc[i, [ room_id_features[j], player_features[j], home_team_features[j], result_features[j] ] + total_over_features[j * feat_num : (j + 1) * feat_num]].tolist()
                    
                    new_data.loc[6 * i + j, value_name] = feat_values[i][j * feat_num : (j + 1) * feat_num]
                
                rows = [6 * i + n for n in range(3,6)]
                shap_ranks, shap_index2 = rank(opp_shap_values, -1)
                new_data.loc[rows, 'shap_order_team'] = shap_ranks

            new_data['shap_score'] = shap_values.reshape(-1)
            return new_data

        # print('比赛数据：',len(data))
        money_values, shap_values, feat_values = compute_shap_values(shap_values.values)
        # 分割数据
        num_cores = 10
        chunk_size = len(data) // num_cores
        chunks = [(data[i:i + chunk_size].reset_index(drop = True), feat_values[i : i + chunk_size], shap_values[i : i + chunk_size], i // chunk_size * 100 // num_cores) for i in range(0, len(data), chunk_size)]
        # 使用线程池处理数据

        # 创建进程池，并利用多个核心进行并行处理
        with multiprocessing.Pool(processes=num_cores) as pool:
            results = pool.map(process_data, chunks)

        # 合并处理结果
        data_list = [results[i] for i in range(num_cores)]

        processed_data = pd.concat(data_list).reset_index(drop = True)
        

        # 众包结果
        mvp_voting_dict = {
            
        '44aaf8e0e7444939aadbce242ee4617a': ['詹姆斯', '波尔津吉斯', '保罗乔治'],

        '4a84741428374ffcb0f7cad730118797': ['塔图姆', '波尔津吉斯', '汤普森'],

        '4b60cdc0d940493cad5325d4e6dda650': ['欧文', '恩比德', '詹姆斯'],

        '132da64f9523480baca99acbd2a051d9': ['哈登', '戴维斯', '杜兰特'],

        '2bd1d5bb056841769a67123f758976e2': ['塔图姆', '哈登', '易建联'],

        '3fddcf35d43645d9b75545948495d8ae': ['西亚卡姆', '塔图姆', '保罗乔治'],

        '492027882ae243069517006be73d19e8': ['伦纳德', '东契奇', '戴维斯'],

        '0f6161914d084e15a18986eb30b8e89a': ['欧文', '戴维斯', '伦纳德'],

        '17f5795303d94816858b0bf77c07c431': ['哈登', '卡佩拉', '杜兰特'],

        '251be03c7f8445fbb70b3297b840425f': ['塔图姆', '伦纳德', '易建联'],

        '2ba99026de424a82bde390629ef19080': ['戴维斯', '塔图姆', '布克'],

        '145a060eca4d4e91bfbd048e454f9678': ['库里', '卡佩拉', '詹姆斯'],

        '26a0d99d56af41dd8aaea420a2d6d13b': ['欧文', '李梦', '卡佩拉'],

        '15000ca7389e42e2b557fae9b8be4170': ['哈登', '戴维斯', '保罗乔治'],

        '41279aaf56444040848b2046dc082417': ['欧文', '戴维斯', '巴特勒'],

        '1b77f7e4b9d540fc8ab8c07cac225acb': ['库里', '戴维斯', '傅值'],

        '1b33a8fa7f5a435f84887e1165f18f45': ['欧文', '戴维斯', '塔图姆'],

        '00958142ddcf471bb9b61ad133adf797': ['塔图姆', '波尔津吉斯', '李梦'],

        '14673bfcecc84487be0dfdc89c8fa27f': ['欧文', '戴维斯', '杜兰特'],

        '23f53dcc987d40a7a61301c06fb1d199': ['戴维斯', '塔图姆', '汤普森'],

        '3415c388012047a4ab18560263e98634': ['哈登', '恩比德', '杜兰特'],

        '0882aac2ac6446eeb3b46ab3056e9bb8': ['西亚卡姆', '塔图姆', '伦纳德'],

        '3f07433fcea44462becc70134e6e1ba4': ['哈登', '恩比德', '伦纳德'],

        '4b04bf914f624944bfea2420e1ea8237': ['东契奇', '保罗乔治', '塔图姆'],

        '331a2bd4af214bc892c77c7fcff32922': ['拉文', '波尔津吉斯', '保罗乔治'],

        '4ef11bfeb0014fa6aeec4f6cd39b0135': ['哈登', '李梦', '易建联'],

        '47959b41ab5b40898b55a1df7e37b841': ['李梦', '戴维斯', '詹姆斯'],

        '3a4b451fd4614379ac467ed5681db71d': ['东契奇', '戴维斯', '塔图姆'],

        '1ee629e9ae7e4177993cdae771411c6e': ['欧文', '卡佩拉', '伦纳德'],

        '1e274241bd14481d815f9f6121c5f6b7': ['东契奇', '戴维斯', '保罗乔治'],

        '1e5df39d5a1d4f35a6167af4091beb2a': ['哈登', '戴维斯', '李梦'],

        '0eda5d032b3f4344b4ed66dafbee5800': ['东契奇', '波尔津吉斯', '詹姆斯'],

        '202acf755a174e1fbc3cad1f3c40e68f': ['塔图姆', '李梦', '波尔津吉斯'],

        '02b3c81c5e1149cd86492338505b0bba': ['塔图姆', '李梦', '波尔津吉斯'],

        '020abe41e9c745e1816761133aece5a1': ['东契奇', '戴维斯', '保罗乔治'],

        '240e8121e61e408d957e66eeb5226cc2': ['阿隆戈登', '胡里奥', '汤普森'],

        '28739d5b6043463e80251ac566b7f3fa': ['西亚卡姆', '伦纳德', '波尔津吉斯'],

        '090ee4ac56514571a130dc4f90be3120': ['塔图姆', '易建联', '李梦'],

        '37cb26fb806047c8b5c2009301e557dd': ['欧文', '戴维斯', '杜兰特'],

        '09d94b1d998e46538ef971162d4c8e13': ['巴特勒', '戴维斯', '詹姆斯'],

        '21fdf5e1a92a47988424fb06190992f1': ['库里', '恩比德', '伦纳德'],

        '0bb8539921654104b5355288eb1850e4': ['李梦', '卡佩拉', '塔图姆'],

        '40ff9fbd01a44ed6a89aa0936754ab59': ['塔图姆', '汤普森', '保罗乔治'],

        '07ddba906b1e460e899960ced74a55e7': ['杜兰特', '波尔津吉斯', '塔图姆'],

        '2c070a32be5a4b3f8476fa939a68df1a': ['欧文', '戴维斯', '李梦']

        }

        #球员在所有胜场的shapley值排名平均
        df_test = processed_data.copy()
        df_test['player'] = df_test['player'].astype('int64').map(player_dict)
        ARD = []
        SRCC = []
        # R = []
        count=0
        num = 0
        # print(len(mvp_voting_dict))
        for roomid in mvp_voting_dict.keys():
            num+=1
            # print(roomid)
            mvp_voting = mvp_voting_dict[roomid]
            topk = len(mvp_voting)
            #取出胜方球员
            result_df = df_test[df_test['room_id'] == roomid] 
            # print(len(result_df))
            result_df_win = result_df[result_df['result'] == 1]
            mvp_by_shap = result_df_win.groupby('player')['shap_order_team'].mean().sort_values(ascending=True)
            # print(len(mvp_by_shap),mvp_by_shap[:10])

            mvp_rank = mvp_by_shap.rank().sort_values()
            # print(mvp_rank)
            rank_mean_err = 0
            predicted_labels = []
            for i in range(topk):
                predicted_labels.append(mvp_rank[mvp_voting[i]]-1)
                rank_mean_err += abs(mvp_rank[mvp_voting[i]]-i-1)
            # print('rank_mean_err:', rank_mean_err/topk)
            actual_labels = list(range(topk))
            # print(actual_labels)
            # print(predicted_labels)
            # 计算Spearman相关系数
            correlation, p_value = spearmanr(actual_labels, predicted_labels) 
            # print("Spearman相关系数:", correlation) 
            # print("p值:", p_value) 
            #预测第1名计算召回率=精度：
            if predicted_labels[0]==actual_labels[0]: 
                count+=1
        
        acc = count/num
        print(f"{num}场top1准确率: ", count/num)
        if acc>acc_best:
            acc_best = acc
            processed_data.to_csv('./data/l33_mvp_values_v7.csv')
            # features写入到 txt 文件
            with open('./results/select_features_v7.txt', 'w') as file:
                for item in select_features:
                    file.write(f"'{item}',")
                    