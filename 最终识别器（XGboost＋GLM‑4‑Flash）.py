# -*- coding: utf-8 -*-
"""
工业级批量识别脚本（修复版）：XGBoost + GLM-4-Flash
功能：
1. 【修复】文件读取增加“暴力容错模式”，解决 UnicodeDecodeError
2. 自动检测环境并安装
3. 支持断点续传
"""

import sys
import subprocess
import os


# ==========================================
# 0. 自动安装缺失的库
# ==========================================
def auto_install_packages():
    requirements = {
        "zhipuai": "zhipuai",
        "pandas": "pandas",
        "xgboost": "xgboost",
        "sklearn": "scikit-learn",
        "tqdm": "tqdm",
        "openpyxl": "openpyxl",
        "scipy": "scipy"
    }
    print("-" * 30)
    print("正在检查运行环境...")
    for import_name, pip_name in requirements.items():
        try:
            __import__(import_name)
        except ImportError:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name, "-i",
                                       "https://pypi.tuna.tsinghua.edu.cn/simple"])
                print(f"✅ {pip_name} 安装成功！")
            except Exception:
                pass
    print("✅ 环境检查完毕。")


auto_install_packages()

import csv
import time
import threading
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from zhipuai import ZhipuAI

# ==========================================
# 1. 用户配置区 (请修改这里)
# ==========================================

# 【必填】你的 GLM-4 API Key
API_KEY = "47c79b53c6de4a0ea193c0e7b3fa6393.WGMR1QBd8VoouIBM"

# 路径配置
TRAIN_SAMPLE = r"D:\Desktop\sample_train_3000_glm.csv"
TRAIN_PRED = r"D:\Desktop\train_pred_sample_3000_glm.csv"

TARGET_FILE = r"D:\Desktop\去重土耳其地震.csv"
CACHE_FILE = r"D:\Desktop\temp_glm_cache.csv"
FINAL_FILE = r"D:\Desktop\最终识别结果_27k.xlsx"

MAX_WORKERS = 5


# ==========================================
# 2. 强力文件读取函数 (核心修复)
# ==========================================

def robust_read_csv(file_path):
    """
    尝试多种编码读取文件，如果都失败，使用'replace'模式强制读取
    """
    print(f"[Load] 正在尝试读取文件: {file_path}")

    # 方法1: 尝试 UTF-8-SIG (Excel常用)
    try:
        return pd.read_csv(file_path, encoding='utf-8-sig', dtype={'id': str})
    except UnicodeDecodeError:
        pass

    # 方法2: 尝试 GB18030 (GBK的超集，支持更多字符)
    try:
        return pd.read_csv(file_path, encoding='gb18030', dtype={'id': str})
    except UnicodeDecodeError:
        pass

    # 方法3: 绝杀 - 忽略错误强制读取 (encoding_errors='replace')
    print("[Warn] 标准编码读取失败，启用暴力容错模式（非法字符将被替换为 ?）...")
    try:
        return pd.read_csv(file_path, encoding='utf-8', encoding_errors='replace', dtype={'id': str})
    except Exception as e:
        print(f"❌ 严重错误: 文件无法读取。原因: {e}")
        sys.exit(1)


# ==========================================
# 3. GLM-4 API 调用模块
# ==========================================

try:
    client = ZhipuAI(api_key=API_KEY)
except Exception:
    print("API Key 配置错误或 SDK 初始化失败")
    sys.exit(1)


def call_glm4_get_score(text, row_id):
    prompt_content = f"""
    请分析以下推文，判断其是否由机器人/智能体(Agent)生成。
    推文内容："{text}"

    请只输出一个 0 到 1 之间的概率数值，表示它是机器人的可能性。
    1.0 表示肯定是机器人，0.0 表示肯定是人类。
    不要输出任何其他文字，只输出数字。
    """

    for _ in range(3):
        try:
            response = client.chat.completions.create(
                model="glm-4-flash",
                messages=[{"role": "user", "content": prompt_content}],
                temperature=0.1,
                top_p=0.7
            )
            content = response.choices[0].message.content.strip()
            import re
            nums = re.findall(r"0\.\d+|1\.0|0|1", content)
            if nums:
                return float(nums[0])
            return 0.5
        except Exception:
            time.sleep(1)
    return 0.5


# ==========================================
# 4. 批量处理逻辑
# ==========================================

def batch_process_glm(df):
    total = len(df)

    if os.path.exists(CACHE_FILE):
        print(f"[Resume] 检测到缓存文件，正在加载进度...")
        try:
            df_cache = pd.read_csv(CACHE_FILE, dtype={'id': str})
            processed_map = dict(zip(df_cache['id'], df_cache['p_ai_glm']))
            print(f"[Resume] 已完成 {len(processed_map)} 条。")
        except:
            processed_map = {}
    else:
        processed_map = {}
        with open(CACHE_FILE, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'p_ai_glm'])

    pending_rows = []
    for idx, row in df.iterrows():
        rid = str(row['id'])
        if rid not in processed_map:
            pending_rows.append((rid, str(row['text'])))

    if not pending_rows:
        print("[Skip] 所有数据均已处理完毕！")
        df['p_ai'] = df['id'].astype(str).map(processed_map)
        return df

    print(f"[Start] 开始处理 {len(pending_rows)} 条数据...")
    file_lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_id = {
            executor.submit(call_glm4_get_score, text, rid): rid
            for rid, text in pending_rows
        }

        pbar = tqdm(total=len(pending_rows), desc="GLM识别中")

        for future in as_completed(future_to_id):
            rid = future_to_id[future]
            try:
                score = future.result()
            except:
                score = 0.5

            with file_lock:
                with open(CACHE_FILE, 'a', newline='', encoding='utf-8-sig') as f:
                    writer = csv.writer(f)
                    writer.writerow([rid, score])

            processed_map[rid] = score
            pbar.update(1)

        pbar.close()

    df['p_ai'] = df['id'].astype(str).map(processed_map)
    df['p_ai'] = df['p_ai'].fillna(0.5)
    return df


# ==========================================
# 5. 模型训练与融合模块
# ==========================================

def retrain_model_from_samples():
    print("-" * 30)
    print("[Train] 正在恢复 XGBoost 模型...")

    # 同样使用 robust_read_csv 防止样本读取报错
    df_train = robust_read_csv(TRAIN_SAMPLE)
    df_pred = robust_read_csv(TRAIN_PRED)

    texts = df_train['text'].fillna("").astype(str).tolist()
    labels = df_train['label'].astype(int).values
    glm_scores = df_pred['p_ai'].fillna(0.5).values.reshape(-1, 1)

    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)
    X_tfidf = vec.fit_transform(texts)

    X_final = hstack([X_tfidf, glm_scores])

    clf = XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, n_jobs=4
    )
    clf.fit(X_final, labels)

    return vec, clf


# ==========================================
# 6. 主程序
# ==========================================

if __name__ == "__main__":
    if "你的智谱" in API_KEY:
        print("❌ 错误：请先在代码第 59 行填入你的 API Key！")
        sys.exit(1)

    # 1. 使用修复后的函数读取文件
    df_target = robust_read_csv(TARGET_FILE)
    print(f"✅ 文件读取成功！共 {len(df_target)} 条数据。")

    # 2. 识别流程
    df_target = batch_process_glm(df_target)
    vectorizer, clf = retrain_model_from_samples()

    print("[Predict] 正在进行最终推理...")
    new_texts = df_target['text'].fillna("").astype(str).tolist()
    X_tfidf = vectorizer.transform(new_texts)
    X_glm = df_target['p_ai'].values.reshape(-1, 1)
    X_final = hstack([X_tfidf, X_glm])

    probs = clf.predict_proba(X_final)[:, 1]
    preds = (probs >= 0.7).astype(int)

    df_target['is_agent'] = preds
    df_target['agent_prob'] = probs

    print(f"[Save] 保存结果到: {FINAL_FILE}")
    df_target.to_excel(FINAL_FILE, index=False)

    print("\n" + "=" * 30)
    print(f"🎉 成功！识别出智能体: {sum(preds)} 个")