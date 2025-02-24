from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
import os
import json

app = FastAPI()

# 启用CORS以允许前端访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 数据模型
class Paper(BaseModel):
    id: str
    title: str
    authors: str
    categories: str
    url: str

# 文件路径
DATA_DIR = "recommender_data"
MODEL_PATH = os.path.join(DATA_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(DATA_DIR, "vectorizer.pkl")
HISTORY_PATH = os.path.join(DATA_DIR, "history.json")

# 确保数据目录存在
os.makedirs(DATA_DIR, exist_ok=True)

class PaperRecommender:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.model = SGDClassifier(loss='modified_huber')
        self.history = []
        self.load_data()

    def load_data(self):
        """从磁盘加载模型和历史数据"""
        try:
            if os.path.exists(VECTORIZER_PATH):
                with open(VECTORIZER_PATH, 'rb') as f:
                    self.vectorizer = pickle.load(f)
            
            if os.path.exists(MODEL_PATH):
                with open(MODEL_PATH, 'rb') as f:
                    self.model = pickle.load(f)
            
            if os.path.exists(HISTORY_PATH):
                with open(HISTORY_PATH, 'r') as f:
                    self.history = json.load(f)
                    
                # 如果有历史数据，确保模型已训练
                if self.history:
                    self.retrain_model()
                    
        except Exception as e:
            print(f"加载数据时出错: {e}")

    def save_data(self):
        """保存模型和历史数据到磁盘"""
        try:
            with open(VECTORIZER_PATH, 'wb') as f:
                pickle.dump(self.vectorizer, f)
            
            with open(MODEL_PATH, 'wb') as f:
                pickle.dump(self.model, f)
            
            with open(HISTORY_PATH, 'w') as f:
                json.dump(self.history, f)
        except Exception as e:
            print(f"保存数据时出错: {e}")

    def get_paper_features(self, paper: Paper) -> str:
        """将论文转换为特征文本"""
        return f"{paper.title} {paper.authors} {paper.categories}"

    def retrain_model(self):
        """使用历史数据重新训练模型"""
        if not self.history:
            return

        texts = [item['text'] for item in self.history]
        labels = [item['clicked'] for item in self.history]
        
        # 检查是否只有一个类别
        unique_labels = set(labels)
        if len(unique_labels) < 2:
            # 如果只有正样本，添加一个负样本
            if 1 in unique_labels:
                texts.append(texts[0])  # 使用第一个样本作为负样本
                labels.append(0)
            # 如果只有负样本，添加一个正样本
            else:
                texts.append(texts[0])
                labels.append(1)
        
        # 重新训练词向量和模型
        X = self.vectorizer.fit_transform(texts)
        # 每次训练时创建新的模型实例
        self.model = SGDClassifier(loss='modified_huber')
        self.model.fit(X, labels)

    def predict_interest(self, papers: List[Paper]) -> List[float]:
        """预测论文的兴趣度"""
        if not self.history:  # 如果没有历史数据，返回中等兴趣度
            return [0.5] * len(papers)

        texts = [self.get_paper_features(paper) for paper in papers]
        X = self.vectorizer.transform(texts)
        
        # 使用predict_proba获取正类的概率
        probas = self.model.predict_proba(X)
        return [float(p[1]) for p in probas]  # 返回正类概率作为兴趣度

    def update_history(self, paper: Paper):
        """更新历史数据并重新训练模型"""
        text = self.get_paper_features(paper)
        self.history.append({
            'text': text,
            'clicked': 1  # 用户点击表示感兴趣
        })
        
        # 立即为每个正样本添加一个负样本
        self.history.append({
            'text': text,
            'clicked': 0  # 同样的论文作为负样本
        })

        self.retrain_model()
        self.save_data()

    def update_uninterested(self, papers: List[Paper]):
        """将一批论文标记为不感兴趣"""
        for paper in papers:
            text = self.get_paper_features(paper)
            # 检查这篇论文是否已经在历史记录中
            if not any(h['text'] == text for h in self.history):
                self.history.append({
                    'text': text,
                    'clicked': 0  # 标记为不感兴趣
                })
        
        self.retrain_model()
        self.save_data()

recommender = PaperRecommender()

@app.post("/ask")
async def ask(papers: List[Paper]):
    scores = recommender.predict_interest(papers)
    return scores

@app.post("/update")
async def update(paper: Paper):
    recommender.update_history(paper)
    return {"status": "success"}

@app.post("/update_uninterested")
async def update_uninterested(papers: List[Paper]):
    """将一批论文标记为不感兴趣"""
    for paper in papers:
        text = recommender.get_paper_features(paper)
        # 检查这篇论文是否已经在历史记录中
        if not any(h['text'] == text for h in recommender.history):
            recommender.history.append({
                'text': text,
                'clicked': 0  # 标记为不感兴趣
            })
    
    recommender.retrain_model()
    recommender.save_data()
    return {"status": "success"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=9898) 