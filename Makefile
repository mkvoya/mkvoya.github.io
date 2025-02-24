.PHONY: paper install

# 默认目标
all: paper

# 安装依赖
install:
	pip install -r requirements.txt

# 运行论文推荐服务器
paper:
	python paper_recommender.py

# 清理生成的数据文件
clean:
	rm -rf recommender_data/ 