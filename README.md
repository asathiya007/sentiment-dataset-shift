# sentiment-dataset-shift
Course project for CS 8803 DMM (Data Management and Machine Learning) at Georgia Tech  (Fall 2021). Quantifies the dataset shift of sentiment classification datasets.

# Getting Started 
Create a Python virtual environment (with Python 3.8) and install the dependencies from the `./requirements.txt` file. 

There are a set of default datasets in the `./datasets.zip` file. If using custom datasets, add them to the `./datasets` directory (produced by unzipping the `./datasets.zip` file) and specify custom data loading functions in the `./data.py` file. The `./data.py` file contains default data loading functions for the default datasets in the `./datasets` directory. 

# Default Datasets
The default datasets used in this paper are listed below. 

1. IMDB

Citation: Maas, A. L., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., & Potts, C. (2011). Learning Word Vectors for Sentiment Analysis. Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies, 142-150. http://www.aclweb.org/anthology/P11-1015. https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews  

2. Movie Reviews 

Citation: Pang, B., & Lee, L. (2004). A Sentimental Education: Sentiment Analysis Using Subjectivity Summarization Based on Minimum Cuts. Proceedings of the ACL. https://www.kaggle.com/nltkdata/movie-review?select=movie_review.csv  

3. Emotions Dataset

Citation: Saravia, E., Liu, H.T., Huang, Y., Wu, J., & Chen, Y. (2018). CARER: Contextualized Affect Representations for Emotion Recognition. Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, 3687-3697. https://aclanthology.org/D18-1404. https://www.kaggle.com/praveengovi/emotions-dataset-for-nlp?select=train.txt  

4. Twitter 

Citation: Merin, S. (2019). Twitter Emotion Analysis. Kaggle. Retrieved October 29, 2021, from https://www.kaggle.com/shainy/twitter-emotion-analysis/data  

5. Financial News

Citation: CrowdFlower. (2016, November 21). \textit{Sentiment Analysis in Text}. data.world. Retrieved November 2, 2021, from https://data.world/crowdflower/sentiment-analysis-in-text.

6. Crowd Flower 

Citation: Ayuya, C. (2020, November 30). \textit{Correcting Dataset Shift in Machine Learning}. Section. Retrieved November 2, 2021, from https://www.section.io/engineering-education/correcting-data-shift/. 
