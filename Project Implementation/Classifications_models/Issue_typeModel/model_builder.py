from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier



def build_model(num_classes):
    model = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, min_df=2, ngram_range=(1, 2))),

        ('clf', XGBClassifier(
            objective='multi:softprob',
            num_class=num_classes,
            n_estimators=100,
            learning_rate=0.1,
            
            max_depth=3,
            subsample=0.7,
            colsample_bytree=0.7,
            gamma=2,
            reg_alpha=0.5,
            reg_lambda=1,
            
            random_state=42,
            n_jobs=-1
        ))
    ])
    
    return model