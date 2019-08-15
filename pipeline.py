import luigi
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib

class CleanDataTask(luigi.Task):
    """ Cleans the input CSV file by removing any rows without valid geo-coordinates.

        Output file should contain just the rows that have geo-coordinates and
        non-(0.0, 0.0) files.
    """
    tweet_file = luigi.Parameter()
    output_file = luigi.Parameter(default='clean_data.csv')

    def output(self):
        return luigi.LocalTarget(self.output_file)
    
    def _clean_tweet_coord(self, df):
        # remove NaN
        remove_nan_df = df[~df["tweet_coord"].isnull()]
        # remove (0,0)
        remove_00_df = remove_nan_df[~remove_nan_df["tweet_coord"].str.contains('[0.0, 0.0]', regex = False)]
        return remove_00_df

    def run(self):
        airline_tweets_df = pd.read_csv(self.tweet_file, encoding = "ISO-8859-1")
        clean_df = self._clean_tweet_coord(airline_tweets_df)
        clean_df.to_csv(self.output_file, index=False, encoding = "ISO-8859-1") 
   
class TrainingDataTask(luigi.Task):
    """ Extracts features/outcome variable in preparation for training a model.

        Output file should have columns corresponding to the training data:
        - y = airline_sentiment (coded as 0=negative, 1=neutral, 2=positive)
        - X = a one-hot coded column for each city in "cities.csv"
    """
    tweet_file = luigi.Parameter()
    cities_file = luigi.Parameter(default='cities.csv')
    output_file = luigi.Parameter(default='features.csv')

    def output(self):
        return luigi.LocalTarget(self.output_file)
    
    def requires(self):
        return CleanDataTask(self.tweet_file)

    def run(self):
        
        clean_df = pd.read_csv(self.input().path, encoding = "ISO-8859-1")
        clean_df["tweet_coord"] = clean_df["tweet_coord"].str.findall(r"[-+]?\d*\.\d+|\d+")
        
        # get the latitude, longitude, city name list
        cities_df = pd.read_csv(self.cities_file, encoding = "ISO-8859-1")
        city_coord = cities_df[['latitude', 'longitude']].values        
        city_names = cities_df['name'].values
        
        # get the closest city to the "tweet_coord" using Euclidean distance
        
        clean_df['city_id'] = clean_df["tweet_coord"].map(
            lambda x: np.argmin([(float(x[0])- coord[0])**2 + (float(x[1])-coord[1])**2 for coord in city_coord]))
        
        clean_df['city_name'] = clean_df['city_id'].map(lambda x: city_names[x])
        
        # get one-hot encoding of the city names.
        cityname_onehot = pd.get_dummies(cities_df['name'], columns=['name'])
        city_oh_array = cityname_onehot.values
        
        # set X
        clean_df["X"] = clean_df['city_id'].map(lambda x: city_oh_array[x])
        # set Y
        sentiment_type = ['negative', 'neutral', 'positive'] 
        clean_df['y'] = clean_df['airline_sentiment'].map(lambda x: sentiment_type.index(x))
        
        # get output ready
        features_df = pd.DataFrame(columns=['city_id', 'city_name', "X", "y"])
        features_df[['city_id', 'city_name', "X", "y"]] = clean_df[['city_id', 'city_name', "X", "y"]]
        
        # encode X to string
        features_df["X"] = features_df["X"].map(lambda x: ' '.join([str(aa) for aa in x]))
        features_df.to_csv(self.output_file, index=False, encoding = "ISO-8859-1") 
        
class TrainModelTask(luigi.Task):
    """ Trains a classifier to predict negative, neutral, positive
        based only on the input city.

        Output file should be the pickle'd model.
    """
    tweet_file = luigi.Parameter()
    output_file = luigi.Parameter(default='model.pkl')
    
    def output(self):
        return luigi.LocalTarget(self.output_file)
    
    def requires(self):
        return TrainingDataTask(self.tweet_file)
    
    def run(self):
        features_df = pd.read_csv(self.input().path, encoding = "ISO-8859-1")

        # decode X
        features_df["X"] = features_df["X"].map(lambda x: x.split(" "))
        
        # get training input
        X = features_df['X'].values.tolist() 
        y = features_df["y"].values.tolist()
        X = np.asarray(X).astype(float)

        # build a LogisticRegression Model
        clf = LogisticRegression(random_state=0, solver='lbfgs',
                             multi_class='multinomial').fit(X, y)

        joblib.dump(clf, 'model.pkl') 
        
class ScoreTask(luigi.Task):
    """ Uses the scored model to compute the sentiment for each city.

        Output file should be a four column CSV with columns:
        - city name
        - negative probability
        - neutral probability
        - positive probability
    """
    tweet_file = luigi.Parameter()
    output_file = luigi.Parameter(default='scores.csv')

    def output(self):
        return luigi.LocalTarget(self.output_file)
    
    def requires(self):
        return {"model":TrainModelTask(self.tweet_file), "features":TrainingDataTask(self.tweet_file)}
    
    def run(self):
        features_df = pd.read_csv(self.input()["features"].path, encoding = "ISO-8859-1")
        features_df["X"] = features_df["X"].map(lambda x: x.split(" "))
        
        # get training input
        X = features_df['X'].values.tolist() 
        y = features_df["y"].values.tolist()
        X = np.asarray(X, dtype = float)

        # load a LogisticRegression Model
        clf_load = joblib.load(self.input()["model"].path) 
        y_pred_prob = clf_load.predict_proba(X)
        
        sentiment_type = ['negative', 'neutral', 'positive']
        scores_df = pd.DataFrame(y_pred_prob, columns = sentiment_type)
        scores_df["city name"] = features_df["city_name"]
        scores_df = scores_df.drop_duplicates()
        
        scores_df = scores_df.sort_values('positive', ascending=False)
        
        scores_df.to_csv(self.output_file, index=False, encoding = "ISO-8859-1")
        
if __name__ == "__main__":
    luigi.run()
