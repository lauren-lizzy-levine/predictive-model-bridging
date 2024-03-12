from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

np.random.seed(42)


def read_data():
    facts = pd.read_csv("gum_data.tab", sep="\t", quoting=3)
    # Make a dev set based on complete documents
    dev_docs = {"GUM_academic_game", "GUM_conversation_family", "GUM_bio_moreau", "GUM_fiction_claus",
                "GUM_interview_ants", "GUM_news_afghan", "GUM_speech_albania", "GUM_textbook_cognition",
                "GUM_reddit_callout", "GUM_vlog_covid", "GUM_voyage_fortlee", "GUM_whow_ballet"}  # for example
    train = facts[~facts['doc_id'].isin(dev_docs)]
    dev = facts[facts['doc_id'].isin(dev_docs)]

    # Get a baseline:
    print("Baseline:")
    print(sum(train["response"] / len(train["response"])))

    features = ["x", "y", "z"]  # some features

    # Scale data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[features])
    y_train = train["response"]
    X_dev = scaler.transform(dev[features])
    y_dev = dev["response"]

    # Ordinally encode categorical variables
    encoder = OrdinalEncoder()
    cat_labels = []  # some categorical features
    train_copy = train.copy()
    dev_copy = dev.copy()

    for feat in cat_labels:
        # Check which values are frequent
        frequent_val_rows = train[feat].value_counts() > 5
        # Get names of those rows in value_counts()
        frequent_values = train[feat].value_counts()[frequent_val_rows].index.tolist()

        train_copy.loc[~train_copy[feat].isin(frequent_values), feat] = "UNKNOWN"
        dev_copy.loc[~dev_copy[feat].isin(frequent_values), feat] = "UNKNOWN"

    train_cat = encoder.fit_transform(train_copy[cat_labels])
    dev_cat = encoder.transform(dev_copy[cat_labels])

    train_with_cat = np.hstack([X_train, train_cat])
    dev_with_cat = np.hstack([X_dev, dev_cat])

    all_features = features + cat_labels

    return train_with_cat, y_train, dev_with_cat, y_dev, all_features


def random_forest(X_train, y_train, X_dev, y_dev, features):
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=3)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_dev)
    print(rf.__class__.__name__, accuracy_score(y_dev, y_pred))

    # feature importance
    feat_scores = rf.feature_importances_
    named_importances = [(feat_scores[i], features[i]) for i in range(len(features))]

    output = [str(tup) for tup in sorted(named_importances, reverse=True)]
    print("\n".join(output))

    return


def main():
    X_train, y_train, X_dev, y_dev, all_features = read_data()
    random_forest(X_train, y_train, X_dev, y_dev, all_features)
    return


if __name__ == "__main__":
    main()
