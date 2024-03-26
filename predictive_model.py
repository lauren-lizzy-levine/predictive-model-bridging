from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


np.random.seed(42)


def read_data():
    train = pd.read_csv("train.csv", sep="\t")
    dev = pd.read_csv("dev.csv", sep="\t")
    # Make a dev set based on complete documents

    # Get a baseline:
    print("Baseline:")
    print(sum(train["bridge"]) / len(train["bridge"]))

    features = []  # numerical features
    # "t_n_dist"

    # Scale data, if necessary
    #scaler = StandardScaler()
    #X_train = scaler.fit_transform(train[features])
    X_train = train[features]
    y_train = train["bridge"]
    #X_dev = scaler.transform(dev[features])
    X_dev = dev[features]
    y_dev = dev["bridge"]

    # Ordinally encode categorical variables
    encoder = OrdinalEncoder()
    cat_labels = ["t_entity_type", "t_head_lemma", "t_head_deprel", "t_head_xpos", "t_head_number", "t_infostat", "genre",
                  "n_entity_type", "n_head_lemma", "n_head_deprel", "n_head_xpos", "n_head_number"] # categorical features
    #t/n_entity_text", "t/n_head_form"
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


def save_predictions(predictions, data_file, column_name):
    dev = pd.read_csv(data_file, sep="\t")
    dev[column_name] = predictions
    dev.to_csv(data_file, sep='\t', index=False)
    return


def random_forest(X_train, y_train, X_dev, y_dev, features):
    param_grid = {'n_estimators': [50, 100, 200, 300],
                  'max_depth': [25, 50, 75, None]}

    rf = RandomForestClassifier(random_state=42)
    # Use random search to find the best hyperparameters
    rand_search = GridSearchCV(rf, param_grid=param_grid, cv=5)
    rand_search.fit(X_train, y_train)
    rf = rand_search.best_estimator_

    # Print the best hyperparameters
    print('Best hyperparameters:', rand_search.best_params_)

    y_pred = rf.predict(X_dev)
    print(rf.__class__.__name__, accuracy_score(y_dev, y_pred))

    # save off predictions
    save_predictions(y_pred, "dev.csv", "preds")

    # feature importance
    feat_scores = rf.feature_importances_
    named_importances = [(feat_scores[i], features[i]) for i in range(len(features))]

    output = [str(tup) for tup in sorted(named_importances, reverse=True)]
    print("\n".join(output))

    # feature importance chart
    sorted_importance = sorted(named_importances, key=lambda x: x[0], reverse=True)
    sorted_features = [x[1] for x in sorted_importance]
    sorted_feat_scores = [x[0] for x in sorted_importance]
    plt.bar(sorted_features, sorted_feat_scores)
    plt.xticks(rotation=15)

    # Adding labels and title
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('Feature Importance in Bridging Prediction Model')

    # Displaying the graph
    plt.show()

    return


def sort_predictions():
    dev = pd.read_csv("dev.csv", sep="\t")
    true_positives = dev[(dev["bridge"] == 1) & (dev["preds"] == 1)]
    false_positives = dev[(dev["bridge"] == 0) & (dev["preds"] == 1)]
    true_negatives = dev[(dev["bridge"] == 0) & (dev["preds"] == 0)]
    false_negatives = dev[(dev["bridge"] == 1) & (dev["preds"] == 0)]
    true_positives.to_csv("dev_true_positives.csv", sep='\t', index=False)
    false_positives.to_csv("dev_false_positives.csv", sep='\t', index=False)
    true_negatives.to_csv("dev_true_negatives.csv", sep='\t', index=False)
    false_negatives.to_csv("dev_false_negatives.csv", sep='\t', index=False)
    return


def distr_by_genre():
    data = pd.read_csv("train_dev.tab", sep="\t")
    genre_instances = data.groupby('genre')['bridge'].sum().reset_index(name='Total')
    genre_instances = genre_instances.sort_values(by="Total", ascending=False)
    print(genre_instances)
    # creating a bar plot
    plt.bar(genre_instances["genre"], genre_instances["Total"])
    # adding labels and title
    plt.xlabel('Genre')
    plt.ylabel('Sum of Bridging Instances')
    plt.title('Distribution of Bridging Instances by Genre')
    plt.xticks(rotation=30)
    # displaying the plot
    plt.show()

    return


def acc_by_genre():
    data = pd.read_csv("dev.csv", sep="\t")
    genres = data["genre"].unique()
    scores = []
    for genre in genres:
        genre_data = data[data["genre"] == genre]
        y_dev = genre_data["bridge"]
        y_pred = genre_data["preds"]
        scores.append((genre, accuracy_score(y_dev, y_pred)))
    sorted_genre_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    sorted_scores = [x[1] for x in sorted_genre_scores]
    sorted_genre = [x[0] for x in sorted_genre_scores]
    plt.bar(sorted_genre, sorted_scores)
    plt.xticks(rotation=15)

    # Adding labels and title
    plt.xlabel('Genres')
    plt.ylabel('Accuracy')
    plt.title('Dev Accuracy by Genre')  # (with Distance)

    # Displaying the graph
    plt.show()

    return


def confusion_matrix():
    data = pd.read_csv("dev.csv", sep="\t")
    #data = data[(data["bridge"]==1) | (data["coref"]==1)]
    y_dev = data["bridge"]
    y_pred = data["preds"]
    print(classification_report(y_dev, y_pred))
    cm = confusion_matrix(y_dev, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()
    return


def main():
    # read in data
    X_train, y_train, X_dev, y_dev, all_features = read_data()
    # train classifier and make/save predictions on dev
    random_forest(X_train, y_train, X_dev, y_dev, all_features)
    # divide dev predictions in FP, FN, TP, TN files for manual analysis
    sort_predictions()
    # making graphs for analysis
    distr_by_genre()
    acc_by_genre()
    confusion_matrix()
    return


if __name__ == "__main__":
    main()
