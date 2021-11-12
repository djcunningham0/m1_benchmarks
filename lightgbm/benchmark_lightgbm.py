from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time

from lightgbm import LGBMClassifier


# params for generating data
N_SAMPLES = 1_000_000
N_FEATURES = 50
N_INFORMATIVE = 30
N_REDUNDANT = 5
N_REPEATED = 0
N_CLASSES = 10
N_CLUSTERS_PER_CLASS = 2
CLASS_SEP = 1
FLIP_Y = 0.01
TEST_SIZE = 0.3
RANDOM_STATE = 1

# params for lightgbm
N_ESTIMATORS = 100
NUM_LEAVES = 64
MAX_DEPTH = 5
LEARNING_RATE = 0.1


print(f"creating {N_CLASSES}-class classification dataset...")
X, y = make_classification(
    n_samples=N_SAMPLES,
    n_features=N_FEATURES,
    n_informative=N_INFORMATIVE,
    n_redundant=N_REDUNDANT,
    n_repeated=N_REPEATED,
    n_classes=N_CLASSES,
    n_clusters_per_class=N_CLUSTERS_PER_CLASS,
    class_sep=CLASS_SEP,
    flip_y=FLIP_Y,
    random_state=RANDOM_STATE,
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

print(f"training classifier...")
clf = LGBMClassifier(
    n_estimators=N_ESTIMATORS,
    num_leaves=NUM_LEAVES,
    max_depth=MAX_DEPTH,
    learning_rate=LEARNING_RATE,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
start = time.time()
clf.fit(X_train, y_train)
elapsed = time.time() - start
y_pred = clf.predict(X_test)
print(f"test accuracy: {accuracy_score(y_test,y_pred)}")
print(f"completed in {elapsed:.5f} seconds")

