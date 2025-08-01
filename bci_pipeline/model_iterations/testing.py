import joblib
from sklearn.utils.validation import check_is_fitted
from sklearn.pipeline import Pipeline
from mne.decoding import CSP
from sklearn.svm import SVC

print("Loading model from 'optimised_bci_model.pkl'...")
model = joblib.load('optimised_bci_model.pkl')

try:
    # This function from scikit-learn is designed to check if a model is fitted.
    # It will raise an error if the model is not fitted.
    check_is_fitted(model)
    print("\n✅ SUCCESS: The saved model is correctly fitted.")
except Exception as e:
    print(f"\n❌ FAILURE: The saved model is NOT fitted. The error is: {e}")