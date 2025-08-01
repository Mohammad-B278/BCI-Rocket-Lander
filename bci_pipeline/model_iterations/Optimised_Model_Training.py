import numpy as np
import mne
import random
import joblib
from mne.datasets import eegbci
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

# --- 1. Split Subjects into Training and Test Sets ---
print("Splitting subjects into training and testing groups...")
all_subjects = list(range(1, 110))
random.seed(42) # Set a seed for reproducible shuffling
random.shuffle(all_subjects)

split_point = int(0.8 * len(all_subjects))
train_subjects = all_subjects[:split_point]
test_subjects = all_subjects[split_point:]

# Save the test subject list for the evaluation script
np.save('test_subjects.npy', test_subjects)
print(f"Full training set: {len(train_subjects)} subjects.")
print(f"Final test set: {len(test_subjects)} subjects (saved to 'test_subjects.npy').")

# --- 2. Load and Process Data for the FULL Training Group ---
all_training_epochs = []
print(f"\nLoading and processing data for all {len(train_subjects)} training subjects...")

for subject_id in train_subjects: # ***CORRECTION: Loop through the full training set***
    try:
        runs_lr = [4, 8, 12]
        runs_f = [6, 10, 14]
        fnames_lr = eegbci.load_data(subject_id, runs=runs_lr, verbose=False)
        fnames_f = eegbci.load_data(subject_id, runs=runs_f, verbose=False)

        raw_lr = mne.concatenate_raws([mne.io.read_raw_edf(f, preload=True, verbose=False) for f in fnames_lr])
        raw_f = mne.concatenate_raws([mne.io.read_raw_edf(f, preload=True, verbose=False) for f in fnames_f])

        def process_and_epoch(raw, event_id_map, event_id_labels):
            raw.filter(l_freq=8., h_freq=35., verbose=False)
            events, _ = mne.events_from_annotations(raw, event_id=event_id_map, verbose=False)
            epochs = mne.Epochs(raw, events, event_id_labels, tmin=-0.5, tmax=3.5, preload=True,
                                baseline=None, picks='eeg', verbose=False)
            epochs.resample(160., verbose=False)
            return epochs

        epochs_lr = process_and_epoch(raw_lr, {'T1': 1, 'T2': 2}, {'left_fist': 1, 'right_fist': 2})
        epochs_f = process_and_epoch(raw_f, {'T2': 2}, {'both_feet': 2})

        all_training_epochs.append(mne.concatenate_epochs([epochs_lr, epochs_f], verbose=False))
        print(f"  Successfully processed subject {subject_id}.")

    except Exception as e:
        print(f"  Skipping subject {subject_id} due to error: {e}")

# --- 3. Train and Save the FINAL OPTIMIZED Model ---
if all_training_epochs:
    print("\nTraining the final, optimized model on all collected data...")
    final_training_epochs = mne.concatenate_epochs(all_training_epochs, verbose=False)

    labels = final_training_epochs.events[:, -1]
    data = final_training_epochs.get_data(copy=False)

    # Define the final pipeline using the best parameters
    svm = SVC(C=10, kernel='rbf', gamma='scale')
    final_pipeline = Pipeline([
        ('Covariances', Covariances(estimator='lwf')),
        ('TangentSpace', TangentSpace(metric='riemann')),
        ('Classifier', svm)
    ])

    # *** THE MISSING STEP: Train the pipeline on your data ***
    print("Fitting the final model...")
    final_pipeline.fit(data, labels)
    print("Fitting complete.")

    # Now, save the FITTED model
    joblib.dump(final_pipeline, 'optimised_bci_model.pkl')
    print(f"\n✅ Optimised model has been trained and saved as 'optimised_bci_model.pkl'.")
else:
    print("\n❌ No data was processed, model could not be trained.")
