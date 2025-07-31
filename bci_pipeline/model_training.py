import numpy as np
import mne
import random
import joblib
import os
from mne.datasets import eegbci
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.multiclass import OneVsRestClassifier
from mne.decoding import CSP

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

# --- Create a smaller debug group from the training set for a quick test run ---
num_debug_subjects = int(len(train_subjects) * 0.1)
if num_debug_subjects == 0:
    num_debug_subjects = 1 
debug_subjects = train_subjects[:num_debug_subjects]
print(f"\nUsing a debug group of {len(debug_subjects)} subjects for this run.")


# --- 2. Load and Process Data for the Debug Group ---
all_training_epochs = []
print("Loading and processing data for the debug group...")

for subject_id in debug_subjects: # Looping through the SMALL debug group
    try:
        runs_lr = [4, 8, 12]
        runs_f = [6, 10, 14]
        fnames_lr = eegbci.load_data(subject_id, runs=runs_lr, verbose=False)
        fnames_f = eegbci.load_data(subject_id, runs=runs_f, verbose=False)

        raw_lr = mne.concatenate_raws([mne.io.read_raw_edf(f, preload=True, verbose=False) for f in fnames_lr])
        raw_f = mne.concatenate_raws([mne.io.read_raw_edf(f, preload=True, verbose=False) for f in fnames_f])

        def process_and_epoch(raw, event_id):
            raw.filter(l_freq=8., h_freq=35., verbose=False)
            raw.notch_filter(freqs=50, verbose=False)
            events, _ = mne.events_from_annotations(raw, event_id={'T1': 1, 'T2': 2}, verbose=False)
            epochs = mne.Epochs(raw, events, event_id, tmin=-1., tmax=4., preload=True, baseline=None, picks='eeg', verbose=False)
            return epochs

        epochs_lr = process_and_epoch(raw_lr, event_id={'left_fist': 1, 'right_fist': 2})
        epochs_f = process_and_epoch(raw_f, event_id={'feet': 2})
        all_training_epochs.append(mne.concatenate_epochs([epochs_lr, epochs_f], verbose=False))
        print(f"  Successfully processed subject {subject_id}.")
    except Exception as e:
        print(f"  Skipping subject {subject_id} due to error: {e}")

# --- 3. Train and Save the Model (on the debug data) ---
if all_training_epochs:
    print("\nTraining the final model on the collected debug data...")
    # Combine all epochs from all training subjects
    final_training_epochs = mne.concatenate_epochs(all_training_epochs, verbose=False)

    labels = final_training_epochs.events[:, -1]
    data = final_training_epochs.get_data()

    # Define the pipeline
    csp = CSP(n_components=4, reg=None, log=True)
    lda = LDA()
    ovr_pipeline = OneVsRestClassifier(Pipeline([('CSP', csp), ('LDA', lda)]))

    # Fit the model ONCE on the entire training dataset
    ovr_pipeline.fit(data, labels)

    # Save the trained model
    joblib.dump(ovr_pipeline, 'final_bci_model.pkl')
    print("Debug model has been trained and saved as 'final_bci_model.pkl'.")
else:
    print("No data was processed, model could not be trained.")