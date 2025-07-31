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

# --- 2. Load and Process Data for the Debug Group ---
all_training_epochs = []
print("\nLoading and processing data for training...")

# Define the target sampling frequency
TARGET_SFREQ = 160.

for subject_id in train_subjects:
    try:
        # T1 -> left fist, T2 -> right fist
        runs_lr = [4, 8, 12]
        # T1 -> both fists, T2 -> both feet
        runs_f = [6, 10, 14]

        # Get file paths
        fnames_lr = eegbci.load_data(subject_id, runs=runs_lr, verbose=False)
        fnames_f = eegbci.load_data(subject_id, runs=runs_f, verbose=False)

        # Concatenate raw files
        raw_lr = mne.concatenate_raws([mne.io.read_raw_edf(f, preload=True, verbose=False) for f in fnames_lr])
        raw_f = mne.concatenate_raws([mne.io.read_raw_edf(f, preload=True, verbose=False) for f in fnames_f])

        # Standardize channel names to a common montage
        eegbci.standardize(raw_lr)
        eegbci.standardize(raw_f)
        montage = mne.channels.make_standard_montage('standard_1005')
        raw_lr.set_montage(montage, on_missing='ignore')
        raw_f.set_montage(montage, on_missing='ignore')

        # Define a unified processing function
        def process_and_epoch(raw, event_map):
            # Apply common filters
            raw.filter(l_freq=8., h_freq=35., verbose=False)
            
            # Extract events using the provided map
            events, event_id = mne.events_from_annotations(raw, event_id=event_map, verbose=False)
            
            # Create epochs
            epochs = mne.Epochs(raw, events, event_id, tmin=-0.5, tmax=3.5, preload=True,
                                baseline=None, picks='eeg', verbose=False)
            
            # RESAMPLE to ensure consistency
            epochs.resample(TARGET_SFREQ, verbose=False)
            return epochs

        # Define event maps based on the task runs
        # Run 4, 8, 12: T1=left fist, T2=right_fist
        event_map_lr = {'T1': 1, 'T2': 2}
        event_id_lr = {'left_fist': 1, 'right_fist': 2}
        
        # Run 6, 10, 14: T1=both fists, T2=both feet
        event_map_f = {'T1': 3, 'T2': 4}
        event_id_f = {'both_fists': 3, 'both_feet': 4}

        # Process the two sets of runs
        epochs_lr = process_and_epoch(raw_lr, event_map_lr)
        epochs_lr.event_id = event_id_lr # Assign the correct labels
        
        epochs_f = process_and_epoch(raw_f, event_map_f)
        epochs_f.event_id = event_id_f # Assign the correct labels
        
        # Combine epochs for this subject (we'll keep all 4 classes for now)
        # The OneVsRestClassifier will handle the multi-class problem
        combined_epochs = mne.concatenate_epochs([epochs_lr, epochs_f], verbose=False)
        all_training_epochs.append(combined_epochs)

        print(f"  Successfully processed subject {subject_id}.")

    except Exception as e:
        print(f"  Skipping subject {subject_id} due to error: {e}")


# --- 3. Train and Save the Final Model ---
if all_training_epochs:
    print("\nTraining the final model on all collected training data...")
    # Combine all epochs from all training subjects
    final_training_epochs = mne.concatenate_epochs(all_training_epochs, verbose=False)

    # We are interested in 3 classes: left, right, both_fists
    # Let's drop the 'both_feet' class for our 3-command lander game
    final_training_epochs = final_training_epochs["left_fist", "right_fist", "both_fists"]

    # Extract data and labels
    labels = final_training_epochs.events[:, -1]
    data = final_training_epochs.get_data(copy=False) # Use copy=False for memory efficiency

    # Define the classification pipeline
    # Use 8 CSP components for better spatial filtering in a multi-class scenario
    csp = CSP(n_components=8, reg=None, log=True, norm_trace=False)
    lda = LDA()
    
    # The pipeline that will be wrapped by OneVsRestClassifier
    clf_pipeline = Pipeline([('CSP', csp), ('LDA', lda)])

    # Use OneVsRestClassifier to handle the 3-class problem
    ovr_pipeline = OneVsRestClassifier(clf_pipeline)
    
    # Fit the model ONCE on the entire training dataset
    ovr_pipeline.fit(data, labels)

    # Save the trained model
    model_filename = 'bci_model.pkl'
    joblib.dump(ovr_pipeline, model_filename)
    print(f"\n✅ Model has been trained and saved as '{model_filename}'.")
else:
    print("\n❌ No data was processed, model could not be trained.")