import numpy as np
import mne
import joblib
from mne.datasets import eegbci

# --- 1. Load the Saved Model and Test Subjects List ---
print("Loading the trained model and the list of test subjects...")
model = joblib.load('optimised_bci_model.pkl')
test_subjects = np.load('test_subjects.npy')

# --- 2. Evaluate Model on Each Test Subject ---
test_accuracies = []
print("Evaluating model performance on the hold-out test set...")
for subject_id in test_subjects:
    try:
        # Load and process the test subject's data exactly as before
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
        test_epochs = mne.concatenate_epochs([epochs_lr, epochs_f], verbose=False)

        # Get data and use the loaded model to score
        data = test_epochs.get_data()
        labels = test_epochs.events[:, -1]
        accuracy = model.score(data, labels)
        test_accuracies.append(accuracy)
        print(f"  Accuracy for unseen subject {subject_id}: {accuracy:.4f}")

    except Exception as e:
        print(f"  Skipping subject {subject_id} due to error: {e}")

# --- 3. Report Final Results ---
print("\n--- Final Model Generalization Performance ---")
if test_accuracies:
    overall_mean = np.mean(test_accuracies)
    overall_std = np.std(test_accuracies)
    print(f"Mean accuracy across {len(test_accuracies)} unseen subjects: {overall_mean:.4f}")
    print(f"Standard deviation: {overall_std:.4f}")