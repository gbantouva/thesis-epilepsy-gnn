EEG Preprocessing

The raw EEG recordings were obtained from the Temple University Hospital (TUH) EEG Epilepsy Corpus, version 2.0.1 (TUEP), which contains annotated sessions from subjects with epilepsy and matched non-epileptic controls. Preprocessing was performed in Python using the MNE-Python library (Gramfort et al., 2013), following established practices in clinical neurophysiology and computational neuroscience to ensure both signal quality and reproducibility.

Channel selection and montage

From each European Data Format (EDF) file, only EEG channels were retained, excluding non-EEG sensors such as EKG or EMG. Channel labels were standardized (e.g., EEG Fp1-LE was mapped to Fp1) to enforce consistency across subjects and sessions. A subset of electrodes corresponding to the international 10–20 system was selected (Jasper, 1958), ensuring comparability between recordings. The standard 10–20 montage was applied to provide canonical electrode coordinates. Electrodes T1 and T2, present in the TUH corpus but absent from the canonical system, were preserved by assigning approximate positions at the temporal mastoids, consistent with common clinical practice.

Referencing and filtering

All signals were re-referenced to the common average reference (CAR), a method that minimizes reference bias and enhances spatial resolution in EEG (Ludwig et al., 2009). To attenuate powerline interference, a 60 Hz notch filter was applied. Subsequently, the data were bandpass filtered between 0.5 Hz and 40 Hz, which preserved the main frequency ranges of interest for epileptiform activity (delta, theta, alpha, beta, and lower gamma) while removing low-frequency drifts and high-frequency noise (Urigüen & Garcia-Zapirain, 2015).

Artifact reduction with Independent Component Analysis

Independent Component Analysis (ICA; Bell & Sejnowski, 1995) was fitted to the continuous EEG for each session to facilitate the identification of common artifacts, including ocular movements, cardiac activity, and muscle noise. Although ICA components were not automatically removed in this preprocessing stage, the decomposition provided a basis for optional manual or automated artifact rejection in downstream analyses. ICA is widely regarded as a robust approach for isolating non-neural sources from scalp EEG (Urigüen & Garcia-Zapirain, 2015).

Resampling and initial cropping

To standardize temporal resolution and reduce computational load, signals were resampled to 200 Hz, which is sufficient to capture epileptiform transients while improving efficiency for graph-based learning. For non-epileptic recordings, the first 10 seconds were cropped to minimize the influence of electrode placement artifacts and initial recording instabilities, which are often observed at the beginning of clinical EEG sessions.

Epoching and artifact rejection

Continuous EEG was segmented into non-overlapping 2-second epochs. This epoch length is widely used in the analysis of interictal epileptiform discharges, as it balances temporal precision with the need for stable spectral estimation (Zhao et al., 2023). To reduce the influence of high-amplitude noise, an adaptive artifact rejection scheme was applied: the peak-to-peak amplitude of each epoch was computed, and epochs exceeding the 95th percentile of the amplitude distribution for that session were discarded. This percentile-based approach adapts to inter-subject variability and avoids applying a fixed threshold that may bias results across heterogeneous recordings.

Normalization

For each session, the remaining epochs were z-score normalized to zero mean and unit variance across channels. This normalization reduces inter-subject and inter-session variability, which is especially important when training machine learning models on large multi-subject corpora (Zhao et al., 2023).

Output data

For each EDF file, the following outputs were generated:

Continuous preprocessed EEG signal

Segmented, cleaned epochs

Binary diagnostic labels (epilepsy vs. non-epilepsy)

Metadata including electrode names, sampling frequency, and preprocessing parameters

These processed data formed the input for subsequent graph-based self-supervised learning experiments.

References

Bell, A. J., & Sejnowski, T. J. (1995). An information-maximization approach to blind separation and blind deconvolution. Neural Computation, 7(6), 1129–1159.

Gramfort, A., et al. (2013). MEG and EEG data analysis with MNE-Python. Frontiers in Neuroscience, 7, 267.

Jasper, H. (1958). The ten-twenty electrode system of the International Federation. Electroencephalography and Clinical Neurophysiology, 10, 371–375.

Ludwig, K. A., Miriani, R. M., Langhals, N. B., Joseph, M. D., Anderson, D. J., & Kipke, D. R. (2009). Using a common average reference to improve cortical neuron recordings. Journal of Neurophysiology, 101(3), 1679–1689.

Urigüen, J. A., & Garcia-Zapirain, B. (2015). EEG artifact removal—state-of-the-art and guidelines. Journal of Neural Engineering, 12(3), 031001.

Zhao, Y., Xu, Y., & Pan, G. (2023). Self-supervised learning for epileptic EEG analysis using graph neural networks. arXiv:2311.03764.