import librosa
import numpy as np
from librosa.feature import spectral_centroid, zero_crossing_rate, mfcc, chroma_cqt
from librosa.core import estimate_tuning
from scipy.signal import find_peaks

class MusicAnalyzer:
    def __init__(self):
        self.audio = None
        self.sr = None
        self.dominant_pitch = None
        self.tempo = None
        self.beat_frames = None
        self.avg_loudness = None
        self.melody = None
        self.spec_centroid = None
        self.zcr = None
        self.mfcc_features = None
        self.avg_mfccs = None
        self.tuning_offset = None
        self.key = None
        self.harmonic_to_percussive_ratio = None
        self.chroma_features = None
        self.melody_summary = None

    def load_audio(self, audio_file_path, sr=16000):
        """
        Load an audio file.
        """
        self.audio, self.sr = librosa.load(audio_file_path, sr=sr, mono=True)

    def extract_pitch_features(self):
        """
        Extract pitch features using librosa.
        """
        pitches, magnitudes = librosa.piptrack(y=self.audio, sr=self.sr)
        self.dominant_pitch = np.mean(pitches[magnitudes > np.median(magnitudes)])  # Mean of significant pitches

    def extract_tempo_features(self):
        """
        Extract tempo and rhythm features using librosa.
        """
        self.tempo, self.beat_frames = librosa.beat.beat_track(y=self.audio, sr=self.sr)

    def extract_loudness_features(self):
        """
        Extract loudness (RMS energy) features.
        """
        rms = librosa.feature.rms(y=self.audio)
        self.avg_loudness = np.mean(rms)

    def extract_melody(self):
        """
        Extract the melody using librosa’s harmonic-percussive source separation.
        """
        harmonic, _ = librosa.effects.hpss(self.audio)
        pitches, magnitudes = librosa.piptrack(y=harmonic, sr=self.sr)
        self.melody = np.max(pitches, axis=0)  # Maximum pitch at each frame
        self.melody[self.melody == 0] = np.nan  # Replace zeros with NaN for better analysis

    def extract_spectral_features(self):
        """
        Extract spectral features like centroid and zero-crossing rate.
        """
        self.spec_centroid = spectral_centroid(y=self.audio, sr=self.sr).mean()
        self.zcr = zero_crossing_rate(y=self.audio).mean()

    def extract_mfcc_features(self, n_mfcc=13):
        """
        Extract Mel-frequency cepstral coefficients (MFCCs).
        """
        self.mfcc_features = mfcc(y=self.audio, sr=self.sr, n_mfcc=n_mfcc)
        self.avg_mfccs = np.mean(self.mfcc_features, axis=1)  # Average MFCCs across time

    def extract_tuning(self):
        """
        Estimate the tuning offset of the audio.
        """
        self.tuning_offset = estimate_tuning(y=self.audio, sr=self.sr)

    def extract_key_and_harmonic_features(self):
        """
        Estimate the musical key using chroma features and harmonic/percussive balance.
        """
        harmonic, _ = librosa.effects.hpss(self.audio)
        chroma = chroma_cqt(y=harmonic, sr=self.sr)
        self.chroma_features = np.mean(chroma, axis=1)

        # Map chroma to key
        key_idx = np.argmax(self.chroma_features)
        major_keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        minor_keys = ['a', 'a#', 'b', 'c', 'c#', 'd', 'd#', 'e', 'f', 'f#', 'g', 'g#']

        # Estimate key and mode
        if np.mean(self.chroma_features[:6]) > np.mean(self.chroma_features[6:]):  # Rough heuristic
            self.key = major_keys[key_idx] + " Major"
        else:
            self.key = minor_keys[key_idx] + " Minor"

        # Harmonic-to-Percussive Ratio
        percussive = librosa.effects.hpss(self.audio)[1]
        self.harmonic_to_percussive_ratio = np.sum(harmonic**2) / (np.sum(percussive**2) + 1e-6)

    def summarize_melody(self):
        """
        Summarize the melody with key statistics, peaks, and patterns.
        """
        if self.melody is None:
            raise ValueError("Melody has not been extracted. Please run extract_melody() first.")

        # Remove NaN values for analysis
        valid_melody = self.melody[~np.isnan(self.melody)]

        if valid_melody.size == 0:
            return "No valid melody data available."

        # Key statistics
        mean_pitch = np.mean(valid_melody)
        median_pitch = np.median(valid_melody)
        std_dev_pitch = np.std(valid_melody)
        min_pitch = np.min(valid_melody)
        max_pitch = np.max(valid_melody)

        # Detect peaks
        peaks, _ = find_peaks(valid_melody, height=np.mean(valid_melody))
        peak_values = valid_melody[peaks]

        # Group pitches into clusters
        bins = [200, 800, 1600, 3000, np.inf]
        cluster_labels = ["Low", "Mid", "High", "Very High"]
        pitch_clusters = np.digitize(valid_melody, bins)
        cluster_summary = {
            cluster_labels[i]: np.sum(pitch_clusters == (i + 1)) / len(valid_melody) * 100
            for i in range(len(cluster_labels))
        }

        # Build summary
        summary = {
            "Statistics": {
                "Mean Pitch (Hz)": mean_pitch,
                "Median Pitch (Hz)": median_pitch,
                "Standard Deviation (Hz)": std_dev_pitch,
                "Min Pitch (Hz)": min_pitch,
                "Max Pitch (Hz)": max_pitch,
            },
            "Peaks": peak_values.tolist(),
            "Clusters": cluster_summary,
        }

        self.melody_summary = summary


def main():
    music = MusicAnalyzer()
    music.load_audio('audio.wav')
    music.extract_pitch_features()
    music.extract_tempo_features()
    music.extract_loudness_features()
    music.extract_melody()
    music.extract_spectral_features()
    music.extract_mfcc_features()
    music.extract_tuning()
    music.extract_key_and_harmonic_features()
    music.summarize_melody()

    print(f'Dominant pitch: {music.dominant_pitch}')
    print(f'Tempo: {music.tempo} beats per minute')
    print(f'Beat frames: {music.beat_frames}')
    print(f'Average loudness: {music.avg_loudness} RMS energy')
    print(f'Melody summary: {music.melody_summary}')
    print(f'Spectral centroid: {music.spec_centroid}')
    print(f'Zero-crossing rate: {music.zcr}')
    print(f'MFCC features: {music.mfcc_features}')
    print(f'Average MFCCs: {music.avg_mfccs}')
    print(f'Tuning offset: {music.tuning_offset} Hz')
    print(f'Key: {music.key}')
    print(f'Harmonic-to-Percussive Ratio: {music.harmonic_to_percussive_ratio}')
    print(f'Chroma Features: {music.chroma_features}')

if __name__ == '__main__':
    main()
