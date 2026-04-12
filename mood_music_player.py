"""
╔══════════════════════════════════════════════════════════════╗
║         ML-POWERED MOOD MUSIC PLAYER                        ║
║         100% Open Source — No API Key Needed                ║
╠══════════════════════════════════════════════════════════════╣
║  HOW IT WORKS:                                              ║
║  1. You provide sample MP3s in subfolders per mood          ║
║  2. The app extracts audio features (MFCC, tempo, energy)   ║
║  3. Trains a Random Forest classifier on those features     ║
║  4. Classifies your entire music library automatically      ║
║  5. GUI lets you pick a mood → playlist stays in that mood  ║
╠══════════════════════════════════════════════════════════════╣
║  FOLDER STRUCTURE EXPECTED:                                 ║
║    samples/                                                 ║
║      happy/      ← 6-10 MP3s                               ║
║      sad/        ← 6-10 MP3s                               ║
║      energetic/  ← 6-10 MP3s                               ║
║      calm/       ← 6-10 MP3s                               ║
║      romantic/   ← 6-10 MP3s                               ║
║      angry/      ← 6-10 MP3s                               ║
╠══════════════════════════════════════════════════════════════╣
║  INSTALL:                                                   ║
║    pip install -r requirements.txt                          ║
║  RUN:                                                       ║
║    python mood_music_player.py                              ║
╚══════════════════════════════════════════════════════════════╝
"""

# ── Standard library ─────────────────────────────────────────
import os
import json
import random
import threading
import warnings
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Third-party ───────────────────────────────────────────────
try:
    import numpy as np
    import librosa
    import joblib
    import pygame
    from mutagen.mp3 import MP3
    from mutagen.id3 import ID3, TIT2, TPE1
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import classification_report
except ImportError as e:
    print(f"\n[ERROR] Missing library: {e}")
    print("Run:  pip install librosa scikit-learn pygame mutagen joblib\n")
    exit(1)

# ────────────────────────────────────────────────────────────
#  CONSTANTS
# ────────────────────────────────────────────────────────────

MOODS = ["Happy", "Sad", "Energetic", "Calm", "Romantic", "Angry"]

MOOD_COLORS = {
    "Happy":     "#F9C74F",
    "Sad":       "#74B3CE",
    "Energetic": "#F94144",
    "Calm":      "#90BE6D",
    "Romantic":  "#F4A5C3",
    "Angry":     "#E05C5C",
    "Unknown":   "#888888",
}

MOOD_EMOJI = {
    "Happy":     "😊",
    "Sad":       "😢",
    "Energetic": "⚡",
    "Calm":      "🌿",
    "Romantic":  "❤️",
    "Angry":     "🔥",
    "Unknown":   "❓",
}

MODEL_FILE  = "mood_model.pkl"
CACHE_FILE  = "mood_cache.json"
FEATURE_DIM = 193


# ════════════════════════════════════════════════════════════
#  AUDIO FEATURE EXTRACTOR
# ════════════════════════════════════════════════════════════

def extract_features(filepath: str, duration: float = 30.0):
    """
    Extract 193 numerical audio features from an MP3:
      - 40 MFCCs  mean+std  → 80  (timbre / tone colour)
      - 12 Chroma mean+std  → 24  (harmonic / pitch content)
      -  7 Spectral contrast mean+std → 14  (brightness peaks)
      -  6 Tonnetz mean+std → 12  (tonal centroid / chord feel)
      -  1 Zero-crossing rate mean+std → 2  (noisiness)
      -  1 Spectral centroid mean+std  → 2  (brightness)
      -  1 RMS energy mean+std         → 2  (loudness)
      -  1 Tempo (BPM)                 → 1  (speed)
    All padded/trimmed to FEATURE_DIM=193.
    """
    try:
        y, sr = librosa.load(filepath, mono=True, duration=duration, sr=22050)
        if len(y) < sr * 2:
            return None

        feats = []

        # 1. MFCCs — timbral texture (most discriminative for mood)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        feats.extend(np.mean(mfcc, axis=1))
        feats.extend(np.std(mfcc,  axis=1))

        # 2. Chroma — harmonic / pitch class content
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        feats.extend(np.mean(chroma, axis=1))
        feats.extend(np.std(chroma,  axis=1))

        # 3. Spectral contrast — peaks vs valleys in spectrum
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        feats.extend(np.mean(contrast, axis=1))
        feats.extend(np.std(contrast,  axis=1))

        # 4. Tonnetz — tonal centroid (chord/harmony detection)
        harmonic = librosa.effects.harmonic(y)
        tonnetz  = librosa.feature.tonnetz(y=harmonic, sr=sr)
        feats.extend(np.mean(tonnetz, axis=1))
        feats.extend(np.std(tonnetz,  axis=1))

        # 5. Zero-crossing rate — noisiness / percussiveness
        zcr = librosa.feature.zero_crossing_rate(y)
        feats.append(float(np.mean(zcr)))
        feats.append(float(np.std(zcr)))

        # 6. Spectral centroid — perceived brightness
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        feats.append(float(np.mean(centroid)))
        feats.append(float(np.std(centroid)))

        # 7. RMS energy — overall loudness
        rms = librosa.feature.rms(y=y)
        feats.append(float(np.mean(rms)))
        feats.append(float(np.std(rms)))

        # 8. Tempo — beats per minute
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        feats.append(float(np.atleast_1d(tempo)[0]))

        vec = np.array(feats, dtype=np.float32)
        if len(vec) < FEATURE_DIM:
            vec = np.pad(vec, (0, FEATURE_DIM - len(vec)))
        else:
            vec = vec[:FEATURE_DIM]

        return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)

    except Exception as e:
        print(f"  [feature error] {Path(filepath).name}: {e}")
        return None


# ════════════════════════════════════════════════════════════
#  ML TRAINER
# ════════════════════════════════════════════════════════════

class MoodTrainer:
    """
    Trains a Random Forest on labeled MP3 samples.
    Pipeline: StandardScaler → RandomForestClassifier (300 trees)
    Saves model to mood_model.pkl for reuse.
    """

    def __init__(self):
        self.model         = None
        self.label_encoder = LabelEncoder()
        self.is_trained    = False

    def train(self, samples_root: str, progress_cb=None) -> dict:
        X, y = [], []
        samples_root = Path(samples_root)

        for mood_dir in sorted(samples_root.iterdir()):
            if not mood_dir.is_dir():
                continue
            mood_label = mood_dir.name.capitalize()
            if mood_label not in MOODS:
                continue
            mp3_files = list(mood_dir.glob("*.mp3"))
            if not mp3_files:
                continue

            for i, fp in enumerate(mp3_files):
                if progress_cb:
                    progress_cb(
                        f"Extracting: {mood_label} [{i+1}/{len(mp3_files)}] {fp.name}")
                vec = extract_features(str(fp))
                if vec is not None:
                    X.append(vec)
                    y.append(mood_label)

        if len(X) < 6:
            raise ValueError(
                f"Need at least 6 total samples. Got {len(X)}.")

        X     = np.array(X)
        y_enc = self.label_encoder.fit_transform(y)

        # Pipeline: scale features first, then classify
        # Random Forest chosen because it handles small datasets well
        # and gives probability estimates (confidence scores)
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_leaf=1,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            ))
        ])

        # Cross-validation for realistic accuracy estimate
        n_splits = min(5, len(X) // 2)
        cv_accuracy = 0.0
        if n_splits >= 2:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            scores = cross_val_score(self.model, X, y_enc, cv=cv, scoring="accuracy")
            cv_accuracy = float(np.mean(scores))

        # Train on ALL data
        self.model.fit(X, y_enc)
        self.is_trained = True

        y_pred = self.model.predict(X)
        report = classification_report(
            y_enc, y_pred,
            target_names=self.label_encoder.classes_,
            output_dict=True
        )

        found_moods = list(self.label_encoder.classes_)

        joblib.dump({
            "model":         self.model,
            "label_encoder": self.label_encoder,
            "moods":         found_moods,
        }, MODEL_FILE)

        return {
            "cv_accuracy": cv_accuracy,
            "n_samples":   len(X),
            "n_moods":     len(found_moods),
            "moods":       found_moods,
            "report":      report,
        }

    def predict(self, filepath: str) -> tuple:
        if not self.is_trained:
            return "Unknown", 0.0
        vec = extract_features(filepath)
        if vec is None:
            return "Unknown", 0.0
        v2  = vec.reshape(1, -1)
        idx = self.model.predict(v2)[0]
        pr  = self.model.predict_proba(v2)[0]
        return str(self.label_encoder.inverse_transform([idx])[0]), float(np.max(pr))

    def load(self, path: str = MODEL_FILE) -> bool:
        if not os.path.exists(path):
            return False
        try:
            data = joblib.load(path)
            self.model         = data["model"]
            self.label_encoder = data["label_encoder"]
            self.is_trained    = True
            return True
        except Exception:
            return False


# ════════════════════════════════════════════════════════════
#  SONG MODEL
# ════════════════════════════════════════════════════════════

class Song:
    def __init__(self, filepath: str):
        self.filepath   = filepath
        self.title      = Path(filepath).stem
        self.artist     = ""
        self.duration   = 0
        self.mood       = "Unknown"
        self.confidence = 0.0
        self._read_tags()

    def _read_tags(self):
        try:
            self.duration = int(MP3(self.filepath).info.length)
            tags = ID3(self.filepath)
            if TIT2 in tags:
                self.title = str(tags[TIT2])
            if TPE1 in tags:
                self.artist = str(tags[TPE1])
        except Exception:
            pass

    @property
    def display_name(self):
        return f"{self.title} — {self.artist}" if self.artist else self.title

    @property
    def duration_str(self):
        m, s = divmod(self.duration, 60)
        return f"{m}:{s:02d}"


# ════════════════════════════════════════════════════════════
#  PLAYER ENGINE
# ════════════════════════════════════════════════════════════

class MusicPlayer:
    def __init__(self):
        pygame.mixer.init()
        self.songs:         list = []
        self.current_index: int  = -1
        self.is_playing:    bool = False
        self.is_paused:     bool = False

    def filtered(self, mood="All"):
        return self.songs if mood == "All" else [s for s in self.songs if s.mood == mood]

    def play(self, song):
        pygame.mixer.music.load(song.filepath)
        pygame.mixer.music.play()
        self.is_playing    = True
        self.is_paused     = False
        self.current_index = self.songs.index(song)

    def pause_resume(self):
        if self.is_paused:
            pygame.mixer.music.unpause()
        else:
            pygame.mixer.music.pause()
        self.is_paused = not self.is_paused

    def stop(self):
        pygame.mixer.music.stop()
        self.is_playing = False
        self.is_paused  = False

    def next_song(self, mood="All"):
        pool    = self.filtered(mood)
        current = self.songs[self.current_index] if self.current_index >= 0 else None
        choices = [s for s in pool if s != current] or pool
        return random.choice(choices) if choices else None

    def prev_song(self, mood="All"):
        pool = self.filtered(mood)
        if not pool:
            return None
        if self.current_index >= 0:
            cur = self.songs[self.current_index]
            if cur in pool:
                return pool[(pool.index(cur) - 1) % len(pool)]
        return pool[-1]

    @property
    def song_ended(self):
        return self.is_playing and not pygame.mixer.music.get_busy() and not self.is_paused


# ════════════════════════════════════════════════════════════
#  MAIN GUI
# ════════════════════════════════════════════════════════════

class MoodPlayerApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("🎵 ML Mood Music Player")
        self.geometry("880x720")
        self.configure(bg="#1E1E2E")
        self.resizable(True, True)

        self.player      = MusicPlayer()
        self.trainer     = MoodTrainer()
        self.active_mood = tk.StringVar(value="All")
        self.cache       = self._load_cache()
        self.last_report = None

        self._apply_styles()
        self._build_ui()

        if self.trainer.load():
            self._set_status(
                "✅  Model loaded from disk — open your music folder to classify songs.")
        else:
            self._set_status(
                "No trained model found — click 🧠 Train Model first.")

        self._poll()

    # ── CACHE ────────────────────────────────────────────────

    def _load_cache(self):
        try:
            if os.path.exists(CACHE_FILE):
                with open(CACHE_FILE) as f:
                    return json.load(f)
        except Exception:
            pass
        return {}

    def _save_cache(self):
        with open(CACHE_FILE, "w") as f:
            json.dump(self.cache, f, indent=2)

    # ── STYLES ───────────────────────────────────────────────

    def _apply_styles(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure("Mood.Treeview",
                    background="#1E1E2E", foreground="#CDD6F4",
                    fieldbackground="#1E1E2E", rowheight=26,
                    font=("Helvetica", 10))
        s.configure("Mood.Treeview.Heading",
                    background="#313244", foreground="#CDD6F4",
                    font=("Helvetica", 9, "bold"))
        s.map("Mood.Treeview",
              background=[("selected", "#45475A")],
              foreground=[("selected", "#CDD6F4")])
        s.configure("Mood.Horizontal.TProgressbar",
                    troughcolor="#313244", background="#F9C74F", thickness=4)

    # ── BUILD UI ─────────────────────────────────────────────

    def _build_ui(self):

        # Toolbar
        tb = tk.Frame(self, bg="#181825")
        tb.pack(fill="x")

        tk.Label(tb, text="🎵  ML Mood Music Player",
                 font=("Helvetica", 15, "bold"),
                 bg="#181825", fg="#CDD6F4",
                 padx=14, pady=10).pack(side="left")

        for txt, cmd in [
            ("📊 Model Report", self._show_report),
            ("📂 Open Music",   self._open_music),
            ("🧠 Train Model",  self._train_dialog),
        ]:
            tk.Button(tb, text=txt, command=cmd,
                      bg="#313244", fg="#CDD6F4",
                      relief="flat", padx=10, pady=6,
                      font=("Helvetica", 10),
                      cursor="hand2").pack(side="right", padx=4, pady=6)

        # Now-playing card
        card = tk.Frame(self, bg="#313244")
        card.pack(fill="x", padx=14, pady=8)

        self.mood_badge = tk.Label(card, text="🎵",
                                   font=("Helvetica", 28),
                                   bg="#313244", width=3)
        self.mood_badge.pack(side="left", padx=12, pady=10)

        info = tk.Frame(card, bg="#313244")
        info.pack(side="left", fill="both", expand=True, pady=10)

        self.title_lbl = tk.Label(info, text="No song selected",
                                  font=("Helvetica", 13, "bold"),
                                  bg="#313244", fg="#CDD6F4", anchor="w")
        self.title_lbl.pack(fill="x")

        self.detail_lbl = tk.Label(info,
                                   text="Train a model, then open your music folder",
                                   font=("Helvetica", 10),
                                   bg="#313244", fg="#6C7086", anchor="w")
        self.detail_lbl.pack(fill="x")

        self.conf_lbl = tk.Label(card, text="",
                                 font=("Helvetica", 9),
                                 bg="#313244", fg="#6C7086", padx=12)
        self.conf_lbl.pack(side="right", pady=10)

        # Progress
        self.progress = ttk.Progressbar(self, orient="horizontal",
                                        mode="determinate",
                                        style="Mood.Horizontal.TProgressbar")
        self.progress.pack(fill="x", padx=14, pady=(0, 2))
        self.time_lbl = tk.Label(self, text="0:00 / 0:00",
                                 font=("Helvetica", 9),
                                 bg="#1E1E2E", fg="#6C7086")
        self.time_lbl.pack()

        # Controls
        ctrl = tk.Frame(self, bg="#1E1E2E")
        ctrl.pack(pady=6)
        bs = dict(bg="#313244", fg="#CDD6F4", relief="flat",
                  font=("Helvetica", 15), width=3, cursor="hand2",
                  padx=4, pady=4)
        tk.Button(ctrl, text="⏮", command=self._prev, **bs).pack(side="left", padx=3)
        self.play_btn = tk.Button(ctrl, text="▶", command=self._play_pause, **bs)
        self.play_btn.pack(side="left", padx=3)
        tk.Button(ctrl, text="⏹", command=self._stop, **bs).pack(side="left", padx=3)
        tk.Button(ctrl, text="⏭", command=self._next, **bs).pack(side="left", padx=3)

        tk.Label(ctrl, text="🔊", bg="#1E1E2E", fg="#CDD6F4",
                 font=("Helvetica", 12)).pack(side="left", padx=(16, 2))
        vol = tk.Scale(ctrl, from_=0, to=100, orient="horizontal",
                       length=110, bg="#1E1E2E", fg="#CDD6F4",
                       highlightthickness=0, troughcolor="#313244",
                       command=lambda v: pygame.mixer.music.set_volume(int(v)/100))
        vol.set(70)
        vol.pack(side="left")

        # Mood pills
        pr = tk.Frame(self, bg="#1E1E2E")
        pr.pack(pady=(4, 6))
        tk.Label(pr, text="Mood filter:",
                 bg="#1E1E2E", fg="#6C7086",
                 font=("Helvetica", 10)).pack(side="left", padx=(4, 8))

        self.mood_btns = {}
        for mood in ["All"] + MOODS:
            lbl = mood if mood == "All" else f"{MOOD_EMOJI[mood]} {mood}"
            b = tk.Button(pr, text=lbl,
                          bg="#313244", fg="#CDD6F4",
                          relief="flat", padx=8, pady=3,
                          font=("Helvetica", 9), cursor="hand2",
                          command=lambda m=mood: self._filter_mood(m))
            b.pack(side="left", padx=2)
            self.mood_btns[mood] = b
        self._highlight_pill("All")

        # Playlist
        pf = tk.Frame(self, bg="#1E1E2E")
        pf.pack(fill="both", expand=True, padx=14)

        hdr = tk.Frame(pf, bg="#1E1E2E")
        hdr.pack(fill="x", pady=(0, 4))
        tk.Label(hdr, text="PLAYLIST",
                 font=("Helvetica", 9, "bold"),
                 bg="#1E1E2E", fg="#6C7086").pack(side="left")
        self.count_lbl = tk.Label(hdr, text="",
                                  font=("Helvetica", 9),
                                  bg="#1E1E2E", fg="#6C7086")
        self.count_lbl.pack(side="right")

        cols = ("mood", "conf", "title", "dur")
        self.tree = ttk.Treeview(pf, columns=cols,
                                 show="headings", height=14,
                                 style="Mood.Treeview")
        self.tree.heading("mood", text="Mood")
        self.tree.heading("conf", text="Confidence")
        self.tree.heading("title", text="Song")
        self.tree.heading("dur",  text="Duration")
        self.tree.column("mood", width=120, anchor="center")
        self.tree.column("conf", width=90,  anchor="center")
        self.tree.column("title", width=510)
        self.tree.column("dur",  width=72,  anchor="center")

        sb = ttk.Scrollbar(pf, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscroll=sb.set)
        self.tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self.tree.bind("<Double-1>", self._on_dbl)

        # Status bar
        self.status = tk.Label(self, text="",
                               font=("Helvetica", 9),
                               bg="#181825", fg="#6C7086",
                               anchor="w", padx=10, pady=4)
        self.status.pack(fill="x", side="bottom")

    # ── TRAINING DIALOG ───────────────────────────────────────

    def _train_dialog(self):
        folder = filedialog.askdirectory(
            title="Select samples/ root folder (with mood subfolders)")
        if not folder:
            return

        subs = [d.name.capitalize()
                for d in Path(folder).iterdir() if d.is_dir()]
        valid = [m for m in subs if m in MOODS]
        if not valid:
            messagebox.showerror("Wrong folder",
                f"No valid mood subfolders found.\n"
                f"Expected: {', '.join(m.lower() for m in MOODS)}\n"
                f"Found: {', '.join(subs)}")
            return

        win = tk.Toplevel(self)
        win.title("Training Model")
        win.geometry("500x340")
        win.configure(bg="#1E1E2E")
        win.grab_set()

        tk.Label(win, text="🧠  Training ML Model",
                 font=("Helvetica", 13, "bold"),
                 bg="#1E1E2E", fg="#CDD6F4").pack(pady=(18, 6))

        tk.Label(win,
                 text="Extracting audio features (MFCC, chroma, tempo, energy...)\n"
                      "and training a Random Forest classifier on your samples.",
                 font=("Helvetica", 10), bg="#1E1E2E", fg="#6C7086",
                 justify="center").pack()

        prog = ttk.Progressbar(win, mode="indeterminate", length=400)
        prog.pack(pady=12)
        prog.start(12)

        log_var    = tk.StringVar(value="Starting ...")
        result_var = tk.StringVar()

        tk.Label(win, textvariable=log_var,
                 font=("Helvetica", 9), bg="#1E1E2E", fg="#90BE6D",
                 wraplength=460, justify="center").pack(pady=2)
        tk.Label(win, textvariable=result_var,
                 font=("Helvetica", 11, "bold"),
                 bg="#1E1E2E", fg="#F9C74F",
                 wraplength=460, justify="center").pack(pady=6)

        close_btn = tk.Button(win, text="Close", command=win.destroy,
                              bg="#313244", fg="#CDD6F4",
                              relief="flat", padx=20, pady=6,
                              state="disabled")
        close_btn.pack(pady=6)

        def run():
            try:
                report = self.trainer.train(
                    folder,
                    progress_cb=lambda msg: log_var.set(msg))
                self.last_report = report
                acc = report["cv_accuracy"]
                n   = report["n_samples"]
                m   = report["n_moods"]
                result_var.set(
                    f"✅  Done!  {n} samples · {m} moods · "
                    f"CV accuracy: {acc*100:.1f}%")
                self._set_status(
                    f"Model trained — {n} samples, {m} moods, "
                    f"CV accuracy {acc*100:.1f}%")
            except Exception as e:
                result_var.set(f"❌  Error: {e}")
                self._set_status(f"Training failed: {e}")
            finally:
                prog.stop()
                close_btn.config(state="normal")

        threading.Thread(target=run, daemon=True).start()

    # ── REPORT DIALOG ─────────────────────────────────────────

    def _show_report(self):
        if not self.trainer.is_trained:
            messagebox.showinfo("No Model", "Train a model first.")
            return
        if not self.last_report:
            messagebox.showinfo("No Report",
                "Train a model in this session to see the detailed report.")
            return

        r   = self.last_report
        rep = r["report"]
        win = tk.Toplevel(self)
        win.title("Model Report")
        win.geometry("500x440")
        win.configure(bg="#1E1E2E")

        tk.Label(win, text="📊  Model Performance Report",
                 font=("Helvetica", 13, "bold"),
                 bg="#1E1E2E", fg="#CDD6F4").pack(pady=(16, 4))
        tk.Label(win,
                 text=f"Samples: {r['n_samples']}   "
                      f"Moods: {r['n_moods']}   "
                      f"CV Accuracy: {r['cv_accuracy']*100:.1f}%",
                 font=("Helvetica", 10),
                 bg="#1E1E2E", fg="#F9C74F").pack(pady=(0, 10))

        tbl = tk.Frame(win, bg="#313244")
        tbl.pack(padx=20, pady=4, fill="x")

        headers = ["Mood", "Precision", "Recall", "F1", "Samples"]
        widths  = [13, 10, 10, 10, 9]
        for c, (h, w) in enumerate(zip(headers, widths)):
            tk.Label(tbl, text=h, font=("Helvetica", 9, "bold"),
                     bg="#313244", fg="#CDD6F4",
                     width=w).grid(row=0, column=c, padx=4, pady=4)

        row_bg = ["#1E1E2E", "#252535"]
        for ri, mood in enumerate(MOODS):
            if mood not in rep:
                continue
            d = rep[mood]
            bg = row_bg[ri % 2]
            for ci, (v, w) in enumerate(zip(
                [f"{MOOD_EMOJI.get(mood,'')} {mood}",
                 f"{d['precision']:.2f}",
                 f"{d['recall']:.2f}",
                 f"{d['f1-score']:.2f}",
                 str(int(d['support']))],
                widths
            )):
                tk.Label(tbl, text=v, font=("Helvetica", 10),
                         bg=bg, fg="#CDD6F4",
                         width=w).grid(row=ri+1, column=ci, padx=4, pady=3)

        wa = rep.get("weighted avg", {})
        tk.Label(win,
                 text=f"Weighted avg — P: {wa.get('precision',0):.2f}  "
                      f"R: {wa.get('recall',0):.2f}  "
                      f"F1: {wa.get('f1-score',0):.2f}",
                 font=("Helvetica", 9),
                 bg="#1E1E2E", fg="#6C7086").pack(pady=8)
        tk.Label(win,
                 text="CV accuracy is the realistic estimate.\n"
                      "Training-set numbers will be higher.",
                 font=("Helvetica", 9),
                 bg="#1E1E2E", fg="#6C7086",
                 justify="center").pack()

    # ── OPEN MUSIC FOLDER ─────────────────────────────────────

    def _open_music(self):
        if not self.trainer.is_trained:
            messagebox.showwarning("No Model",
                "Please train the model first (🧠 Train Model).\n"
                "Point it at your samples/ folder.")
            return

        folder = filedialog.askdirectory(title="Select your music folder")
        if not folder:
            return

        mp3s = list(Path(folder).glob("*.mp3"))
        if not mp3s:
            messagebox.showinfo("No MP3s", "No MP3 files found.")
            return

        self._set_status(
            f"Found {len(mp3s)} MP3s — classifying with ML model ...")
        self.tree.delete(*self.tree.get_children())
        self.player.songs = []

        def run():
            songs = []
            for i, fp in enumerate(mp3s):
                song = Song(str(fp))
                key  = song.filepath
                if key in self.cache:
                    song.mood       = self.cache[key]["mood"]
                    song.confidence = self.cache[key]["conf"]
                else:
                    self.after(0, lambda t=song.title, n=i:
                        self._set_status(
                            f"Classifying [{n+1}/{len(mp3s)}]: {t}"))
                    mood, conf      = self.trainer.predict(key)
                    song.mood       = mood
                    song.confidence = conf
                    self.cache[key] = {"mood": mood, "conf": conf}
                songs.append(song)

            self._save_cache()
            self.player.songs = songs
            self.after(0, lambda: self._refresh_playlist(
                self.active_mood.get()))

        threading.Thread(target=run, daemon=True).start()

    # ── PLAYLIST ─────────────────────────────────────────────

    def _refresh_playlist(self, mood="All"):
        self.tree.delete(*self.tree.get_children())
        songs = self.player.filtered(mood)
        for s in songs:
            conf = f"{s.confidence*100:.0f}%" if s.confidence else "—"
            self.tree.insert("", "end", iid=s.filepath,
                             values=(f"{MOOD_EMOJI.get(s.mood,'❓')} {s.mood}",
                                     conf, s.display_name, s.duration_str))
        n = len(songs)
        self.count_lbl.config(text=f"{n} songs")
        suffix = f" · mood: {mood}" if mood != "All" else ""
        self._set_status(f"Showing {n} songs{suffix}")

    def _filter_mood(self, mood):
        self.active_mood.set(mood)
        self._highlight_pill(mood)
        self._refresh_playlist(mood)

    def _highlight_pill(self, active):
        for mood, btn in self.mood_btns.items():
            if mood == active:
                btn.config(bg=MOOD_COLORS.get(mood, "#F9C74F"), fg="#1E1E2E")
            else:
                btn.config(bg="#313244", fg="#CDD6F4")

    def _on_dbl(self, _):
        sel = self.tree.selection()
        if not sel:
            return
        song = next((s for s in self.player.songs if s.filepath == sel[0]), None)
        if song:
            self._play_song(song)

    # ── PLAYBACK ─────────────────────────────────────────────

    def _play_song(self, song):
        self.player.play(song)
        self.play_btn.config(text="⏸")
        color = MOOD_COLORS.get(song.mood, "#CDD6F4")
        self.mood_badge.config(text=MOOD_EMOJI.get(song.mood, "🎵"))
        self.title_lbl.config(text=song.title, fg=color)
        self.detail_lbl.config(
            text=f"{song.artist or 'Unknown artist'}  ·  {song.mood}  ·  {song.duration_str}")
        self.conf_lbl.config(
            text=f"Confidence: {song.confidence*100:.0f}%" if song.confidence else "")
        self.progress.config(maximum=max(song.duration, 1))
        self._set_status(f"Playing: {song.display_name}")
        if self.tree.exists(song.filepath):
            self.tree.selection_set(song.filepath)
            self.tree.see(song.filepath)

    def _play_pause(self):
        if not self.player.songs:
            return
        if not self.player.is_playing:
            s = self.player.next_song(self.active_mood.get())
            if s:
                self._play_song(s)
        else:
            self.player.pause_resume()
            self.play_btn.config(
                text="▶" if self.player.is_paused else "⏸")

    def _stop(self):
        self.player.stop()
        self.play_btn.config(text="▶")
        self.progress["value"] = 0
        self.time_lbl.config(text="0:00 / 0:00")

    def _next(self):
        s = self.player.next_song(self.active_mood.get())
        if s:
            self._play_song(s)

    def _prev(self):
        s = self.player.prev_song(self.active_mood.get())
        if s:
            self._play_song(s)

    def _set_status(self, msg):
        self.status.config(text=f"  {msg}")

    # ── POLL LOOP ─────────────────────────────────────────────

    def _poll(self):
        if self.player.song_ended:
            self._next()

        if self.player.is_playing and not self.player.is_paused:
            pos = pygame.mixer.music.get_pos()
            if pos >= 0 and self.player.current_index >= 0:
                song  = self.player.songs[self.player.current_index]
                pos_s = pos // 1000
                self.progress["value"] = min(pos_s, song.duration)
                self.time_lbl.config(
                    text=f"{pos_s//60}:{pos_s%60:02d} / {song.duration_str}")

        self.after(500, self._poll)


# ════════════════════════════════════════════════════════════
#  ENTRY POINT
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════╗
║      ML Mood Music Player — Starting ...            ║
╠══════════════════════════════════════════════════════╣
║  Step 1 → 🧠 Train Model                            ║
║           Select your samples/ folder               ║
║           (subfolders: happy/ sad/ energetic/ ...)   ║
║                                                      ║
║  Step 2 → 📂 Open Music                             ║
║           Select your full music library folder     ║
║                                                      ║
║  Step 3 → Click a mood pill to filter               ║
║           Next song always matches selected mood    ║
╚══════════════════════════════════════════════════════╝
""")
    app = MoodPlayerApp()
    app.mainloop()
