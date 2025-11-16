import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import os
import time
import tempfile
import logging
import pygame 
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import yt_dlp
from pydub import AudioSegment
from pydub.generators import Sine
import vlc 

# --- Setup Logging ---
def setup_logging():
    """Configures the logging for the application."""
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    
    # File handler
    file_handler = logging.FileHandler("app.log", mode='w')
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(logging.DEBUG)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    console_handler.setLevel(logging.INFO)
    
    # Get the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG) # Set root logger to lowest level
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

logger = logging.getLogger(__name__)

# --- Constants ---
TEMP_DIR = tempfile.gettempdir()
TEMP_DOWNLOAD_BASENAME = os.path.join(TEMP_DIR, "temp_original_download")
TEMP_DOWNLOAD_FINAL_MP3 = os.path.join(TEMP_DIR, "temp_original_download.mp3")
TEMP_CLICK_FILE = os.path.join(TEMP_DIR, "temp_click.wav")

class AudioToolApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YouTube Audio Tool")
        self.root.geometry("800x700")

        # --- Internal State ---
        self.current_file = None
        self.is_paused = False
        self.is_playing = False
        self.playback_line = None
        self.total_duration_sec = 0
        self.metronome_active = tk.BooleanVar(value=False)
        self.loop_active = tk.BooleanVar(value=False)
        self.speed_label_var = tk.StringVar(value="1.0x")

        # --- Loop State ---
        self.loop_start_sec = None
        self.loop_end_sec = None
        self.loop_span = None


        # --- Metronome State ---
        self.click_sound = None
        self.metronome_channel = None
        self.click_interval_ms = 0
        self.next_click_time_ms = 0
        self.beat_offset_ms = 0 
        self.tap_times = []
        self.last_tap_time = 0

        # --- Pygame Mixer (FOR CLICK ONLY) ---
        try:
            pygame.mixer.init()
            self.metronome_channel = pygame.mixer.Channel(0) 
            self.generate_click_sound()
        except Exception as e:
            logger.exception("Could not initialize audio mixer")
            messagebox.showerror("Pygame Error", f"Could not initialize audio mixer: {e}")
            root.destroy()
            return
            
        # --- NEW: libvlc Player Setup ---
        try:
            self.vlc_instance = vlc.Instance()
            self.player = self.vlc_instance.media_player_new()
        except Exception as e:
            logger.exception("Could not initialize libvlc")
            messagebox.showerror("VLC Error", f"Could not initialize libvlc. Is it installed? \nError: {e}")
            root.destroy()
            return

        self.create_widgets()

    def generate_click_sound(self):
        try:
            click = Sine(1000).to_audio_segment(duration=50).apply_gain(-10).fade_out(20)
            click.export(TEMP_CLICK_FILE, format="wav")
            self.click_sound = pygame.mixer.Sound(TEMP_CLICK_FILE)
        except Exception as e:
            logger.error(f"Error generating click sound: {e}")
            self.click_sound = None

    def create_widgets(self):
        # --- 1. Load Audio Frame ---
        load_frame = ttk.LabelFrame(self.root, text="1. Load Audio File")
        load_frame.pack(fill="x", padx=10, pady=5) 
        
        # --- Download Frame ---
        # (REMOVED Speed slider from this section)
        yt_frame = ttk.LabelFrame(load_frame, text="Download from YouTube")
        yt_frame.pack(fill="x", expand=True, padx=5, pady=5)
        yt_frame.columnconfigure(1, weight=1)
        
        ttk.Label(yt_frame, text="URL:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.url_entry = ttk.Entry(yt_frame, width=60)
        self.url_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        self.paste_button = ttk.Button(yt_frame, text="Paste", command=self.paste_from_clipboard)
        self.paste_button.grid(row=0, column=2, padx=5, pady=5)
        
        ttk.Label(yt_frame, text="Save As:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.save_path_entry = ttk.Entry(yt_frame, width=60)
        self.save_path_entry.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        self.browse_button = ttk.Button(yt_frame, text="Browse...", command=self.browse_save_location)
        self.browse_button.grid(row=1, column=2, padx=5, pady=5)
        
        self.process_button = ttk.Button(yt_frame, text="Download & Save", command=self.start_processing_thread)
        self.process_button.grid(row=0, column=3, rowspan=2, padx=10, pady=5, ipady=10)

        # --- Load from Disk Frame ---
        disk_frame = ttk.LabelFrame(load_frame, text="Load from Disk")
        disk_frame.pack(fill="x", expand=True, padx=5, pady=5)
        self.load_disk_button = ttk.Button(disk_frame, text="Load Audio File...", command=self.load_audio_from_disk)
        self.load_disk_button.pack(fill="x", padx=5, pady=5, ipady=5)

        # --- 2. Playback Settings Frame (MODIFIED) ---
        settings_frame = ttk.LabelFrame(self.root, text="2. Playback Settings")
        settings_frame.pack(fill="x", padx=10, pady=5)
        
        # Metronome Row
        metro_frame = ttk.Frame(settings_frame)
        metro_frame.pack(fill="x")
        ttk.Label(metro_frame, text="Metronome (BPM):").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.bpm_entry = ttk.Entry(metro_frame, width=10)
        self.bpm_entry.insert(0, "0")
        self.bpm_entry.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        self.metronome_check = ttk.Checkbutton(metro_frame, text="Active", variable=self.metronome_active)
        self.metronome_check.grid(row=0, column=2, padx=5, pady=5, sticky="w")
        self.tap_button = ttk.Button(metro_frame, text="Tap", command=self.tap_tempo)
        self.tap_button.grid(row=0, column=3, padx=5, pady=5, sticky="w")
        self.sync_button = ttk.Button(metro_frame, text="Auto-Sync 🎵", command=self.start_sync_thread)
        self.sync_button.grid(row=0, column=4, padx=5, pady=5, sticky="w")

        # --- NEW: Live Speed Row ---
        speed_frame = ttk.Frame(settings_frame)
        speed_frame.pack(fill="x")
        
        ttk.Label(speed_frame, text="Playback Speed:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.speed_scale = ttk.Scale(
            speed_frame, 
            from_=0.5, to=2.0, 
            orient="horizontal", 
            length=200,
            command=self.set_playback_speed # Link to new function
        )
        self.speed_scale.set(1.0)
        self.speed_scale.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        self.speed_label = ttk.Label(speed_frame, textvariable=self.speed_label_var, width=5)
        self.speed_label.grid(row=0, column=2, padx=5, pady=5, sticky="w")

        # --- NEW: Loop Row ---
        loop_frame = ttk.Frame(settings_frame)
        loop_frame.pack(fill="x")
        self.loop_check = ttk.Checkbutton(loop_frame, text="Loop Selection", variable=self.loop_active)
        self.loop_check.pack(side="left", padx=5, pady=5)
        self.clear_loop_button = ttk.Button(loop_frame, text="Clear Loop", command=self.clear_loop_selection)
        self.clear_loop_button.pack(side="left", padx=5, pady=5)


        # --- 3. Waveform & Controls Frame ---
        playback_frame = ttk.LabelFrame(self.root, text="3. Playback")
        playback_frame.pack(fill="both", expand=True, padx=10, pady=5)
        # (Waveform canvas - unchanged)
        self.fig, self.ax = plt.subplots(figsize=(8, 2))
        self.fig.tight_layout()
        self.canvas = FigureCanvasTkAgg(self.fig, master=playback_frame)
        canvas_widget = self.canvas.get_tk_widget()
        canvas_widget.pack(fill="both", expand=True, padx=5, pady=5)
        
        # --- FIX: Use matplotlib's event handling system ---
        self.canvas.mpl_connect('button_press_event', self.on_waveform_press)
        self.canvas.mpl_connect('motion_notify_event', self.on_waveform_drag)
        self.canvas.mpl_connect('button_release_event', self.on_waveform_release)

        self.ax.set_title("Audio Waveform")
        self.ax.set_yticks([])
        self.ax.set_xticks([])
        # (Controls - unchanged)
        controls_frame = ttk.Frame(playback_frame)
        controls_frame.pack(fill="x", padx=5, pady=5)
        self.rewind_button = ttk.Button(controls_frame, text="<<|", command=self.rewind_to_start)
        self.rewind_button.pack(side="left", fill="x", expand=True, padx=5, pady=5)
        self.skip_back_button = ttk.Button(controls_frame, text="-10s", command=self.skip_backward)
        self.skip_back_button.pack(side="left", fill="x", expand=True, padx=5, pady=5)
        self.play_button = ttk.Button(controls_frame, text="▶ Play", command=self.play_audio)
        self.play_button.pack(side="left", fill="x", expand=True, padx=5, pady=5)
        self.pause_button = ttk.Button(controls_frame, text="|| Pause", command=self.pause_audio)
        self.pause_button.pack(side="left", fill="x", expand=True, padx=5, pady=5)
        self.stop_button = ttk.Button(controls_frame, text="■ Stop", command=self.stop_audio)
        self.stop_button.pack(side="left", fill="x", expand=True, padx=5, pady=5)
        self.skip_fwd_button = ttk.Button(controls_frame, text="+10s", command=self.skip_forward)
        self.skip_fwd_button.pack(side="left", fill="x", expand=True, padx=5, pady=5)

        # --- 4. Status Bar & Progress ---
        status_frame = ttk.Frame(self.root)
        status_frame.pack(side="bottom", fill="x", padx=10, pady=5)
        self.status_label = ttk.Label(self.root, text="Ready. Load a file or download from YouTube.", anchor="w")
        self.status_label.pack(side="top", fill="x", padx=10)
        self.progressbar = ttk.Progressbar(self.root, orient="horizontal", mode="determinate", length=100)
        self.progressbar.pack(side="bottom", fill="x", padx=10, pady=5)
        
        self.set_playback_speed(self.speed_scale.get())

        self.root.bind("<space>", self.toggle_play_pause)

    # --- NEW: Live speed control ---
    def set_playback_speed(self, value_str):
        """Called by the speed slider to change speed LIVE."""
        try:
            value = float(value_str)
            rounded_value = round(value / 0.1) * 0.1
            self.speed_label_var.set(f"{rounded_value:.1f}x")
            
            # This is the magic line!
            self.player.set_rate(rounded_value) 
            
        except (ValueError, tk.TclError):
            pass

    def toggle_play_pause(self, event=None):
        """Toggles play/pause. Bound to spacebar."""
        # Ignore spacebar presses when an entry widget has focus
        if isinstance(self.root.focus_get(), (ttk.Entry, tk.Entry)):
            return
            
        if not self.current_file:
            return
            
        if self.is_playing:
            self.pause_audio()
        else:
            self.play_audio()

    def paste_from_clipboard(self):
        try:
            clipboard_content = self.root.clipboard_get()
            self.url_entry.delete(0, tk.END)
            self.url_entry.insert(0, clipboard_content)
            self.update_status("Pasted from clipboard.")
        except tk.TclError:
            self.update_status("Error: Clipboard is empty or contains invalid text.")

    def browse_save_location(self):
        filepath = filedialog.asksaveasfilename(
            title="Choose Save Location",
            defaultextension=".mp3",
            filetypes=[("MP3 Audio Files", "*.mp3"), ("All Files", "*.*")]
        )
        if filepath:
            self.save_path_entry.delete(0, tk.END)
            self.save_path_entry.insert(0, filepath)
            self.update_status("Save location set.")

    # --- (MODIFIED) ---
    def load_audio_from_disk(self):
        filepath = filedialog.askopenfilename(
            title="Select an MP3 File",
            filetypes=[("MP3 Audio Files", "*.mp3"), ("All Files", "*.*")]
        )
        if not filepath:
            self.update_status("File load cancelled.")
            return
        
        # Use our new helper function to load the file
        self.load_file_to_player(filepath)
        self.update_status(f"Loaded file: {os.path.basename(filepath)}")
        self.progressbar['value'] = 100
        
        # Reset settings
        self.speed_scale.set(1.0) 
        self.bpm_entry.delete(0, tk.END)
        self.bpm_entry.insert(0, "0")
        self.metronome_active.set(False)

    # --- NEW HELPER ---
    def load_file_to_player(self, filepath):
        """Helper function to load a file into VLC and our app state."""
        try:
            self.current_file = filepath
            
            # Load into VLC
            media = self.vlc_instance.media_new(filepath)
            self.player.set_media(media)
            
            self.stop_audio() 
            self.plot_waveform(self.current_file)
            self.beat_offset_ms = 0
            self.clear_loop_selection()
            
            # Reset speed to 1.0x when loading a new file
            self.speed_scale.set(1.0)
            self.player.set_rate(1.0)

        except Exception as e:
            logger.exception(f"Could not load the selected file: {filepath}")
            self.update_status(f"Error loading file: {e}")
            messagebox.showerror("Load Error", f"Could not load the selected file.\n\nError: {e}")
            self.current_file = None


    def start_processing_thread(self):
        url = self.url_entry.get()
        save_path = self.save_path_entry.get()
        if not url or not save_path:
            messagebox.showwarning("Input Missing", "Please provide a YouTube URL and a Save Location.")
            return
        self.process_button.config(state="disabled")
        self.load_disk_button.config(state="disabled")
        self.sync_button.config(state="disabled")
        threading.Thread(target=self.download_and_process, daemon=True).start()

    # --- (MODIFIED) ---
    def download_and_process(self):
        """Downloads the file but does NOT apply speed."""
        try:
            url = self.url_entry.get()
            final_save_path = self.save_path_entry.get()
            
            try:
                self.root.after(0, self.update_progressbar, 0)
            except tk.TclError:
                return # Stop if window is closed
            self.update_status(f"Downloading...")

            ydl_opts = {
                'format': 'bestaudio/best',
                'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3'}],
                'outtmpl': TEMP_DOWNLOAD_BASENAME,
                'noplaylist': True,
                'quiet': True,
                'overwrites': True,
                'progress_hooks': [self.progress_hook],
                'extractor_args': {"youtube": {"player_client": ["default"]}}
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])

            # --- NO SPEED CHANGE HERE ---
            # We just copy the temp file to the final path
            self.set_indeterminate_progress(f"Saving file to {final_save_path}...")
            # We use pydub just to load/export, ensuring it's a valid MP3
            sound = AudioSegment.from_mp3(TEMP_DOWNLOAD_FINAL_MP3)
            sound.export(final_save_path, format="mp3")
            
            # --- Load the new file ---
            try:
                self.root.after(0, self.load_file_to_player, final_save_path)
            except tk.TclError:
                return

            self.update_status(f"Ready to play: {os.path.basename(final_save_path)}")
            try:
                self.root.after(0, self.update_progressbar, 100)
            except tk.TclError:
                return

        except Exception as e:
            logger.exception("An error occurred during download and process")
            self.update_status(f"Error: {e}")
            try:
                self.root.after(0, self.update_progressbar, 0)
            except tk.TclError:
                return
        finally:
            try:
                self.root.after(0, self.progressbar.stop)
                self.root.after(0, lambda: self.progressbar.config(mode='determinate'))
                self.root.after(0, lambda: self.process_button.config(state="normal"))
                self.root.after(0, lambda: self.load_disk_button.config(state="normal"))
                self.root.after(0, lambda: self.sync_button.config(state="normal"))
            except tk.TclError:
                pass
    
    def update_status(self, message):
        logger.info(message)
        try:
            if self.root.winfo_exists():
                self.root.after(0, lambda: self.status_label.config(text=message))
        except tk.TclError:
            pass 

    def update_progressbar(self, value):
        try:
            if self.root.winfo_exists():
                self.root.after(0, lambda: self.progressbar.config(mode='determinate', value=value))
        except tk.TclError:
            pass

    def set_indeterminate_progress(self, message):
        try:
            self.update_status(message)
            if self.root.winfo_exists():
                self.root.after(0, lambda: self.progressbar.config(mode='indeterminate'))
                self.root.after(0, self.progressbar.start, 10)
        except tk.TclError:
            pass

    def progress_hook(self, d):
        if d['status'] == 'downloading':
            percent_str = d.get('_percent_str', '0.0%').strip().replace('%', '')
            try:
                percent_float = float(percent_str)
                if self.root.winfo_exists():
                    self.root.after(0, self.update_progressbar, percent_float)
                self.update_status(f"Downloading: {percent_float:.1f}%")
            except (ValueError, tk.TclError):
                pass
        elif d['status'] == 'finished':
            try:
                if self.root.winfo_exists():
                    self.root.after(0, self.update_progressbar, 100)
                self.update_status("Download finished. Processing...")
            except tk.TclError:
                pass
                
    def change_audio_speed(self, sound, speed=1.0):
        # This function is now only used if we re-add processing.
        # But for now, libvlc handles it.
        new_frame_rate = int(sound.frame_rate * speed)
        return sound._spawn(sound.raw_data, overrides={"frame_rate": new_frame_rate})

    def plot_waveform(self, file_path):
        def task():
            try:
                y, sr = librosa.load(file_path, sr=None)
                self.total_duration_sec = librosa.get_duration(y=y, sr=sr)
                self.ax.clear()
                librosa.display.waveshow(y, sr=sr, ax=self.ax, color="blue")
                if self.playback_line:
                    try: self.playback_line.remove()
                    except: pass
                self.playback_line = self.ax.axvline(x=0, color='r', linestyle='--')
                self.ax.set_title("Audio Waveform")
                self.ax.set_yticks([])
                self.ax.set_xlabel("Time")
                self.fig.tight_layout()
                if self.root.winfo_exists():
                    self.canvas.draw()
            except tk.TclError:
                pass
            except Exception as e:
                logger.error(f"Error plotting waveform: {e}")
                self.update_status(f"Error plotting waveform: {e}")
        try:
            if self.root.winfo_exists():
                self.root.after(0, task)
        except tk.TclError:
            pass

    # --- (MODIFIED) ---
    def update_playback_head(self):
        if not self.is_playing:
            return 
        try:
            # --- THE BIG CHANGE ---
            # Get time from VLC (in ms)
            current_ms = self.player.get_time()
            # ---
            
            current_sec = current_ms / 1000.0
            if current_sec < 0: return

            # --- NEW: Loop Logic ---
            if self.loop_active.get() and self.loop_start_sec is not None and self.loop_end_sec is not None:
                if current_sec >= self.loop_end_sec:
                    self.player.set_time(int(self.loop_start_sec * 1000))
                    # After setting the time, we get the new time to avoid a flicker
                    current_ms = self.player.get_time()
                    current_sec = current_ms / 1000.0
                    self.reset_metronome_timer(current_ms)

            # Check if finished
            if self.player.get_state() == vlc.State.Ended:
                self.stop_audio()
                return
                
            if self.playback_line:
                self.playback_line.set_xdata([current_sec, current_sec])
                if self.root.winfo_exists():
                    self.canvas.draw_idle()

            if self.metronome_active.get() and not self.is_paused and self.click_interval_ms > 0 and self.click_sound:
                if current_ms >= self.next_click_time_ms:
                    self.metronome_channel.play(self.click_sound)
                    self.next_click_time_ms += self.click_interval_ms
                    if self.next_click_time_ms < current_ms:
                        self.reset_metronome_timer(current_ms)
            
            if self.root.winfo_exists():
                self.root.after(50, self.update_playback_head)
        except tk.TclError:
            pass 
        except Exception as e:
            logger.error(f"Error in update_playback_head: {e}")
            self.is_playing = False

    def reset_metronome_timer(self, current_ms=None):
        if current_ms is None:
            current_ms = self.player.get_time()
            if current_ms < 0: current_ms = 0
            
        if self.click_interval_ms > 0:
            time_since_offset = current_ms - self.beat_offset_ms
            if time_since_offset < 0:
                self.next_click_time_ms = self.beat_offset_ms
            else:
                beats_passed = time_since_offset / self.click_interval_ms
                next_beat_num = int(beats_passed) + 1
                self.next_click_time_ms = self.beat_offset_ms + (next_beat_num * self.click_interval_ms)
        else:
            self.next_click_time_ms = 0
            
    # --- (MODIFIED) ---
    def play_audio(self):
        if self.current_file:
            try:
                bpm = float(self.bpm_entry.get()) 
                if bpm > 0:
                    self.click_interval_ms = 60000.0 / bpm
                else:
                    self.click_interval_ms = 0
            except ValueError:
                self.click_interval_ms = 0
            
            if not self.is_playing:
                # If stopped, reset timer
                if not self.is_paused:
                    self.reset_metronome_timer(0)
                    
                self.player.play()
                
            self.is_playing = True
            self.is_paused = False
            self.pause_button.config(text="|| Pause")
            self.update_status(f"Playing {os.path.basename(self.current_file)}")
            self.update_playback_head()
        else:
            self.update_status("No audio file loaded. Please select a file or download from YouTube.")

    # --- (MODIFIED) ---
    def pause_audio(self):
        if self.player.is_playing():
            self.player.pause() # This toggles
            if self.is_paused:
                self.is_paused = False
                self.is_playing = True
                self.pause_button.config(text="|| Pause")
                self.update_status("Resumed.")
                self.reset_metronome_timer()
                self.update_playback_head()
            else:
                self.is_paused = True
                self.is_playing = False
                self.pause_button.config(text="▶ Resume")
                self.update_status("Paused.")
                self.metronome_channel.stop()

    # --- (MODIFIED) ---
    def stop_audio(self):
        self.player.stop()
        self.metronome_channel.stop()
        self.is_paused = False
        self.is_playing = False
        self.pause_button.config(text="|| Pause")
        self.update_status("Stopped.")
        self.reset_metronome_timer(0)
        self.clear_loop_selection()
        
        try:
            if self.playback_line:
                self.playback_line.set_xdata([0, 0])
                self.canvas.draw_idle()
        except tk.TclError:
            pass

    def rewind_to_start(self):
        if not self.current_file:
            return
        self.stop_audio()
        self.play_audio()
        if not self.is_playing:
            self.stop_audio()
        self.update_status("Rewound to beginning.")

    def skip_backward(self):
        self.skip(-10.0)

    def skip_forward(self):
        self.skip(10.0)

    # --- (MODIFIED) ---
    def skip(self, seconds_to_skip):
        """VLC skip logic is much simpler and more robust."""
        if not self.current_file:
            return

        # Get current time
        current_ms = self.player.get_time()
        current_sec = current_ms / 1000.0
        
        new_sec = current_sec + seconds_to_skip
        
        # Get total duration
        total_ms = self.player.get_length()
        if total_ms > 0:
            total_sec = total_ms / 1000.0
            if new_sec >= total_sec:
                self.stop_audio()
                self.update_status("Reached end of audio.")
                return
        
        if new_sec < 0:
            new_sec = 0.0
        
        # --- The new logic ---
        self.player.set_time(int(new_sec * 1000))
        
        # Update metronome and line
        self.reset_metronome_timer(new_sec * 1000.0)
        try:
            if self.playback_line:
                self.playback_line.set_xdata([new_sec, new_sec])
                self.canvas.draw_idle()
        except tk.TclError:
            pass
            
        self.update_status(f"Skipped to {new_sec:.1f}s")


    def on_waveform_press(self, event):
        """Handles the start of a loop selection (left-click) or clears the loop (right-click)."""
        if event.inaxes != self.ax:
            return

        if event.button == 1:
            if not self.current_file:
                return
            self.clear_loop_selection()
            self.loop_start_sec = max(0, event.xdata)
            # FIX: Get the Polygon object returned by axvspan
            self.loop_span = self.ax.axvspan(self.loop_start_sec, self.loop_start_sec, color='red', alpha=0.3)
            self.canvas.draw_idle()

        elif event.button == 3:
            self.clear_loop_selection()
            self.loop_start_sec = None
            self.loop_end_sec = None
            self.canvas.draw_idle()

    def on_waveform_drag(self, event):
        """Updates the loop selection region as the user drags the mouse."""
        if not self.loop_span or event.inaxes != self.ax:
            return

        end_sec = max(0, event.xdata)
        if end_sec != self.loop_start_sec:
            self.loop_end_sec = end_sec
            verts = self.loop_span.get_xy()
            verts[0][0] = verts[3][0] = self.loop_start_sec
            verts[1][0] = verts[2][0] = self.loop_end_sec
            self.loop_span.set_xy(verts)
            self.canvas.draw_idle()

    def on_waveform_release(self, event):
        """Handles the end of a loop selection."""
        logger.debug(f"on_waveform_release: event={event.button}, xdata={event.xdata}, loop_span={'exists' if self.loop_span else 'None'}, inaxes={event.inaxes == self.ax}")
        if self.loop_span is None or event.inaxes != self.ax or event.button != 1:
            logger.debug("on_waveform_release: No loop_span, release outside axes, or not left-click, ignoring.")
            return
            
        self.loop_end_sec = max(0, event.xdata)
        logger.debug(f"on_waveform_release: Left-click released, loop_end_sec set to {self.loop_end_sec}")

        if self.loop_start_sec is not None and self.loop_end_sec is not None:
            if self.loop_start_sec > self.loop_end_sec:
                logger.debug(f"on_waveform_release: Swapping loop_start_sec ({self.loop_start_sec}) and loop_end_sec ({self.loop_end_sec})")
                self.loop_start_sec, self.loop_end_sec = self.loop_end_sec, self.loop_start_sec
            
            verts = self.loop_span.get_xy()
            verts[0][0] = self.loop_start_sec
            verts[1][0] = self.loop_end_sec
            verts[2][0] = self.loop_end_sec
            verts[3][0] = self.loop_start_sec
            verts[4][0] = self.loop_start_sec
            self.loop_span.set_xy(verts)
            self.canvas.draw_idle()
            self.update_status(f"Loop selected from {self.loop_start_sec:.2f}s to {self.loop_end_sec:.2f}s")
            self.loop_active.set(True) # Automatically activate loop on selection
            logger.debug(f"on_waveform_release: Loop finalized: start={self.loop_start_sec}, end={self.loop_end_sec}, loop_active={self.loop_active.get()}")

    def clear_loop_selection(self):
        """Removes the visual loop span and resets the state."""
        if self.loop_span:
            try:
                self.loop_span.remove()
            except Exception as e:
                logger.error(f"Error removing loop span: {e}")
        self.loop_span = None
        self.loop_start_sec = None
        self.loop_end_sec = None
        self.loop_active.set(False)
        try:
            self.canvas.draw_idle()
        except (tk.TclError, AttributeError):
            pass # Canvas might be gone


    def tap_tempo(self):
        current_time_sec = time.time()
        if (current_time_sec - self.last_tap_time) > 2.0:
            self.tap_times = []
        self.tap_times.append(current_time_sec)
        self.last_tap_time = current_time_sec
        if len(self.tap_times) > 4:
            self.tap_times.pop(0)
        if len(self.tap_times) < 2:
            return

        total_time_diff = self.tap_times[-1] - self.tap_times[0]
        num_intervals = len(self.tap_times) - 1
        avg_delta_sec = total_time_diff / num_intervals
        if avg_delta_sec == 0:
            return
            
        bpm = 60.0 / avg_delta_sec
        self.click_interval_ms = avg_delta_sec * 1000.0
        
        self.bpm_entry.delete(0, tk.END)
        self.bpm_entry.insert(0, f"{bpm:.1f}")
        self.update_status(f"Tap Tempo: {bpm:.1f} BPM")
        self.metronome_active.set(True)
        
        current_ms = self.player.get_time()
        if current_ms < 0:
            current_ms = 0
        self.beat_offset_ms = current_ms 
        
        if self.click_sound:
            self.metronome_channel.play(self.click_sound)
        self.next_click_time_ms = current_ms + self.click_interval_ms
        

    def start_sync_thread(self):
        if not self.current_file:
            messagebox.showwarning("No File", "Please load a file before trying to sync.")
            return
            
        self.sync_button.config(state="disabled")
        self.process_button.config(state="disabled")
        self.load_disk_button.config(state="disabled")
        
        self.set_indeterminate_progress("Analyzing beats... (this may take a moment)")
        
        threading.Thread(target=self.run_beat_analysis, daemon=True).start()

    def run_beat_analysis(self):
        try:
            y, sr = librosa.load(self.current_file, sr=None)
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            beat_times = librosa.frames_to_time(beats, sr=sr)
            
            if len(beat_times) == 0:
                if self.root.winfo_exists():
                    self.root.after(0, self.update_status, "Error: Auto-Sync could not detect any beats.")
                return 

            first_beat_sec = beat_times[0]
            if self.root.winfo_exists():
                self.root.after(0, self.apply_sync_results, tempo, first_beat_sec)
            
        except tk.TclError:
            pass
        except Exception as e:
            if self.root.winfo_exists():
                self.root.after(0, self.update_status, f"Sync Error: {e}")
            logger.exception("Beat analysis error")
        finally:
            try:
                if self.root.winfo_exists():
                    self.root.after(0, self.progressbar.stop)
                    self.root.after(0, lambda: self.progressbar.config(mode='determinate'))
                    self.root.after(0, lambda: self.sync_button.config(state="normal"))
                    self.root.after(0, lambda: self.process_button.config(state="normal"))
                    self.root.after(0, lambda: self.load_disk_button.config(state="normal"))
            except tk.TclError:
                pass

    def apply_sync_results(self, tempo, first_beat_sec):
        try:
            if isinstance(tempo, np.ndarray):
                tempo_float = float(tempo[0])
            else:
                tempo_float = float(tempo)

            if tempo_float <= 0:
                self.update_status(f"Error: Detected invalid BPM ({tempo_float:.1f})")
                return
                
            self.bpm_entry.delete(0, tk.END)
            self.bpm_entry.insert(0, f"{tempo_float:.1f}")
            self.click_interval_ms = 60000.0 / tempo_float
            self.beat_offset_ms = first_beat_sec * 1000.0
            self.metronome_active.set(True)
            self.reset_metronome_timer()
            self.update_status(f"Sync complete: {tempo_float:.1f} BPM, offset {first_beat_sec:.2f}s")
            
        except tk.TclError:
            pass
        except Exception as e:
            logger.error(f"Error applying sync: {e}")
            self.update_status(f"Error applying sync: {e}")

    def on_closing(self):
        # 1. Stop all loops
        self.is_playing = False 
        
        # 2. Stop audio engines
        self.stop_audio() # This stops self.player.stop() and self.metronome_channel.stop()
        
        # --- THIS IS THE FIX ---
        # We must explicitly release all VLC objects in the correct
        # order (player first, then the instance) *before* destroying
        # the Tkinter root.
        try:
            if self.player:
                self.player.release()
            if self.vlc_instance:
                self.vlc_instance.release()
            pygame.mixer.quit()
        except Exception as e:
            logger.error(f"Error during audio shutdown: {e}")
        # --- END OF FIX ---

        # 3. Destroy the root window (this must be LAST)
        self.root.destroy()

if __name__ == "__main__":
    setup_logging()
    root = tk.Tk()
    app = AudioToolApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

