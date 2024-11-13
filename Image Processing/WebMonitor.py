import requests
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
import numpy as np
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import tkinter as tk
from tkinter import ttk
import threading
from PIL import Image, ImageTk
import io
import time

class WebsiteMonitor:
    def __init__(self, urls, history_size=30):
        # Setup data structures
        self.urls = urls
        self.history_size = history_size
        self.times = {url: deque(maxlen=history_size) for url in urls}
        self.status_history = {url: deque(maxlen=history_size) for url in urls}
        self.response_times = {url: deque(maxlen=history_size) for url in urls}
        
        # Setup plot
        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(12, 6))
        self.fig.patch.set_facecolor('#1C1C1C')
        
        # Create grid layout
        gs = self.fig.add_gridspec(len(urls), 2, width_ratios=[0.15, 0.85])
        self.status_axes = []
        self.plot_axes = []
        
        # Create subplots for each website
        for i in range(len(urls)):
            status_ax = self.fig.add_subplot(gs[i, 0])
            plot_ax = self.fig.add_subplot(gs[i, 1])
            
            self.status_axes.append(status_ax)
            self.plot_axes.append(plot_ax)
            
            # Configure axes
            status_ax.set_xticks([])
            status_ax.set_yticks([])
            
            plot_ax.set_facecolor('#2C2C2C')
            plot_ax.grid(True, linestyle='--', alpha=0.3)
            
        self.fig.tight_layout(pad=3)
        
    def check_single_status(self, url):
        try:
            start_time = datetime.time.time()
            response = requests.get(url, timeout=5)
            response_time = (datetime.time.time() - start_time) * 1000  # Convert to ms
            return url, "UP" if response.status_code == 200 else "DOWN", response_time
        except:
            return url, "DOWN", 0
            
    def check_status(self):
        current_time = datetime.datetime.now()
        
        # Use ThreadPoolExecutor for parallel requests
        with ThreadPoolExecutor(max_workers=len(self.urls)) as executor:
            results = executor.map(self.check_single_status, self.urls)
            
        for url, status, response_time in results:
            self.times[url].append(current_time)
            self.status_history[url].append(1 if status == "UP" else 0)
            self.response_times[url].append(response_time)
    
    def update_plot(self, frame):
        self.check_status()
        
        for idx, url in enumerate(self.urls):
            status_ax = self.status_axes[idx]
            plot_ax = self.plot_axes[idx]
            
            # Clear previous plots
            status_ax.clear()
            plot_ax.clear()
            
            # Draw status indicator
            status = 1 if self.status_history[url][-1] == 1 else 0
            color = '#00FF00' if status else '#FF0000'
            circle = Circle((0.5, 0.5), 0.4, color=color, alpha=0.8)
            status_ax.add_patch(circle)
            
            # Add website name
            status_ax.text(0.5, -0.2, url.split('//')[1][:20],
                         ha='center', va='center',
                         fontsize=8, color='white')
            
            # Plot response time history
            times_list = list(self.times[url])
            status_list = list(self.status_history[url])
            response_times = list(self.response_times[url])
            
            if len(times_list) > 1:
                plot_ax.plot(times_list, response_times, 
                           color='#00FF00' if status else '#FF0000',
                           linewidth=2, alpha=0.8)
                
                # Fill area under curve
                plot_ax.fill_between(times_list, response_times,
                                   color='#00FF00' if status else '#FF0000',
                                   alpha=0.1)
                
                # Configure axis
                plot_ax.set_xlim(times_list[0], times_list[-1])
                plot_ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
                plot_ax.set_ylabel('Response (ms)', fontsize=8)
                
                if idx == len(self.urls) - 1:
                    plot_ax.set_xlabel('Time', fontsize=8)
            
            plot_ax.grid(True, linestyle='--', alpha=0.3)
            
        self.fig.suptitle('Website Status Monitor', 
                         color='white', fontsize=12, y=0.95)
        
    def start_monitoring(self):
        anim = FuncAnimation(self.fig, self.update_plot,
                           interval=10000,  # Update every 10 seconds
                           cache_frame_data=False)
        plt.show()

class WebsiteStatusGUI:
    def __init__(self, websites):
        self.websites = websites
        self.window = tk.Tk()
        self.window.title("Website Status Monitor")
        self.window.geometry("400x500")
        self.window.configure(bg='#2c2c2c')
        
        # Header
        header = tk.Label(self.window, 
                         text="Website Status Monitor",
                         fg='white',
                         bg='#2c2c2c',
                         font=('Arial', 16, 'bold'))
        header.pack(pady=20)
        
        # Main container
        self.container = ttk.Frame(self.window)
        self.container.pack(pady=20, expand=True, fill='both')
        
        # Configure style
        style = ttk.Style()
        style.configure('Status.TFrame', background='#2c2c2c')
        style.configure('Website.TLabel', 
                       background='#2c2c2c',
                       foreground='white',
                       font=('Arial', 10))
        
        # Website status frames
        self.status_frames = {}
        self.status_labels = {}
        for website in websites:
            frame = ttk.Frame(self.container, style='Status.TFrame')
            frame.pack(pady=10, fill='x', padx=20)
            
            # Website name
            name = website.split('//')[1][:30]
            label = ttk.Label(frame, 
                            text=name,
                            style='Website.TLabel')
            label.pack(side='left')
            
            # Status text
            status_label = tk.Label(frame,
                                  text="Checking...",
                                  bg='#2c2c2c',
                                  fg='#808080',
                                  font=('Arial', 10))
            status_label.pack(side='right')
            
            self.status_frames[website] = frame
            self.status_labels[website] = status_label
        
        # Refresh button
        self.refresh_btn = tk.Button(self.window,
                                   text="Refresh",
                                   command=self.check_status,
                                   bg='#404040',
                                   fg='white',
                                   activebackground='#505050',
                                   activeforeground='white',
                                   relief='flat',
                                   pady=5,
                                   padx=20)
        self.refresh_btn.pack(pady=20)
        
    def check_single_status(self, url):
        try:
            response = requests.get(url, timeout=5)
            return url, True if response.status_code == 200 else False
        except:
            return url, False
            
    def check_status(self):
        self.refresh_btn.config(state='disabled', text="Checking...")
        
        def update_status(url, is_up):
            color = '#00ff00' if is_up else '#ff0000'
            text = "ACTIVE" if is_up else "DOWN"
            self.status_labels[url].configure(fg=color, text=text)
            
        def enable_refresh():
            self.refresh_btn.config(state='normal', text="Refresh")
            # Schedule next auto-refresh
            self.window.after(30000, self.check_status)
        
        with ThreadPoolExecutor(max_workers=len(self.websites)) as executor:
            for url, status in executor.map(self.check_single_status, self.websites):
                self.window.after(0, update_status, url, status)
        
        self.window.after(0, enable_refresh)
    
    def start(self):
        self.check_status()
        self.window.mainloop()

# Usage (keeping your original websites)
websites = [
    "https://www.nischalskanda.xyz",
    "https://notfedex.000webhostapp.com/",
    "https://www.chatgpt.com"
]

monitor = WebsiteMonitor(websites)
monitor.start_monitoring()

app = WebsiteStatusGUI(websites)
app.start()