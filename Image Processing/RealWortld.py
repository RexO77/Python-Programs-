import requests
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
import numpy as np
from collections import deque
from concurrent.futures import ThreadPoolExecutor

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

# Usage
websites = [
    "https://www.nischalskanda.xyz",
    "https://notfedex.000webhostapp.com/",
    "https://www.chat.com"
]

monitor = WebsiteMonitor(websites)
monitor.start_monitoring()