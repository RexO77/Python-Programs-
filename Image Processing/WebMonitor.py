import streamlit as st
import requests
import time
from concurrent.futures import ThreadPoolExecutor

def check_website_status(url):
    try:
        response = requests.get(url, timeout=5)
        status_code = response.status_code
        if status_code == 200:
            return 'Up'
        else:
            return f'Down (Status Code: {status_code})'
    except requests.exceptions.RequestException:
        return 'Down (No Response)'

def main():
    st.set_page_config(page_title="Website Status Monitor", layout="wide")
    st.title("🌐 Website Status Monitor")

    # Input websites to monitor
    st.sidebar.header("Settings")
    websites_input = st.sidebar.text_area("Enter website URLs (one per line):", value="""https://www.nischalskanda.xyz
https://notfedex.000webhostapp.com/
https://www.chat.com""")
    websites = [url.strip() for url in websites_input.split('\n') if url.strip()]
    refresh_interval = st.sidebar.slider("Refresh interval (seconds):", min_value=5, max_value=60, value=30)

    st.markdown("---")
    status_placeholder = st.empty()

    def get_statuses():
        statuses = {}
        with ThreadPoolExecutor(max_workers=len(websites)) as executor:
            futures = {executor.submit(check_website_status, url): url for url in websites}
            for future in futures:
                url = futures[future]
                status = future.result()
                statuses[url] = status
        return statuses

    while True:
        statuses = get_statuses()
        with status_placeholder.container():
            for url, status in statuses.items():
                if 'Up' in status:
                    st.success(f"✅ **{url}** is **Up**")
                else:
                    st.error(f"❌ **{url}** is **{status}**")
        time.sleep(refresh_interval)

if __name__ == "__main__":
    main()