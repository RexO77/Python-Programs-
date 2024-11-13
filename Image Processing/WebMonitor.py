import streamlit as st
import requests

def check_website_status(url):
    try:
        if not url.startswith(('http://', 'https://')):
            url = 'http://' + url
        response = requests.get(url, timeout=5)
        status_code = response.status_code
        if response.history:
            # Website was redirected
            redirected_urls = [resp.url for resp in response.history] + [response.url]
            redirect_chain = ' ➔ '.join(redirected_urls)
            if status_code == 200:
                return f"Up (Redirected: {redirect_chain})"
            else:
                return f"Down (Status Code: {status_code}, Redirected: {redirect_chain})"
        else:
            if status_code == 200:
                return 'Up'
            else:
                return f'Down (Status Code: {status_code})'
    except requests.exceptions.RequestException as e:
        return f"Down ({e})"

def main():
    st.set_page_config(page_title="Website Status Monitor", layout="wide")
    st.title("🌐 Website Status Monitor")

    # Sidebar
    st.sidebar.header("Settings")
    websites_input = st.sidebar.text_area("Enter website URLs (one per line):", value="""https://nischalskanda.xyz
http://nischalskanda.xyz
nischalskanda.xyz""")
    if st.sidebar.button('Check Websites'):
        st.session_state['check_multiple'] = True

    # Floating input bar in the middle with check button
    col1, col2, col3 = st.columns([1,2,1])

    with col2:
        st.markdown("###")  # Add vertical space
        st.markdown("###")
        st.markdown("###")
        # Centered container
        with st.container():
            col_input, col_button = st.columns([4,1])
            with col_input:
                website_input = st.text_input("", placeholder="Enter a website URL")
            with col_button:
                check_single = st.button("Check")
        st.markdown("---")

    status_placeholder = st.empty()

    if check_single and website_input:
        status = check_website_status(website_input.strip())
        with status_placeholder.container():
            if 'Up' in status:
                st.success(f"✅ **{website_input}** is **Up**")
                if 'Redirected' in status:
                    st.info(f"🔀 {status.split('Up ')[1]}")
            else:
                st.error(f"❌ **{website_input}** is **{status}**")

    if st.session_state.get('check_multiple'):
        st.subheader("Website Statuses")
        websites = [url.strip() if url.strip().startswith(('http://', 'https://')) else 'http://' + url.strip() for url in websites_input.split('\n') if url.strip()]
        statuses = {}
        with st.spinner('Checking website statuses...'):
            for url in websites:
                status = check_website_status(url)
                statuses[url] = status
        with status_placeholder.container():
            for url, status in statuses.items():
                if 'Up' in status:
                    st.success(f"✅ **{url}** is **Up**")
                    if 'Redirected' in status:
                        st.info(f"🔀 {status.split('Up ')[1]}")
                else:
                    st.error(f"❌ **{url}** is **{status}**")
        st.session_state['check_multiple'] = False

if __name__ == "__main__":
    main()