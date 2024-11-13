import streamlit as st
import requests

def check_website_status(url):
    try:
        if not url.startswith(('http://', 'https://')):
            url = 'http://' + url
        response = requests.get(url, timeout=5)
        status_code = response.status_code

        if response.history:
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
        return f'Down ({e})'

def main():
    st.set_page_config(page_title="Website Status Monitor", layout="centered")
    
    # Custom CSS
    st.markdown("""
        <style>
        .stApp {
            background-color: #f5f5f5;
        }
        div[data-testid="stToolbar"] {
            display: none;
        }
        </style>
    """, unsafe_allow_html=True)

    # Title and description
    st.markdown("<h1 style='text-align: center; padding-top: 2rem;'>🌐 Website Status Monitor</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Check if a website is up or down</p>", unsafe_allow_html=True)

    # Create three columns for centering
    col1, col2, col3 = st.columns([1,2,1])
    
    with col2:
        # Input field and button
        website_input = st.text_input("", placeholder="Enter website URL", label_visibility="collapsed")
        if st.button("Check Status", use_container_width=True):
            if website_input:
                status = check_website_status(website_input.strip())
                if 'Up' in status:
                    st.success(f"✅ **{website_input}** is **Up**")
                    if 'Redirected' in status:
                        st.info(f"🔀 {status.split('Up ')[1]}")
                else:
                    st.error(f"❌ **{website_input}** is **{status}**")
            else:
                st.warning("Please enter a website URL")

if __name__ == "__main__":
    main()