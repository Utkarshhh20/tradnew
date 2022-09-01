mkdir -p ~/.streamlit/
echo "[theme]
primaryColor='#202020'
backgroundColor='#FFFFFF'
secondaryBackgroundColor='#f2f6fa'
textColor='#202020'
font='sans serif'
[server]
headless = true
port = $PORT
enableCORS = false
" > ~/.streamlit/config.toml
