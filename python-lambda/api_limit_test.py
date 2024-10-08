import subprocess
import ccxt

def connect_vpn(config_file):
    subprocess.run(["/opt/homebrew/sbin/openvpn", "--config", config_file])

# Use different VPN configurations for each request
vpn_configs = ["config.ovpn"]

for config in vpn_configs:
    connect_vpn(config)
    exchange = ccxt.binance()
    # After connecting, fetch data using ccxt
    market_data = exchange.fetch_ticker('BTC/USDT')
    print(market_data)