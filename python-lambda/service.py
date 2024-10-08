import socket

def get_ip_address():
    # Create a socket object
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Connect to a remote server to determine the local IP address
        s.connect(("8.8.8.8", 80))
        ip_address = s.getsockname()[0]
    except Exception as e:
        ip_address = "Unable to determine IP address"
        print(f"Error: {e}")
    finally:
        s.close()
    return ip_address


def handler(event, context):
    # Your code goes here!
    ip = get_ip_address()
    return {
        "statusCode": 200,
        "headers": { "Content-Type": "application/json"},
        "body": ip
    }