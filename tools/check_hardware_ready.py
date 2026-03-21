"""Tool to verify network environment for real ESP32 CSI streaming."""
import socket
import sys

def check_udp_port(port=5005):
    """Check if UDP port is available or already in use."""
    print(f"Checking UDP port {port}...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind(("0.0.0.0", port))
        print(f"✅ Port {port} is FREE. You can run the server to listen on it.")
        sock.close()
        return True
    except OSError as e:
        if e.winerror == 10048:
             print(f"⚠️  Port {port} is BUSY. This usually means:")
             print("   1. The WiFi Body server is ALREADY running (Good!)")
             print("   2. Another application is blocking it.")
             return False
        else:
            print(f"❌ Error checking port: {e}")
            return False

def check_firewall():
    """Simple prompt for firewall check (since we can't reliably check via python cross-platform without admin)."""
    print("\n[Firewall Check]")
    print("Ensure Windows Firewall allows python.exe to receive UDP packets on port 5005.")
    print("Command to add rule (Run as Admin):")
    print('  netsh advfirewall firewall add rule name="ESP32 CSI" dir=in action=allow protocol=UDP localport=5005')

def main():
    print("=== WiFi Body: Hardware Readiness Check ===\n")
    check_udp_port(5005)
    check_firewall()
    print("\n=== Next Steps ===")
    print("1. Flash firmware to ESP32s from 'firmware/esp32-csi-node/'")
    print("2. Configure WiFi SSID/Password in 'sdkconfig.defaults'")
    print("3. Start server: python -m server")
    
if __name__ == "__main__":
    main()
