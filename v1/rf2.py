import serial
import time
from serial.tools import list_ports
import argparse
import sys
import keyboard

#command to be sent to read epc values
SEND_CMD = bytes.fromhex("BB 00 22 00 00 22 7E")

#tap-in timeout
TIMEOUT = 3
#log table
log_table = []
#detection counter
detection_counts = {}
#read counter (read_counts : )
read_counts = {}
#stores latest time an epc was read
last_seen = {}
#active tags inside
curr_in = []

directory = {
    ('E2 80 6A 96 00 00 50 21 41 49 E1 92 3B A3', 'E2 80 6A 96 00 00 40 21 41 41 49 8B 9F 49') : "Anna Chen"
    }

def detect_ports():
    ports = list_ports.comports()
    return [(p.device, p.description) for p in ports]


def select_port():
    ports = detect_ports()
    if not ports:
        print("no ports detected.")
        return None
    else:
        print("available ports:")
        for i, (dev, desc) in enumerate(ports):
            print(f"    [{i}] {dev} - {desc}")
        index = input(f"pick port index (0 - {len(ports) - 1}) or type port name: ").strip()
        if index.isdigit():
            index = int(index)
            if 0 <=  index < len(ports):
                return ports[index][0]
            else:
                return None
        return index
    

def open_serial(port : str, baud : int, timeout : float = 0.5) -> serial.Serial:
    ser = serial.Serial(port = port, baudrate = baud, timeout = timeout)
    time.sleep(0.2)
    return ser


def bytes_to_hex(byte: bytes):
    return byte.hex(" ").upper()


#expected framing of raw response: start with 0xBB, end with 0x7E
def parse_epc(response: bytes):
    if not response or not(response.startswith(b"\xBB") and response.endswith(b"\x7E")):
        return None
    else:
        epc_bytes = response[8:-2]
        if epc_bytes:
            return bytes_to_hex(epc_bytes)

#6 detections = 1 read
def handle_detection(epc_bytes : bytes):
    epc_hex = bytes_to_hex(epc_bytes)
    detection_counts[epc_hex] = detection_counts.get(epc_hex, 0) + 1

    if detection_counts[epc_hex] >= 6:
        detection_counts = 0
        if epc_hex not in curr_in:
            curr_in.append(epc_hex)
            log_table.append({
                "epc" : epc_hex,
                "action" : action,
                "timestamp" : time.strftime("%Y-%m-%d %H:%M:%S")
            })
            action = "tap in"
            t0 = time.time()
            last_seen[epc_hex] = t0
            #debug
            print(f"[{action}] EPC: {epc_hex}")
        if epc_hex in curr_in:
            update_curr_in()


def update_curr_in():
    t0 = time.time()
    expired = [epc for epc, timestamp in last_seen.items() if t0 - timestamp ]
    for epc in expired:
        last_seen.pop(epc, None)
    global curr_in
    curr_in = list(last_seen.keys())


#SLATED FOR REMOVAL FROM handle_detections
    '''if detection_counts[epc_hex] >= 6:
        detection_counts[epc_hex] = 0
        read_counts[epc_hex] = read_counts.get(epc_hex, 0) + 1
        read_num = read_counts[epc_hex]

        if epc_hex not in curr_in:
            curr_in.append(epc_hex)
        else:
            curr_in.remove(epc_hex)

        action = "tap in" if read_num % 2 else "tap out"

        log_table.append({
            "read #" : read_num,
            "epc" : epc_hex,
            "action" : action,
            "timestamp" : time.strftime("%Y-%m-%d %H:%M:%S")
        })

        #debug
        print(f"[{action}] EPC: {epc_hex} (Read #{read_num})")'''


#
#read log table
def read_table():
    print("\n--- log table ---")
    for entry in log_table:
        print(f"{entry['timestamp']} | {entry['epc']} | {entry['action']} | {entry['read #']}")
    print("-----------------\n")

#identify based on key and lock id
def id_person(curr_in):
    for pair, name in directory.items():
        if pair.issubset(set(curr_in)):
            return name

#fetch current number of people inside 
def get_curr_in_len():
    global curr_in
    return len(curr_in)

#stop event:stops thread when ctrl + c is pressed
def read_loop(ser : serial.Serial, stop_event):
    print(f"Listening on {ser.port} @ {ser.baudrate} baud. Press Ctrl+C to stop. Press p to read log table.")
    try:
        time0 = time.time()          
        while not(stop_event and stop_event.is_set()):
            ser.write(SEND_CMD)
            time.sleep(0.2)
            response = ser.read_all()

            if response:
                #print("raw response", bytes_to_hex(response))

                #confirms that its the right type of id
                epc = parse_epc(response)
                if epc:
                    #split to important bytes
                    epc_bytes = response[8:-2]
                    if epc_bytes.startswith(b"\xE2"):
                        handle_detection(epc_bytes)
                        

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("stopping reader... (ctrl + c)")
    finally:
        try:
            ser.close()
        except Exception:
            pass



def main(stop_event = None):
    #adding hotkey 'p' to print log
    keyboard.add_hotkey('p', lambda: read_table())

    #add terminal arguments (port, baudrate, timeout, list of serial ports)
    parser = argparse.ArgumentParser(description = "RFID reader")
    parser.add_argument("--port", "-p", help = "serial port (e.g. COM3 or /dev/ttyUSB0). if omitted you will be prompted.")
    parser.add_argument("--baud", "-b", type = int, default=115200, help="baud rate (default: 115200)")
    parser.add_argument("--timeout", "-t", type = float, default=0.5, help="Serial read timeout seconds")
    parser.add_argument("--list", action = "store_true", help="List serial ports and exit")
    args = parser.parse_args()

    #detect port
    if args.list:
        ports = detect_ports()
        if not ports:
            print("no ports found.")
        else:
            print("serial ports: ")
            for dev, desc in ports:
                print(f" - {dev}: {desc}")
        sys.exit(0)
        
    #select port    
    port = args.port
    if not port:
        port = select_port()
        if not port:
            print("no port selected, exiting")
            return
    
    #open serial
    try:
        ser = open_serial(port, args.baud, timeout = args.timeout)
    except Exception as e:
        print(f"failed to open serial port {port} at {args.baud} baud: {e}")
        return

    read_loop(ser, stop_event)
    time.sleep(0.1)
    print('exiting rf2...')
    



if __name__ == "__main__":
    main()


    
