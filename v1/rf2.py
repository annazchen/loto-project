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
#detection_counts = {}
#stores latest time an epc was read
last_seen = {}
#stores epc pairs that are directory keys, and time of when pair is detected
key_seen = {}
#stores users that have successfully removed key before timer
valid_usrs = {}
#loto validation tracker
loto_check = {}
#loto violation counter
loto_bad = []
#active tags inside
curr_in = []

directory = { #(lock epc, key epc) : name
    ('E2 80 6A 96 00 00 50 21 41 49 E1 92 3B A3', 'E2 80 6A 96 00 00 40 21 41 41 49 8B 9F 49') : "Anna Chen"
    }

def list_keys(dict):
    list_dict = []
    for key in dict:
        list_dict.append(key)
    return list_dict

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

def extract_frames(buffer: bytes):
    frames = []
    start = 0

    while True:
        start = buffer.find(b"\xBB", start)
        if start == -1:
            break
        end = buffer.find(b"\x7E", start)
        if end == -1:
            break
        frame = buffer[start : end + 1]
        frames.append(frame)
        start = end + 1
    return frames


#expected framing of raw response: start with 0xBB, end with 0x7E
def parse_epc(frame : bytes):
    if not (frame.startswith(b"\xBB") and frame.endswith(b"\x7E")):
        return None
    epc_bytes = frame[8 : -2]
    if len(epc_bytes) == 14 and epc_bytes.startswith(b"\xE2"):
        return epc_bytes.hex(" ").upper()
    return None

#removes epcs that reach timeout limit in last_seen
def epc_timeout(time): 
    global last_seen
    expired = [epc for epc, timestamp in last_seen.items() if time - timestamp >= TIMEOUT]
    for epc in expired:
        last_seen.pop(epc, None)
    
#6 detections = 1 read
def handle_detection(epc_bytes : bytes):
    global detection_counts
    epc_hex = bytes_to_hex(epc_bytes)
    now = time.time()
    #detection_counts[epc_hex] = detection_counts.get(epc_hex, 0) + 1
    #if detection_counts[epc_hex] >= 3:
    #    detection_counts[epc_hex] = 0
    last_seen[epc_hex] = now
    #if epc_hex not in last_seen:
     #   t0 = time.time()
      #  last_seen[epc_hex] = t0
        #debug
        
    #if epc_hex in last_seen:
    update_curr_in()
    print(f"EPC: {epc_hex}")

#
def update_curr_in():
    testing = {}
    t1 = time.time()
    global key_seen
    global loto_bad
    global curr_in
    epc_timeout(t1)
    for pair, name in directory.items():
        if all(epc in last_seen for epc in pair):
            if name not in curr_in:
                key_seen[pair] = t1
                curr_in.append(name)
                #debug
                print(f"{name} is inside.")
    for pair, t in key_seen.items():
        if (t1 - t) >= 6:
            testing[pair] = t
    for pair, t in testing.items():
        if pair[1] in last_seen:
            #debug
            print(f"loto violation by {directory[pair]}! key in lock!")
            loto_bad.append(directory[pair])
        else:
            valid_usrs[pair] = t
    for usrs, t in valid_usrs.items():
        if usrs[0] not in last_seen:
            curr_in.remove(directory[usrs])
            del key_seen[pair]

def handle_loto():
    if len(loto_bad) > 0:
        return True
    return False

#read log table
def read_table():
    print("\n--- log table ---")
    for entry in log_table:
        print(f"{entry['timestamp']} | {entry['epc']} | {entry['action']}")
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
                #debug
                #print("raw response", bytes_to_hex(response))

                #confirms that its the right type of id
                frames = extract_frames(response)
                for f in frames:
                    epc = parse_epc(f)
                if epc:
                    #split to important bytes
                    handle_detection(bytes.fromhex(epc.replace(" ", "")))
            
            epc_timeout(time.time())
            update_curr_in()
            print("active tags: ", list_keys(last_seen))
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


    
