from pycomm3 import LogixDriver
from ip import PLC_IP
from rf2 import curr_in
from people_detect import detections
import time

plc_fail = 0
TAG_GATE_OPEN = "Cel_060_Gate_Open"
TAG_TEACH_MODE = "Cel_060_Teach_Mode"
TAG_LOTO_ALARM = "Cel_060_LOTO_Alarm"



#safe read single
def read_single(plc, tag_name):
    global plc_fail
    try:
        return plc.read((tag_name)).value
    except Exception as e:
        print(f"PLC read error for tag '{tag_name}': ", e)
        plc_fail += 1
        return False #return safe value
        
#safe write single
def write_single(plc, tag_name, tag_value):
    global plc_fail
    try:
        return plc.write((tag_name, tag_value))
    except Exception as e:
        print(f"PLC write error for tag'{tag_name}:' ", e)
        return False

def teach_logic(num_dets, num_ids):
    if num_dets > 2:
        return True
    elif num_dets <= 2:
        return num_dets > num_ids
    else:
        return False
        

def loto_logic(gate_open, teach_mode, num_dets, num_ids):
    if gate_open:
        if teach_mode:
            return teach_logic(num_dets, num_ids)
        else: 
            return num_dets > num_ids
    else:
        return False

    

def main():
    with LogixDriver(PLC_IP) as plc:
        while True:
            gate_open = read_single(plc, TAG_GATE_OPEN)
            teach_mode = read_single(plc, TAG_TEACH_MODE)

            num_dets = len(detections)
            num_ids = len(curr_in)

            alarm = loto_logic(gate_open, teach_mode, num_dets, num_ids)
            write_single(plc, TAG_LOTO_ALARM, alarm)

            time.sleep(0.1)

if __name__ == "__main__":
    main()  
        
    
        
    
