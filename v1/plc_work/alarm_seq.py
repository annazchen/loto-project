from pycomm3 import LogicDriver
import depthai as dai
###  BARCODE ###
TAG_NAME_BARCODE = "Sta630_MHR1_Serial_Tracking.Serial_Casting"
 
# Connect to device and start pipeline
#with dai.Device(pipeline, maxUsbSpeed=dai.UsbSpeed.HIGH) as device:
with dai.Device(pipeline, maxUsbSpeed=dai.UsbSpeed.SUPER) as device:
 
    controlQueue = device.getInputQueue("control")  # <--- ADD THIS LINE
    with LogicDriver('10.112.12.64') as plc: #10.112.12.12 10.112.12.22 192.168.3.5
        ###  BARCODE ###
        def read_serial_casting():
            result = plc.read(TAG_NAME_BARCODE)
            if result is not None and result.value:
                return result.value
            else:
                return "NO_BARCODE_READ"
 
         # Safe read_single
        def read_single(tag_name):
            global plc_fail
            try:
                return plc.read((tag_name)).value
            except Exception as e:
                print(f"⚠️ PLC read error for tag '{tag_name}':", e)
                plc_fail +=1
                return False  # Important! Return something safe
 
        # Safe write_single
        def write_single(tag_name, tag_value):
            try:
                return plc.write((tag_name, tag_value))
            except Exception as e:
                print(f"⚠️ PLC write error for tag '{tag_name}':", e)
                return False