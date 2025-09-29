import threading
import people_detect
import rf2
from plc_work import alarm_seq
import time

def start_people(stop_event):
    people_detect.main(stop_event)  

def start_rf2(stop_event):
    rf2.main(stop_event)            

def start_plc(stop_event):
    alarm_seq.main(stop_event)        

if __name__ == "__main__":
    stop_event = threading.Event()

    threads = []
    
    # Launch all 3 in parallel
    threads.append(threading.Thread(target=start_people, args = (stop_event,), daemon=True))
    threads.append(threading.Thread(target=start_rf2, args = (stop_event,), daemon=True))
    threads.append(threading.Thread(target=start_plc, args = (stop_event,), daemon=True))

    for t in threads:
        t.start()
    
    try: 
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nstopping all threads :>...")
        stop_event.set()

    # Keep the main thread alive
    for t in threads:
        t.join()
