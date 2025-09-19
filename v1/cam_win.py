import time
import threading
import numpy as np
import cv2
 
try:
    import depthai as dai
except Exception as e:
    raise SystemExit(f"[error] depthai not available: {e}")
 
# ---- Preview / FPS ----
W, H, FPS = 640, 480, 30
 
class OakWorker(threading.Thread):
    """
    Shows OAK-1 Max color preview and applies controls via CameraControl from sliders.
 
    Sliders (in window 'OAK_CTRL'):
      OAK_AE [0/1]
      OAK_Exposure_us [100..33000]
      OAK_ISO [100..1600]
      OAK_Focus [0..255]   (note: many OAK-1 Max units have fixed-focus lenses; command may be ignored)
      OAK_Brightness [-10..10]
      OAK_Saturation [0..10]
    """
    def __init__(self):
        super().__init__(daemon=True)
        self.running = False
        self._last = {}
 
        # Defaults
        self.ae = 1
        self.exp_us = 8000
        self.iso = 400
        self.focus = 120
        self.brightness = 0
        self.saturation = 0
 
    def run(self):
        pipeline = dai.Pipeline()
 
        cam = pipeline.create(dai.node.ColorCamera)
        cam.setFps(FPS)
        cam.setInterleaved(False)
        cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        cam.setPreviewSize(W, H)
 
        xout = pipeline.create(dai.node.XLinkOut)
        xout.setStreamName("preview")
        cam.preview.link(xout.input)
 
        xin = pipeline.create(dai.node.XLinkIn)
        xin.setStreamName("control")
        xin.out.link(cam.inputControl)
 
        try:
            with dai.Device(pipeline) as device:
                q_preview = device.getOutputQueue("preview", maxSize=4, blocking=False)
                q_ctrl = device.getInputQueue("control")
 
                # Windows + sliders
                cv2.namedWindow("OAK", cv2.WINDOW_AUTOSIZE)
                cv2.namedWindow("OAK_CTRL", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("OAK_CTRL", 360, 260)
 
                cv2.createTrackbar("OAK_AE",          "OAK_CTRL", self.ae, 1, lambda v: None)
                cv2.createTrackbar("OAK_Exposure_us", "OAK_CTRL", self.exp_us, 33000, lambda v: None)
                cv2.createTrackbar("OAK_ISO",         "OAK_CTRL", self.iso, 1600, lambda v: None)
                cv2.createTrackbar("OAK_Focus",       "OAK_CTRL", self.focus, 255, lambda v: None)
                cv2.createTrackbar("OAK_Brightness",  "OAK_CTRL", self.brightness + 10, 20, lambda v: None)  # -10..10 -> 0..20
                cv2.createTrackbar("OAK_Saturation",  "OAK_CTRL", self.saturation, 10, lambda v: None)
 
                # Send initial control
                self._send_ctrl(q_ctrl, init=True)
 
                self.running = True
                last_t, frames = time.time(), 0
 
                while self.running:
                    # Read sliders
                    self.ae         = cv2.getTrackbarPos("OAK_AE", "OAK_CTRL")
                    self.exp_us     = max(100,  cv2.getTrackbarPos("OAK_Exposure_us", "OAK_CTRL"))
                    self.iso        = max(100,  cv2.getTrackbarPos("OAK_ISO", "OAK_CTRL"))
                    self.focus      =          cv2.getTrackbarPos("OAK_Focus", "OAK_CTRL")
                    self.brightness =          cv2.getTrackbarPos("OAK_Brightness", "OAK_CTRL") - 10
                    self.saturation =          cv2.getTrackbarPos("OAK_Saturation", "OAK_CTRL")
 
                    # Push control if changed
                    self._send_ctrl(q_ctrl)
 
                    # Show frame
                    pkt = q_preview.get()
                    frame = pkt.getCvFrame()  # BGR
                    cv2.imshow("OAK", frame)
                    frames += 1
 
                    now = time.time()
                    if now - last_t >= 1.0:
                        print(f"[OAK fps] ~{frames/(now-last_t):.1f}")
                        frames, last_t = 0, now
 
                    k = cv2.waitKey(1) & 0xFF
                    if k in (27, ord('q')):  # ESC/q to quit
                        self.running = False
 
        except Exception as e:
            print("[OAK] Device/pipeline error:", e)
        finally:
            try:
                cv2.destroyWindow("OAK")
                cv2.destroyWindow("OAK_CTRL")
            except Exception:
                pass
            print("[OAK] Stopped.")
 
    def _send_ctrl(self, q_ctrl, init=False):
        # Avoid spamming identical controls
        values = dict(ae=self.ae, exp=self.exp_us, iso=self.iso, foc=self.focus,
                      bri=self.brightness, sat=self.saturation)
        if not init and values == self._last:
            return
        self._last = values.copy()
 
        ctrl = dai.CameraControl()
        if self.ae:
            ctrl.setAutoExposureEnable()
        else:
            ctrl.setManualExposure(self.exp_us, self.iso)
 
        # Focus: many OAK-1 Max models have fixed-focus lenses; command may be ignored
        try:
            ctrl.setManualFocus(self.focus)
        except Exception:
            pass
 
        # Image params (supported ranges vary by device)
        try:
            ctrl.setBrightness(self.brightness)   # typically -10..10
        except Exception:
            pass
        try:
            ctrl.setSaturation(self.saturation)   # typically 0..10
        except Exception:
            pass
 
        q_ctrl.send(ctrl)
 
def main():
    print("Controls:")
    print("  - Press 'q' or ESC in the OAK window to quit.")
    print("  - Adjust sliders in 'OAK_CTRL' to change camera settings.")
 
    oak_thread = OakWorker()
    oak_thread.start()
    oak_thread.join()  # single device, just wait for thread to finish
    print("[main] Bye.")
 
if __name__ == "__main__":
    main()