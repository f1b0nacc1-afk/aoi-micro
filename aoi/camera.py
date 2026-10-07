"""Capture backends: Basler (pypylon), Hikrobot (MVS MvImport), or image files.

Both cameras are IMX183 rolling shutter: capture only a static board, fixed exposure,
no auto-gain, Mono8 full frame. Averaging frames lowers noise for the variation model.
"""
from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

import cv2
import numpy as np


def read_gray(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img


def list_images(pattern: str) -> list:
    p = Path(pattern)
    if p.is_dir():
        files = [f for ext in ("png", "bmp", "tif", "tiff", "jpg") for f in p.glob(f"*.{ext}")]
    else:
        files = [Path(f) for f in glob.glob(pattern)]
    return sorted(files)


class Undistorter:
    def __init__(self, npz: Path | None):
        self.maps = None
        if npz:
            d = np.load(npz)
            self.K, self.D = d["K"], d["D"]

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if not hasattr(self, "K"):
            return img
        if self.maps is None:
            h, w = img.shape[:2]
            self.maps = cv2.initUndistortRectifyMap(self.K, self.D, None, self.K, (w, h), cv2.CV_16SC2)
        return cv2.remap(img, *self.maps, cv2.INTER_LINEAR)


class BaslerCamera:
    def __init__(self, cc: dict):
        from pypylon import pylon
        self.pylon = pylon
        self.cam = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        self.cam.Open()
        n = self.cam.GetNodeMap()
        self.cam.PixelFormat.SetValue("Mono8")
        for node, val in (("ExposureAuto", "Off"), ("GainAuto", "Off")):
            if n.GetNode(node) is not None:
                getattr(self.cam, node).SetValue(val)
        self.cam.ExposureTime.SetValue(float(cc["exposure_us"]))
        self.cam.Gain.SetValue(float(cc["gain_db"]))
        self.cam.OffsetX.SetValue(0)
        self.cam.OffsetY.SetValue(0)
        self.cam.Width.SetValue(self.cam.Width.GetMax())
        self.cam.Height.SetValue(self.cam.Height.GetMax())
        self.timeout = int(cc["timeout_ms"])

    def grab(self) -> np.ndarray:
        res = self.cam.GrabOne(self.timeout)
        try:
            if not res.GrabSucceeded():
                raise RuntimeError(f"pylon grab failed: {res.ErrorDescription}")
            return res.Array.copy()
        finally:
            res.Release()

    def close(self):
        self.cam.Close()


class HikrobotCamera:
    def __init__(self, cc: dict):
        path = cc.get("mvs_import_path") or self._default_import_path()
        if path and path not in sys.path:
            sys.path.append(path)
        import ctypes as ct
        from MvCameraControl_class import (MV_ACCESS_Exclusive, MV_CC_DEVICE_INFO,
                                           MV_CC_DEVICE_INFO_LIST, MV_FRAME_OUT,
                                           MV_USB_DEVICE, MvCamera)
        self.ct, self.MV_FRAME_OUT = ct, MV_FRAME_OUT
        devs = MV_CC_DEVICE_INFO_LIST()
        self._ok(MvCamera.MV_CC_EnumDevices(MV_USB_DEVICE, devs), "EnumDevices")
        if devs.nDeviceNum == 0:
            raise RuntimeError("no Hikrobot USB3 camera found")
        info = ct.cast(devs.pDeviceInfo[0], ct.POINTER(MV_CC_DEVICE_INFO)).contents
        self.cam = MvCamera()
        self._ok(self.cam.MV_CC_CreateHandle(info), "CreateHandle")
        self._ok(self.cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0), "OpenDevice")
        c = self.cam
        c.MV_CC_SetEnumValueByString("PixelFormat", "Mono8")
        c.MV_CC_SetEnumValueByString("ExposureAuto", "Off")
        c.MV_CC_SetEnumValueByString("GainAuto", "Off")
        self._ok(c.MV_CC_SetFloatValue("ExposureTime", float(cc["exposure_us"])), "ExposureTime")
        c.MV_CC_SetFloatValue("Gain", float(cc["gain_db"]))
        c.MV_CC_SetEnumValueByString("TriggerMode", "On")
        c.MV_CC_SetEnumValueByString("TriggerSource", "Software")
        self._ok(c.MV_CC_StartGrabbing(), "StartGrabbing")
        self.timeout = int(cc["timeout_ms"])

    @staticmethod
    def _default_import_path():
        for p in (r"C:\Program Files (x86)\MVS\Development\Samples\Python\MvImport",
                  "/opt/MVS/Samples/64/Python/MvImport",
                  "/Library/MVS_SDK/Samples/Python/MvImport"):
            if os.path.isdir(p):
                return p
        return None

    @staticmethod
    def _ok(ret, what):
        if ret != 0:
            raise RuntimeError(f"MVS {what} failed: 0x{ret & 0xFFFFFFFF:08x}")

    def grab(self) -> np.ndarray:
        ct = self.ct
        self._ok(self.cam.MV_CC_SetCommandValue("TriggerSoftware"), "TriggerSoftware")
        fr = self.MV_FRAME_OUT()
        ct.memset(ct.byref(fr), 0, ct.sizeof(fr))
        self._ok(self.cam.MV_CC_GetImageBuffer(fr, self.timeout), "GetImageBuffer")
        try:
            w, h = fr.stFrameInfo.nWidth, fr.stFrameInfo.nHeight
            buf = (ct.c_ubyte * (w * h)).from_address(ct.addressof(fr.pBufAddr.contents))
            return np.frombuffer(buf, np.uint8).reshape(h, w).copy()
        finally:
            self.cam.MV_CC_FreeImageBuffer(fr)

    def close(self):
        self.cam.MV_CC_StopGrabbing()
        self.cam.MV_CC_CloseDevice()
        self.cam.MV_CC_DestroyHandle()


def open_camera(cc: dict):
    b = cc["backend"]
    if b == "basler":
        return BaslerCamera(cc)
    if b == "hikrobot":
        return HikrobotCamera(cc)
    raise ValueError(f"camera backend '{b}' cannot grab; use 'basler' or 'hikrobot'")


def grab_averaged(cam, n: int) -> np.ndarray:
    if n <= 1:
        return cam.grab()
    acc = None
    for _ in range(n):
        f = cam.grab().astype(np.float32)
        acc = f if acc is None else acc + f
    return np.clip(np.round(acc / n), 0, 255).astype(np.uint8)
