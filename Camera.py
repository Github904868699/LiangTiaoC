# -*- coding: utf-8 -*-
"""
精简版 camera_slim_clean_no_autoswitch_yolo.py

改动要点：
- 无 config.json：不再读写配置、无状态
- 海康网络相机 + USB 摄像头选择、USB 索引刷新、重连/停止
- YOLO 开关 + 置信度（0~1，带上下箭头）过滤
- YOLO 多种框样式：
    0: 经典绿色框（统一颜色）
    1: 相机准星 + 中央红点（按类别变色，暗色系）
    2: YOLO 原版风格（按类别变色，亮色系 palette_yolo）
    3: 类别配色方案 A（偏高级感 palette_a）
    4: 类别配色方案 B（palette_a 柔和版）
    5: 极简白框 + 阴影
    6: 霓虹边框（palette_neon）
    7: 半透明填充框（palette_a）
- 标签文字颜色自动根据背景亮度选择黑/白，避免看不清
- 根据图像分辨率自动调整字体大小和线宽（高分辨率不再太小）
"""
import os
import sys
from pathlib import Path
import struct
import ctypes
import json
import socketserver
import threading
from ctypes import POINTER, byref, cast, c_ubyte
from typing import Optional, List, Dict

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QIcon
import cv2
import time
import socket

# YOLO (ultralytics)
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    YOLO_IMPORT_ERROR = None
except Exception as exc:
    YOLO = None  # type: ignore
    YOLO_AVAILABLE = False
    YOLO_IMPORT_ERROR = exc

TARGET_DISPLAY_WIDTH = 1280
UI_TARGET_FPS = 15.0
UI_PAINT_FPS = 12.0
CONFIG_PATH = "config.json"

TRIGGER_REGISTER_ADDR = 0
RESULT_REGISTER_ADDR_1 = 1
RESULT_REGISTER_ADDR_2 = 2

APP_TITLE = "Camera"
APP_ICON = "Camera.ico"
DEFAULT_YOLO_MODEL = "best2.pt"

# ---------------------- SDK import (HIK MVS) ----------------------
try:
    from MvCameraControl_class import MvCamera
    from CameraParams_header import (
        MV_CC_DEVICE_INFO,
        MV_CC_DEVICE_INFO_LIST,
        MV_FRAME_OUT_INFO_EX,
        MVCC_INTVALUE,
        MV_CC_PIXEL_CONVERT_PARAM,
    )
    from CameraParams_const import MV_GIGE_DEVICE, MV_ACCESS_Exclusive
    from PixelType_header import (
        PixelType_Gvsp_BGR8_Packed,
        PixelType_Gvsp_RGB8_Packed,
        PixelType_Gvsp_Mono8,
        PixelType_Gvsp_BayerRG8,
        PixelType_Gvsp_BayerBG8,
        PixelType_Gvsp_BayerGB8,
        PixelType_Gvsp_BayerGR8,
        PixelType_Gvsp_YUV422_Packed,
        PixelType_Gvsp_YUV422_YUYV_Packed,
    )
    from MvErrorDefine_const import MV_OK

    HIK_SDK_AVAILABLE = True
    HIK_SDK_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover
    MvCamera = None  # type: ignore
    MV_CC_DEVICE_INFO = None  # type: ignore
    MV_CC_DEVICE_INFO_LIST = None  # type: ignore
    MV_FRAME_OUT_INFO_EX = None  # type: ignore
    MVCC_INTVALUE = None  # type: ignore
    MV_CC_PIXEL_CONVERT_PARAM = None  # type: ignore
    MV_GIGE_DEVICE = 0  # type: ignore
    MV_ACCESS_Exclusive = 1  # type: ignore
    PixelType_Gvsp_BGR8_Packed = 0  # type: ignore
    PixelType_Gvsp_RGB8_Packed = 0  # type: ignore
    PixelType_Gvsp_Mono8 = 0  # type: ignore
    PixelType_Gvsp_BayerRG8 = 0  # type: ignore
    PixelType_Gvsp_BayerBG8 = 0  # type: ignore
    PixelType_Gvsp_BayerGB8 = 0  # type: ignore
    PixelType_Gvsp_BayerGR8 = 0  # type: ignore
    PixelType_Gvsp_YUV422_Packed = 0  # type: ignore
    PixelType_Gvsp_YUV422_YUYV_Packed = 0  # type: ignore
    MV_OK = 0  # type: ignore
    HIK_SDK_AVAILABLE = False
    HIK_SDK_IMPORT_ERROR = exc


# ---------------------- Helpers ----------------------
def resource_path(rel: str) -> str:
    """打包后获取资源路径（图标 / best.pt）"""
    base = getattr(sys, "_MEIPASS", Path(__file__).parent)
    return str(Path(base, rel))


def safe_load_json(path: str, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def load_config(path: str = CONFIG_PATH) -> Dict:
    cfg = None
    search_paths = []
    if path:
        search_paths.append(path)
        res_path = resource_path(path)
        if res_path != path:
            search_paths.append(res_path)

    for candidate in search_paths:
        cfg = safe_load_json(candidate, default=None)
        if cfg:
            break

    if not cfg:
        cfg = {
            "server": {"host": "0.0.0.0", "port": 502},
            "class_map": {},
            "models": [DEFAULT_YOLO_MODEL],
        }
    cfg.setdefault("server", {"host": "0.0.0.0", "port": 502})
    cfg.setdefault("class_map", {})
    cfg.setdefault("models", [DEFAULT_YOLO_MODEL])
    return cfg


def get_local_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        try:
            return socket.gethostbyname(socket.gethostname())
        except Exception:
            return "127.0.0.1"


def _usb_backend():
    if os.name == "nt":
        return getattr(cv2, "CAP_DSHOW", cv2.CAP_ANY)
    return cv2.CAP_ANY


def scan_usb_indices(max_index: int = 10) -> List[int]:
    backend = _usb_backend()
    found: List[int] = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, backend)
        ok = bool(cap and cap.isOpened())
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        if ok:
            found.append(i)
    return found or [0]


def _hik_decode_text(buf) -> str:
    try:
        raw = bytes(bytearray(buf))
    except Exception:
        return ""
    raw = raw.split(b"\0", 1)[0]
    try:
        return raw.decode("utf-8", errors="ignore").strip()
    except Exception:
        return ""


def _hik_uint_to_ip(value: int) -> str:
    try:
        return socket.inet_ntoa(struct.pack(">I", value))
    except Exception:
        return ""


def _hik_is_same_lan(info, local_ip_int: Optional[int]) -> bool:
    if not local_ip_int:
        return False
    try:
        gige = info.SpecialInfo.stGigEInfo
        cam_ip = int(gige.nCurrentIp)
        mask = int(gige.nCurrentSubNetMask) or 0xFFFFFFFF
        return (local_ip_int & mask) == (cam_ip & mask)
    except Exception:
        return False


def _hik_format_device_name(info, local_ip_int: Optional[int]) -> str:
    try:
        gige = info.SpecialInfo.stGigEInfo
        name = _hik_decode_text(gige.chUserDefinedName) or _hik_decode_text(gige.chModelName)
        ip = _hik_uint_to_ip(int(gige.nCurrentIp))
        if name and ip:
            base = f"{name} ({ip})"
        elif ip:
            base = f"Hik GIGE ({ip})"
        else:
            base = name or "Hik GIGE"
        if _hik_is_same_lan(info, local_ip_int):
            return base + " | 本地网段"
        return base
    except Exception:
        return "Hik GIGE"


def enumerate_hik_devices() -> List[tuple]:
    """Return available HIK devices sorted with same-LAN ones first."""

    if not HIK_SDK_AVAILABLE or MvCamera is None:
        return []

    dev_list = MV_CC_DEVICE_INFO_LIST()
    ret = MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE, dev_list)
    if ret != MV_OK or dev_list.nDeviceNum == 0:
        return []

    local_ip_int = None
    try:
        local_ip_int = struct.unpack(">I", socket.inet_aton(get_local_ip()))[0]
    except Exception:
        local_ip_int = None

    candidates = []
    for idx in range(int(dev_list.nDeviceNum)):
        ptr = dev_list.pDeviceInfo[idx]
        if not ptr:
            continue
        info_copy = MV_CC_DEVICE_INFO()
        ctypes.memmove(byref(info_copy), byref(ptr.contents), ctypes.sizeof(MV_CC_DEVICE_INFO))
        if not MvCamera.MV_CC_IsDeviceAccessible(info_copy, MV_ACCESS_Exclusive):
            continue
        display = _hik_format_device_name(info_copy, local_ip_int)
        same_lan = _hik_is_same_lan(info_copy, local_ip_int)
        candidates.append((info_copy, display, same_lan))

    candidates.sort(key=lambda item: (not item[2], item[1]))
    return candidates


# ---------------------- Modbus ----------------------
class ModbusRegisterModel:
    def __init__(self, size: int = 16):
        self._lock = threading.Lock()
        self._regs = [0] * max(size, 2)

    def read(self, addr: int, count: int) -> List[int]:
        with self._lock:
            if addr < 0:
                return [0] * max(count, 0)
            end = addr + count
            slice_regs = self._regs[addr:end]
            if len(slice_regs) < count:
                slice_regs.extend([0] * (count - len(slice_regs)))
            return list(slice_regs)

    def write(self, addr: int, values: List[int]):
        if addr < 0:
            return
        with self._lock:
            end = addr + len(values)
            if end > len(self._regs):
                self._regs.extend([0] * (end - len(self._regs)))
            for i, v in enumerate(values):
                self._regs[addr + i] = v & 0xFFFF

    def set_register(self, addr: int, value: int):
        self.write(addr, [value])


class ModbusRequestHandler(socketserver.BaseRequestHandler):
    def handle(self):
        while True:
            header = self._recvn(7)
            if not header:
                break
            try:
                tid, pid, length = struct.unpack(">HHH", header[:6])
            except struct.error:
                break
            unit = header[6]
            if length <= 0:
                continue
            payload = self._recvn(length - 1)
            if payload is None:
                break
            if not payload:
                continue
            function = payload[0]
            data = payload[1:]
            response_pdu = self._handle_function(function, data)
            if response_pdu is None:
                continue
            mbap = struct.pack(">HHHB", tid, 0, len(response_pdu) + 1, unit)
            try:
                self.request.sendall(mbap + response_pdu)
            except Exception:
                break

    def _recvn(self, size: int):
        buf = b""
        while len(buf) < size:
            chunk = self.request.recv(size - len(buf))
            if not chunk:
                return None if not buf else buf
            buf += chunk
        return buf

    def _handle_function(self, function: int, data: bytes) -> bytes | None:
        try:
            if function == 3:  # Read Holding Registers
                if len(data) < 4:
                    raise ValueError
                addr, count = struct.unpack(">HH", data[:4])
                regs = self.server.model.read(addr, count)
                payload = struct.pack(">B", len(regs) * 2)
                if regs:
                    payload += struct.pack(">" + "H" * len(regs), *regs)
                return bytes([function]) + payload
            elif function == 6:  # Write Single Register
                if len(data) < 4:
                    raise ValueError
                addr, value = struct.unpack(">HH", data[:4])
                self.server.model.write(addr, [value])
                if self.server.on_write:
                    self.server.on_write(addr, value & 0xFFFF)
                return bytes([function]) + data[:4]
            elif function == 16:  # Write Multiple Registers
                if len(data) < 5:
                    raise ValueError
                addr, count, byte_count = struct.unpack(">HHB", data[:5])
                expected = count * 2
                if byte_count != expected or len(data[5:]) < expected:
                    raise ValueError
                raw = data[5:5 + expected]
                values = list(struct.unpack(">" + "H" * count, raw))
                self.server.model.write(addr, values)
                if self.server.on_write:
                    for i, v in enumerate(values):
                        self.server.on_write(addr + i, v & 0xFFFF)
                return bytes([function]) + struct.pack(">HH", addr, count)
            else:
                return bytes([function | 0x80, 1])
        except Exception:
            return bytes([function | 0x80, 3])


class ModbusTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True

    def __init__(self, host: str, port: int, model: ModbusRegisterModel, on_write=None):
        self.model = model
        self.on_write = on_write
        super().__init__((host, port), ModbusRequestHandler)


def start_modbus_server(host: str, port: int, model: ModbusRegisterModel, on_write=None):
    server = ModbusTCPServer(host, port, model, on_write=on_write)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"[MODBUS] listen on {host}:{port}")
    return server


# ---------------------- Grabbers ----------------------
class HikGrabber(QtCore.QThread):
    frameSignal = QtCore.pyqtSignal(np.ndarray)
    infoSignal = QtCore.pyqtSignal(str)
    errorSignal = QtCore.pyqtSignal(str)
    autoAdjustSignal = QtCore.pyqtSignal()

    def __init__(self, parent=None, device_index: int = 0):
        super().__init__(parent)
        if not HIK_SDK_AVAILABLE or MvCamera is None:
            reason = str(HIK_SDK_IMPORT_ERROR) if HIK_SDK_IMPORT_ERROR else "未检测到海康 SDK"
            raise RuntimeError(f"海康 SDK 未就绪: {reason}")

        self.device_index = int(device_index)

        self.camera: Optional["MvCamera"] = None
        self._running = False

        self._data_buf = None
        self._data_ptr = None
        self._convert_buf = None
        self._convert_ptr = None
        self._convert_buf_size = 0
        self._payload_size = 0

        self._last_emit_ts = 0.0
        self._local_ip_int = self._ip_to_uint(get_local_ip())

        self._bayer_types = {
            PixelType_Gvsp_BayerRG8,
            PixelType_Gvsp_BayerBG8,
            PixelType_Gvsp_BayerGB8,
            PixelType_Gvsp_BayerGR8,
        }
        self._last_stream_error = 0
        self._last_convert_error = 0
        self._last_unsupported_pixel = 0

        self.autoAdjustSignal.connect(self._handle_auto_adjust)

    @staticmethod
    def _ip_to_uint(ip: str) -> Optional[int]:
        try:
            return struct.unpack(">I", socket.inet_aton(ip))[0]
        except Exception:
            return None

    @staticmethod
    def _uint_to_ip(value: int) -> str:
        try:
            return socket.inet_ntoa(struct.pack(">I", value))
        except Exception:
            return ""

    @staticmethod
    def _decode_text(buf) -> str:
        try:
            raw = bytes(bytearray(buf))
        except Exception:
            return ""
        raw = raw.split(b"\0", 1)[0]
        try:
            return raw.decode("utf-8", errors="ignore").strip()
        except Exception:
            return ""

    def _is_same_lan(self, info) -> bool:
        if not self._local_ip_int:
            return False
        try:
            gige = info.SpecialInfo.stGigEInfo
            cam_ip = int(gige.nCurrentIp)
            mask = int(gige.nCurrentSubNetMask) or 0xFFFFFFFF
            return (self._local_ip_int & mask) == (cam_ip & mask)
        except Exception:
            return False

    def _format_device_name(self, info) -> str:
        try:
            gige = info.SpecialInfo.stGigEInfo
            name = self._decode_text(gige.chUserDefinedName) or self._decode_text(gige.chModelName)
            ip = self._uint_to_ip(int(gige.nCurrentIp))
            if name and ip:
                return f"{name} ({ip})"
            if ip:
                return f"Hik GIGE ({ip})"
            return name or "Hik GIGE"
        except Exception:
            return "Hik GIGE"

    def _prepare_payload(self):
        payload = MVCC_INTVALUE()
        ret = self.camera.MV_CC_GetIntValue("PayloadSize", payload)
        if ret != MV_OK or int(payload.nCurValue) <= 0:
            raise RuntimeError(f"获取 PayloadSize 失败: 0x{ret:08X}")
        self._payload_size = int(payload.nCurValue)
        self._data_buf = (c_ubyte * self._payload_size)()
        self._data_ptr = cast(self._data_buf, POINTER(c_ubyte))

    def _get_int_value(self, key: str) -> Optional[int]:
        if not self.camera:
            return None
        value = MVCC_INTVALUE()
        ret = self.camera.MV_CC_GetIntValue(key, value)
        if ret == MV_OK:
            return int(value.nCurValue)
        return None

    def _ensure_convert_buffer(self, size: int):
        if self._convert_buf_size < size:
            self._convert_buf = (c_ubyte * size)()
            self._convert_ptr = cast(self._convert_buf, POINTER(c_ubyte))
            self._convert_buf_size = size

    def _convert_frame(self, frame_info: "MV_FRAME_OUT_INFO_EX") -> Optional[np.ndarray]:
        width = int(frame_info.nWidth)
        height = int(frame_info.nHeight)
        frame_len = int(frame_info.nFrameLen)
        pixel_type = int(frame_info.enPixelType)

        if frame_len <= 0 or width <= 0 or height <= 0:
            return None

        if pixel_type == PixelType_Gvsp_BGR8_Packed:
            arr = np.frombuffer(self._data_buf, dtype=np.uint8, count=frame_len)
            return arr.reshape(height, width, 3).copy()

        if pixel_type == PixelType_Gvsp_RGB8_Packed:
            arr = np.frombuffer(self._data_buf, dtype=np.uint8, count=frame_len)
            rgb = arr.reshape(height, width, 3)
            return rgb[:, :, ::-1].copy()

        if pixel_type == PixelType_Gvsp_Mono8:
            arr = np.frombuffer(self._data_buf, dtype=np.uint8, count=frame_len)
            gray = arr.reshape(height, width)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        if pixel_type in self._bayer_types or pixel_type in {
            PixelType_Gvsp_YUV422_Packed,
            PixelType_Gvsp_YUV422_YUYV_Packed,
        }:
            dst_size = width * height * 3
            self._ensure_convert_buffer(dst_size)

            convert_param = MV_CC_PIXEL_CONVERT_PARAM()
            convert_param.nWidth = width
            convert_param.nHeight = height
            convert_param.enSrcPixelType = pixel_type
            convert_param.pSrcData = self._data_ptr
            convert_param.nSrcDataLen = frame_len
            convert_param.enDstPixelType = PixelType_Gvsp_BGR8_Packed
            convert_param.pDstBuffer = self._convert_ptr
            convert_param.nDstBufferSize = dst_size
            convert_param.nDstLen = dst_size

            ret = self.camera.MV_CC_ConvertPixelType(convert_param)
            if ret != MV_OK:
                if self._last_convert_error != ret:
                    self.infoSignal.emit(f"[HIK] 像素转换失败: 0x{ret:08X}")
                    self._last_convert_error = ret
                return None

            self._last_convert_error = 0
            arr = np.frombuffer(self._convert_buf, dtype=np.uint8, count=dst_size)
            return arr.reshape(height, width, 3).copy()

        if self._last_unsupported_pixel != pixel_type:
            self.infoSignal.emit(f"[HIK] 不支持的像素格式: 0x{pixel_type:08X}")
            self._last_unsupported_pixel = pixel_type
        return None

    @QtCore.pyqtSlot()
    def _handle_auto_adjust(self):
        cam = getattr(self, "camera", None)
        if not cam:
            self.infoSignal.emit("[HIK] 相机未就绪，无法自动调节")
            return

        def _set_enum(name: str, value: int):
            try:
                ret = cam.MV_CC_SetEnumValue(name, value)
                if ret != MV_OK:
                    self.infoSignal.emit(f"[HIK] {name} 自动设置失败: 0x{ret:08X}")
                    return False
                return True
            except Exception as exc:
                self.infoSignal.emit(f"[HIK] {name} 自动设置异常: {exc}")
                return False

        ok_exp = _set_enum("ExposureAuto", 1)  # 1=Once
        ok_gain = _set_enum("GainAuto", 1)     # 1=Once
        ok_wb = _set_enum("BalanceWhiteAuto", 1)  # 1=Once

        if ok_exp or ok_gain or ok_wb:
            self.infoSignal.emit("[HIK] 已执行一键自动调节")
        else:
            self.infoSignal.emit("[HIK] 自动调节失败")

    def _cleanup_camera(self):
        if self.camera:
            try:
                self.camera.MV_CC_StopGrabbing()
            except Exception:
                pass
            try:
                self.camera.MV_CC_CloseDevice()
            except Exception:
                pass
            try:
                self.camera.MV_CC_DestroyHandle()
            except Exception:
                pass
            self.camera = None

        self._data_buf = None
        self._data_ptr = None
        self._convert_buf = None
        self._convert_ptr = None
        self._convert_buf_size = 0

    def stop(self):
        self._running = False

    def run(self):
        initialized = False
        try:
            ret = MvCamera.MV_CC_Initialize()
            if ret != MV_OK:
                raise RuntimeError(f"初始化海康 SDK 失败: 0x{ret:08X}")
            initialized = True

            devices = enumerate_hik_devices()
            if not devices:
                raise RuntimeError("未发现可用的海康相机")

            if self.device_index >= len(devices):
                raise RuntimeError(f"海康相机索引 {self.device_index + 1} 不存在")

            device_info, display_name, _ = devices[self.device_index]
            self.camera = MvCamera()

            ret = self.camera.MV_CC_CreateHandle(device_info)
            if ret != MV_OK:
                raise RuntimeError(f"创建相机句柄失败: 0x{ret:08X}")

            ret = self.camera.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
            if ret != MV_OK:
                raise RuntimeError(f"打开相机失败: 0x{ret:08X}")

            self._prepare_payload()

            ret = self.camera.MV_CC_StartGrabbing()
            if ret != MV_OK:
                raise RuntimeError(f"启动取流失败: 0x{ret:08X}")

            width = self._get_int_value("Width")
            height = self._get_int_value("Height")
            if width and height:
                self.infoSignal.emit(f"[INFO] HIK | {display_name} | {width}x{height}")
            else:
                self.infoSignal.emit(f"[INFO] HIK | {display_name}")

            self._running = True
            frame_info = MV_FRAME_OUT_INFO_EX()

            grabbed = 0
            last_fps_ts = time.time()
            self._last_emit_ts = 0.0

            while self._running:
                ret = self.camera.MV_CC_GetOneFrameTimeout(
                    self._data_ptr, self._payload_size, frame_info, 1000
                )
                if ret != MV_OK:
                    if ret != self._last_stream_error:
                        self.infoSignal.emit(f"[HIK] 取流异常: 0x{ret:08X}")
                        self._last_stream_error = ret
                    continue

                self._last_stream_error = 0
                now = time.time()
                grabbed += 1

                if (now - self._last_emit_ts) < (1.0 / UI_TARGET_FPS):
                    if (now - last_fps_ts) >= 1.0:
                        self.infoSignal.emit(f"[FPS] {grabbed / (now - last_fps_ts):.1f}")
                        grabbed = 0
                        last_fps_ts = now
                    continue

                frame = self._convert_frame(frame_info)
                if frame is None:
                    continue

                self._last_emit_ts = now
                self.frameSignal.emit(frame)

                if (now - last_fps_ts) >= 1.0:
                    self.infoSignal.emit(f"[FPS] {grabbed / (now - last_fps_ts):.1f}")
                    grabbed = 0
                    last_fps_ts = now

                QtCore.QThread.msleep(1)

        except Exception as exc:
            self._running = False
            self.infoSignal.emit(f"[HIK] {exc}")
            self.errorSignal.emit(str(exc))
        finally:
            self._cleanup_camera()
            if initialized:
                try:
                    MvCamera.MV_CC_Finalize()
                except Exception:
                    pass


class UsbGrabber(QtCore.QThread):
    frameSignal = QtCore.pyqtSignal(np.ndarray)
    infoSignal = QtCore.pyqtSignal(str)
    errorSignal = QtCore.pyqtSignal(str)

    def __init__(self, parent=None, index: int = 0):
        super().__init__(parent)
        self.index = int(index)
        self.cap: Optional[cv2.VideoCapture] = None
        self._running = False
        self._last_emit_ts = 0.0

    def run(self):
        try:
            backend = _usb_backend()
            self.cap = cv2.VideoCapture(self.index, backend)
            if (not self.cap or not self.cap.isOpened()) and backend != cv2.CAP_ANY:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = cv2.VideoCapture(self.index)

            if not self.cap or not self.cap.isOpened():
                raise RuntimeError(f"无法打开 USB 摄像头 (index={self.index})")

            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            if width > 0 and height > 0:
                self.infoSignal.emit(f"[INFO] USB | index={self.index} | {width}x{height}")
            else:
                self.infoSignal.emit(f"[INFO] USB | index={self.index}")

            self._running = True
            t0 = time.time()
            grabbed = 0

            while self._running:
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    QtCore.QThread.msleep(5)
                    continue

                grabbed += 1

                if frame.shape[1] > TARGET_DISPLAY_WIDTH:
                    scale = TARGET_DISPLAY_WIDTH / float(frame.shape[1])
                    frame = cv2.resize(
                        frame,
                        (TARGET_DISPLAY_WIDTH, int(frame.shape[0] * scale)),
                        interpolation=cv2.INTER_AREA,
                    )

                now_ts = time.time()
                if (now_ts - self._last_emit_ts) < (1.0 / UI_TARGET_FPS):
                    QtCore.QThread.msleep(5)
                    continue

                self._last_emit_ts = now_ts
                self.frameSignal.emit(frame.copy())

                now = time.time()
                if now - t0 >= 1.0:
                    self.infoSignal.emit(f"[FPS] {grabbed / (now - t0):.1f}")
                    t0 = now
                    grabbed = 0

                QtCore.QThread.msleep(1)

        except Exception as e:
            self.infoSignal.emit(f"[USB] {e}")
            self.errorSignal.emit(str(e))
        finally:
            self._stop_and_close()

    def stop(self):
        self._running = False

    def _stop_and_close(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        self.cap = None


# ---------------------- Main UI ----------------------
class MainWindow(QtWidgets.QMainWindow):
    modbus_trigger_sig = QtCore.pyqtSignal(int)

    def __init__(self):
        super().__init__()

        self.config = load_config(CONFIG_PATH)
        server_cfg = self.config.get("server", {})
        self.class_map: Dict[str, int] = {k: int(v) for k, v in self.config.get("class_map", {}).items()}
        self.models: List[str] = list(dict.fromkeys(self.config.get("models", []) or [DEFAULT_YOLO_MODEL]))
        self.default_model: str = self.models[0] if self.models else DEFAULT_YOLO_MODEL
        self.modbus_host = str(server_cfg.get("host", "0.0.0.0") or "0.0.0.0")
        self.modbus_port = int(server_cfg.get("port", 502))
        self.modbus_model = ModbusRegisterModel(size=8)
        self.modbus_server = None
        self.modbus_error: Optional[str] = None
        self.modbus_trigger_sig.connect(self._on_modbus_trigger)

        # 默认摄像头配置
        self.source = "hik"   # "hik" / "usb"
        self.usb_index = 0
        self.hik_devices: List[tuple] = []
        self.active_slots: List[int] = []

        # 默认 YOLO 配置
        self.yolo_enabled = False
        self.yolo_conf = 0.5
        self.yolo_model_path = self.default_model if self.default_model in self.models else DEFAULT_YOLO_MODEL
        self.yolo_model = None
        self.yolo_style = 0     # 0~7 不同样式

        self.last_frame_bgrs: Dict[int, np.ndarray] = {}
        self.last_frame_bgr: Optional[np.ndarray] = None
        self._last_paint_ts: Dict[int, float] = {}
        self.grabbers: Dict[int, QtCore.QThread] = {}

        self._start_modbus_server(self.modbus_host)

        self.setWindowTitle(APP_TITLE)
        self.resize(1160, 700)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        outer = QtWidgets.QVBoxLayout(central)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(8)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        outer.addWidget(splitter, 1)

        # 左侧
        left = QtWidgets.QWidget()
        left.setMinimumWidth(260)
        left.setMaximumWidth(360)
        splitter.addWidget(left)

        left_lay = QtWidgets.QVBoxLayout(left)
        left_lay.setContentsMargins(8, 8, 8, 8)
        left_lay.setSpacing(10)

        # 摄像头分组
        gb_src = QtWidgets.QGroupBox("摄像头")
        left_lay.addWidget(gb_src)
        form = QtWidgets.QFormLayout(gb_src)
        form.setContentsMargins(10, 10, 10, 10)
        form.setSpacing(8)

        self.source_combo = QtWidgets.QComboBox()
        self.source_combo.addItem("海康网络相机", "hik")
        self.source_combo.addItem("USB", "usb")
        idx = self.source_combo.findData(self.source)
        if idx >= 0:
            self.source_combo.setCurrentIndex(idx)
        self.source_combo.currentIndexChanged.connect(self._on_source_changed)
        form.addRow("摄像头：", self.source_combo)

        usb_row = QtWidgets.QHBoxLayout()
        self.usb_index_combo = QtWidgets.QComboBox()
        self.btn_usb_refresh = QtWidgets.QPushButton("刷新")
        self.btn_usb_refresh.setFixedWidth(70)
        self.btn_usb_refresh.clicked.connect(self._refresh_usb_indices)
        usb_row.addWidget(self.usb_index_combo, 1)
        usb_row.addWidget(self.btn_usb_refresh, 0)
        usb_wrap = QtWidgets.QWidget()
        usb_wrap.setLayout(usb_row)

        self.lbl_hik_devices = QtWidgets.QLabel("—")
        self.lbl_hik_devices.setWordWrap(True)

        self.idx_stack = QtWidgets.QStackedWidget()
        self.idx_stack.setContentsMargins(0, 0, 0, 0)
        self.idx_stack.addWidget(self.lbl_hik_devices)
        self.idx_stack.addWidget(usb_wrap)

        self.usb_index_wrap = usb_wrap
        form.addRow("索引：", self.idx_stack)

        # 控制分组
        gb_ctrl = QtWidgets.QGroupBox("控制")
        left_lay.addWidget(gb_ctrl)
        ctrl_lay = QtWidgets.QHBoxLayout(gb_ctrl)
        ctrl_lay.setContentsMargins(10, 10, 10, 10)
        ctrl_lay.setSpacing(8)

        self.btn_reopen = QtWidgets.QPushButton("重连")
        self.btn_stop = QtWidgets.QPushButton("停止")
        self.btn_auto_adjust = QtWidgets.QPushButton("一键自动调节")
        self.btn_reopen.clicked.connect(self.reopen_camera)
        self.btn_stop.clicked.connect(self.stop_camera)
        self.btn_auto_adjust.clicked.connect(self._on_auto_adjust_clicked)
        ctrl_lay.addWidget(self.btn_reopen, 1)
        ctrl_lay.addWidget(self.btn_stop, 1)
        ctrl_lay.addWidget(self.btn_auto_adjust, 1)

        # YOLO 分组
        gb_yolo = QtWidgets.QGroupBox("YOLO")
        left_lay.addWidget(gb_yolo)
        yolo_form = QtWidgets.QFormLayout(gb_yolo)
        yolo_form.setContentsMargins(10, 10, 10, 10)
        yolo_form.setSpacing(8)

        self.chk_yolo = QtWidgets.QCheckBox("启用 YOLO 识别")
        self.chk_yolo.setChecked(self.yolo_enabled)
        self.chk_yolo.stateChanged.connect(self._on_yolo_enabled_changed)
        yolo_form.addRow("开关：", self.chk_yolo)

        self.spin_conf = QtWidgets.QDoubleSpinBox()
        self.spin_conf.setDecimals(2)
        self.spin_conf.setRange(0.0, 1.0)
        self.spin_conf.setSingleStep(0.01)
        self.spin_conf.setValue(self.yolo_conf)
        self.spin_conf.valueChanged.connect(self._on_yolo_conf_changed)
        yolo_form.addRow("可信度：", self.spin_conf)

        self.combo_model = QtWidgets.QComboBox()
        for m in self.models:
            self.combo_model.addItem(os.path.basename(m), m)
        idx_model = self.combo_model.findData(self.yolo_model_path)
        if idx_model >= 0:
            self.combo_model.setCurrentIndex(idx_model)
        self.combo_model.currentIndexChanged.connect(self._on_yolo_model_changed)
        yolo_form.addRow("模型：", self.combo_model)

        self.combo_style = QtWidgets.QComboBox()
        self.combo_style.addItem("经典绿色框", 0)
        self.combo_style.addItem("相机准星 + 红点", 1)
        self.combo_style.addItem("YOLO 原版风格", 2)
        self.combo_style.addItem("类别配色方案 A", 3)
        self.combo_style.addItem("类别配色方案 B", 4)
        self.combo_style.addItem("极简白框 + 阴影", 5)
        self.combo_style.addItem("霓虹边框", 6)
        self.combo_style.addItem("半透明填充框", 7)
        self.combo_style.setCurrentIndex(self.yolo_style)
        self.combo_style.currentIndexChanged.connect(self._on_yolo_style_changed)
        yolo_form.addRow("框样式：", self.combo_style)

        # Modbus 分组
        gb_modbus = QtWidgets.QGroupBox("Modbus")
        left_lay.addWidget(gb_modbus)
        modbus_form = QtWidgets.QFormLayout(gb_modbus)
        modbus_form.setContentsMargins(10, 10, 10, 10)
        modbus_form.setSpacing(8)

        self.lbl_modbus_status = QtWidgets.QLabel("—")
        self.lbl_modbus_reg1 = QtWidgets.QLabel("—")
        self.lbl_modbus_reg2 = QtWidgets.QLabel("—")
        modbus_form.addRow("状态：", self.lbl_modbus_status)
        modbus_form.addRow("寄存器1：", self.lbl_modbus_reg1)
        modbus_form.addRow("寄存器2：", self.lbl_modbus_reg2)

        left_lay.addStretch(1)

        # 右侧
        right = QtWidgets.QWidget()
        splitter.addWidget(right)
        right_lay = QtWidgets.QVBoxLayout(right)
        right_lay.setContentsMargins(8, 8, 8, 8)
        right_lay.setSpacing(8)

        header_grid = QtWidgets.QGridLayout()
        self.lbl_cam1 = QtWidgets.QLabel("摄像头1: —")
        self.lbl_fps1 = QtWidgets.QLabel("FPS1: —")
        self.lbl_cam2 = QtWidgets.QLabel("摄像头2: —")
        self.lbl_fps2 = QtWidgets.QLabel("FPS2: —")
        header_grid.addWidget(self.lbl_cam1, 0, 0)
        header_grid.addWidget(self.lbl_fps1, 0, 1, QtCore.Qt.AlignRight)
        header_grid.addWidget(self.lbl_cam2, 1, 0)
        header_grid.addWidget(self.lbl_fps2, 1, 1, QtCore.Qt.AlignRight)
        right_lay.addLayout(header_grid)

        video_row = QtWidgets.QHBoxLayout()
        self.video_lbl1 = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self.video_lbl1.setMinimumSize(480, 360)
        self.video_lbl1.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.video_lbl1.setAutoFillBackground(True)

        self.video_lbl2 = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self.video_lbl2.setMinimumSize(480, 360)
        self.video_lbl2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.video_lbl2.setAutoFillBackground(True)

        pal = self.video_lbl1.palette()
        pal.setColor(QtGui.QPalette.Window, QtGui.QColor(30, 30, 30))
        self.video_lbl1.setPalette(pal)
        self.video_lbl2.setPalette(pal)

        video_row.addWidget(self.video_lbl1, 1)
        video_row.addWidget(self.video_lbl2, 1)
        right_lay.addLayout(video_row, 1)

        self.status_label = QtWidgets.QLabel("就绪")
        self.status_label.setMinimumHeight(22)
        self.status_label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        right_lay.addWidget(self.status_label, 0)

        self.toast_timer = QtCore.QTimer(self)
        self.toast_timer.setSingleShot(True)
        self.toast_timer.timeout.connect(lambda: self.status_label.setText(""))

        self.modbus_ui_timer = QtCore.QTimer(self)
        self.modbus_ui_timer.setInterval(800)
        self.modbus_ui_timer.timeout.connect(self._refresh_modbus_status)
        self.modbus_ui_timer.start()
        self._refresh_modbus_status()

        self._refresh_hik_devices()
        self._refresh_usb_indices()
        self._update_usb_controls_enabled()
        self._start_camera()

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([300, 860])

    # ---------- UI 逻辑 ----------
    def _current_source(self) -> str:
        return str(self.source_combo.currentData() or "hik")

    def _video_label_for(self, cam_index: int) -> Optional[QtWidgets.QLabel]:
        return self.video_lbl1 if cam_index == 1 else self.video_lbl2 if cam_index == 2 else None

    def _update_video_visibility(self):
        has_cam2 = 2 in self.active_slots
        if getattr(self.video_lbl2, "setVisible", None):
            self.video_lbl2.setVisible(has_cam2)
        if getattr(self.lbl_cam2, "setVisible", None):
            self.lbl_cam2.setVisible(has_cam2)
        if getattr(self.lbl_fps2, "setVisible", None):
            self.lbl_fps2.setVisible(has_cam2)
        if not has_cam2:
            lbl2 = self._video_label_for(2)
            if lbl2:
                lbl2.clear()
                lbl2.setText("无信号")

    def _refresh_hik_devices(self):
        try:
            devices = enumerate_hik_devices()
        except Exception as exc:
            devices = []
            self._toast(f"海康枚举失败: {exc}")
        self.hik_devices = devices
        self._update_index_controls()

    def _refresh_usb_indices(self):
        indices = scan_usb_indices(max_index=10)
        blocker = QtCore.QSignalBlocker(self.usb_index_combo)
        self.usb_index_combo.clear()
        for i in indices:
            self.usb_index_combo.addItem(str(i), int(i))
        idx = self.usb_index_combo.findData(int(self.usb_index))
        if idx < 0:
            idx = 0
            self.usb_index = int(self.usb_index_combo.itemData(0) or 0)
        self.usb_index_combo.setCurrentIndex(idx)
        del blocker

        try:
            self.usb_index_combo.currentIndexChanged.disconnect()
        except Exception:
            pass
        self.usb_index_combo.currentIndexChanged.connect(self._on_usb_index_changed)

    def _update_usb_controls_enabled(self):
        is_usb = (self._current_source() == "usb")
        self.usb_index_combo.setEnabled(is_usb)
        self.btn_usb_refresh.setEnabled(is_usb)
        self._update_index_controls()
        self._update_control_buttons_enabled()

    def _update_index_controls(self):
        is_usb = (self._current_source() == "usb")
        if getattr(self, "idx_stack", None) is not None:
            target = self.usb_index_wrap if is_usb else self.lbl_hik_devices
            self.idx_stack.setCurrentWidget(target)

        if getattr(self, "lbl_hik_devices", None) is not None:
            device_count = len(self.hik_devices)
            if device_count > 1:
                names = [f"摄像头{idx + 1}" for idx in range(device_count)]
                self.lbl_hik_devices.setText(", ".join(names))
                self.lbl_hik_devices.setVisible(True)
            elif device_count == 0:
                self.lbl_hik_devices.setText("未发现")
                self.lbl_hik_devices.setVisible(True)
            else:
                self.lbl_hik_devices.setText("—")
                self.lbl_hik_devices.setVisible(False)

    def _update_control_buttons_enabled(self):
        is_hik = (self._current_source() == "hik")
        if getattr(self, "btn_auto_adjust", None) is not None:
            self.btn_auto_adjust.setEnabled(is_hik)

    def _on_source_changed(self, _index: int):
        self.source = self._current_source()
        self._update_usb_controls_enabled()
        self._update_control_buttons_enabled()
        self._start_camera()

    def _on_usb_index_changed(self, index: int):
        if index < 0:
            return
        data = self.usb_index_combo.itemData(index)
        if data is None:
            return
        self.usb_index = int(data)
        if self._current_source() == "usb":
            self._start_camera()

    def _on_auto_adjust_clicked(self):
        target = None
        for g in self.grabbers.values():
            if isinstance(g, HikGrabber):
                target = g
                break
        if target is None:
            self._toast("当前非海康相机，无法自动调节")
            return
        target.autoAdjustSignal.emit()
        self._toast("正在自动调节...")

    # ---------- YOLO 控制 ----------
    def _on_yolo_enabled_changed(self, state: int):
        self.yolo_enabled = (state == QtCore.Qt.Checked)
        if self.yolo_enabled:
            self._ensure_yolo_model()

    def _on_yolo_conf_changed(self, value: float):
        self.yolo_conf = max(0.0, min(1.0, float(value)))

    def _on_yolo_model_changed(self, index: int):
        if index < 0:
            return
        data = self.combo_model.itemData(index)
        model_path = str(data or self.combo_model.currentText() or "").strip()
        if not model_path:
            return
        if model_path != self.yolo_model_path:
            self.yolo_model_path = model_path
            self.yolo_model = None
            if self.yolo_enabled:
                self._ensure_yolo_model()

    def _on_yolo_style_changed(self, index: int):
        data = self.combo_style.itemData(index)
        self.yolo_style = int(data) if data is not None else int(index)

    def _ensure_yolo_model(self) -> bool:
        if not YOLO_AVAILABLE:
            self._toast(f"[YOLO] 未安装 ultralytics 库: {YOLO_IMPORT_ERROR}")
            if hasattr(self, "chk_yolo"):
                self.chk_yolo.setChecked(False)
            self.yolo_enabled = False
            return False

        if self.yolo_model is not None:
            return True

        model_path = self.yolo_model_path or DEFAULT_YOLO_MODEL
        if not os.path.isabs(model_path):
            model_path = resource_path(model_path)

        if not os.path.exists(model_path):
            self._toast(f"[YOLO] 模型文件不存在: {model_path}")
            if hasattr(self, "chk_yolo"):
                self.chk_yolo.setChecked(False)
            self.yolo_enabled = False
            return False

        try:
            self.yolo_model = YOLO(model_path)
            self.yolo_model_path = model_path
            self._toast(f"[YOLO] 模型已加载: {os.path.basename(model_path)}", ms=2000)
            return True
        except Exception as exc:
            self._toast(f"[YOLO] 加载模型失败: {exc}")
            if hasattr(self, "chk_yolo"):
                self.chk_yolo.setChecked(False)
            self.yolo_enabled = False
            return False

    def _apply_yolo(self, frame_bgr: np.ndarray) -> np.ndarray:
        """在 BGR 图像上做 YOLO 推理并画框，只显示高于当前阈值的目标。
        同时根据分辨率自动调整字体大小和线宽。
        """
        if self.yolo_model is None:
            return frame_bgr

        try:
            results = self.yolo_model(frame_bgr, conf=float(self.yolo_conf), verbose=False)
            if not results:
                return frame_bgr

            res = results[0]
            boxes = getattr(res, "boxes", None)
            if boxes is None:
                return frame_bgr

            overlay = frame_bgr.copy()
            names = getattr(res, "names", None) or getattr(self.yolo_model, "names", None) or {}

            style = int(getattr(self, "yolo_style", 0))
            font = cv2.FONT_HERSHEY_SIMPLEX

            # ---------- 根据图像高度自适应字体 & 线宽 ----------
            h_img, w_img = overlay.shape[:2]
            base_ref_h = 480.0  # 480 高度基准
            scale_factor = h_img / base_ref_h
            scale_factor = max(0.7, min(3.0, scale_factor))

            base_font_scale = 0.5
            font_scale = base_font_scale * scale_factor
            base_thickness = max(1, int(round(1 * scale_factor)))

            line_thick_1 = max(1, int(round(1 * scale_factor)))
            line_thick_2 = max(1, int(round(2 * scale_factor)))
            line_thick_4 = max(2, int(round(4 * scale_factor)))

            # ----- 颜色方案 -----
            palette_yolo = [
                (0, 255, 255),
                (0, 0, 255),
                (255, 0, 0),
                (0, 255, 0),
                (255, 0, 255),
                (0, 128, 255),
                (255, 255, 0),
                (128, 0, 255),
                (255, 0, 128),
                (0, 255, 128),
            ]
            palette_a = [
                (80, 180, 200),
                (80, 160, 120),
                (180, 160, 100),
                (120, 150, 210),
                (150, 120, 180),
                (120, 120, 120),
                (70, 140, 170),
                (140, 140, 200),
                (160, 140, 120),
                (100, 170, 140),
            ]
            palette_b = [(b // 2 + 20, g // 2 + 20, r // 2 + 20) for (b, g, r) in palette_a]
            palette_neon = [
                (0, 255, 191),
                (255, 0, 191),
                (0, 191, 255),
                (191, 255, 0),
                (255, 128, 0),
                (191, 0, 255),
                (0, 255, 128),
                (255, 0, 128),
                (128, 255, 0),
                (0, 128, 255),
            ]

            def color_for_class(cls_id: int, palette: List[tuple]) -> tuple:
                if not palette:
                    return (0, 255, 0)
                if cls_id < 0:
                    return palette[0]
                return palette[int(cls_id) % len(palette)]

            def pick_text_color(bg_bgr: tuple) -> tuple:
                """根据背景亮度自动选黑/白字，防止看不清。"""
                b, g, r = bg_bgr
                lum = 0.299 * r + 0.587 * g + 0.114 * b
                return (0, 0, 0) if lum > 160 else (255, 255, 255)

            for box in boxes:
                score = float(box.conf[0]) if box.conf is not None else 0.0
                if score < float(self.yolo_conf):
                    continue

                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                cls_id = int(box.cls[0]) if box.cls is not None else -1

                label = str(cls_id)
                if isinstance(names, dict) and cls_id in names:
                    label = str(names[cls_id])
                elif isinstance(names, (list, tuple)) and 0 <= cls_id < len(names):
                    label = str(names[cls_id])

                text = f"{label} {score:.2f}"

                x1 = max(0, min(w_img - 1, x1))
                x2 = max(0, min(w_img - 1, x2))
                y1 = max(0, min(h_img - 1, y1))
                y2 = max(0, min(h_img - 1, y2))
                if x2 <= x1 or y2 <= y1:
                    continue

                box_color = (0, 255, 0)
                label_bg_color = (0, 0, 0)
                txt_color = (255, 255, 255)
                line_thick = line_thick_2

                # ---- 不同样式画框 ----
                if style == 0:
                    # 经典绿色框
                    box_color = (0, 255, 0)
                    label_bg_color = (0, 0, 0)
                    txt_color = (0, 255, 0)
                    line_thick = line_thick_2
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), box_color, line_thick, cv2.LINE_AA)

                elif style == 1:
                    # 相机准星 + 红点，用 palette_b（暗一点）
                    box_color = color_for_class(cls_id, palette_b)
                    label_bg_color = (0, 0, 0)
                    txt_color = (255, 255, 255)
                    line_thick = line_thick_2

                    w_box = x2 - x1
                    h_box = y2 - y1
                    line_len = max(6, min(w_box, h_box) // 4)

                    # 四角线
                    cv2.line(overlay, (x1, y1), (x1 + line_len, y1), box_color, line_thick, cv2.LINE_AA)
                    cv2.line(overlay, (x1, y1), (x1, y1 + line_len), box_color, line_thick, cv2.LINE_AA)

                    cv2.line(overlay, (x2, y1), (x2 - line_len, y1), box_color, line_thick, cv2.LINE_AA)
                    cv2.line(overlay, (x2, y1), (x2, y1 + line_len), box_color, line_thick, cv2.LINE_AA)

                    cv2.line(overlay, (x1, y2), (x1 + line_len, y2), box_color, line_thick, cv2.LINE_AA)
                    cv2.line(overlay, (x1, y2), (x1, y2 - line_len), box_color, line_thick, cv2.LINE_AA)

                    cv2.line(overlay, (x2, y2), (x2 - line_len, y2), box_color, line_thick, cv2.LINE_AA)
                    cv2.line(overlay, (x2, y2), (x2, y2 - line_len), box_color, line_thick, cv2.LINE_AA)

                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    cv2.circle(overlay, (cx, cy), max(2, int(3 * scale_factor)), (0, 0, 200), -1, cv2.LINE_AA)

                elif style == 2:
                    # YOLO 原版：palette_yolo
                    box_color = color_for_class(cls_id, palette_yolo)
                    label_bg_color = box_color
                    txt_color = pick_text_color(label_bg_color)
                    line_thick = line_thick_2
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), box_color, line_thick, cv2.LINE_AA)

                elif style == 3:
                    # 类别配色 A：palette_a
                    box_color = color_for_class(cls_id, palette_a)
                    label_bg_color = box_color
                    txt_color = pick_text_color(label_bg_color)
                    line_thick = line_thick_2
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), box_color, line_thick, cv2.LINE_AA)

                elif style == 4:
                    # 类别配色 B：palette_b（更柔和）
                    box_color = color_for_class(cls_id, palette_b)
                    label_bg_color = box_color
                    txt_color = pick_text_color(label_bg_color)
                    line_thick = line_thick_2
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), box_color, line_thick, cv2.LINE_AA)

                elif style == 5:
                    # 极简白框 + 阴影
                    box_color = (255, 255, 255)
                    label_bg_color = (0, 0, 0)
                    txt_color = (255, 255, 255)
                    line_thick = line_thick_1
                    cv2.rectangle(
                        overlay,
                        (x1 + line_thick_1, y1 + line_thick_1),
                        (x2 + line_thick_1, y2 + line_thick_1),
                        (0, 0, 0),
                        thickness=line_thick_2,
                        lineType=cv2.LINE_AA,
                    )
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), box_color, line_thick, cv2.LINE_AA)

                elif style == 6:
                    # 霓虹边框：palette_neon
                    box_color = color_for_class(cls_id, palette_neon)
                    label_bg_color = box_color
                    txt_color = pick_text_color(label_bg_color)
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), box_color, line_thick_4, cv2.LINE_AA)
                    cv2.rectangle(
                        overlay,
                        (x1 + line_thick_1, y1 + line_thick_1),
                        (x2 - line_thick_1, y2 - line_thick_1),
                        box_color,
                        line_thick_1,
                        cv2.LINE_AA,
                    )
                    line_thick = line_thick_2

                else:
                    # style == 7: 半透明填充框（用 palette_a）
                    box_color = color_for_class(cls_id, palette_a)
                    label_bg_color = box_color
                    txt_color = pick_text_color(label_bg_color)
                    line_thick = line_thick_2
                    roi = overlay[y1:y2, x1:x2]
                    color_layer = np.full_like(roi, box_color, dtype=np.uint8)
                    cv2.addWeighted(color_layer, 0.25, roi, 0.75, 0, roi)
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 255), line_thick_1, cv2.LINE_AA)

                # ---- 标签文字 ----
                (tw, th), baseline = cv2.getTextSize(text, font, font_scale, base_thickness)
                text_x = x1
                text_y = y1 - 4
                if text_y - th - baseline < 0:
                    text_y = y1 + th + baseline + 4

                bg_x1 = text_x
                bg_y1 = text_y - th - baseline
                bg_x2 = text_x + tw
                bg_y2 = text_y + baseline

                bg_x1 = max(0, min(w_img - 1, bg_x1))
                bg_y1 = max(0, min(h_img - 1, bg_y1))
                bg_x2 = max(0, min(w_img - 1, bg_x2))
                bg_y2 = max(0, min(h_img - 1, bg_y2))

                if bg_x2 > bg_x1 and bg_y2 > bg_y1:
                    cv2.rectangle(
                        overlay,
                        (bg_x1, bg_y1),
                        (bg_x2, bg_y2),
                        label_bg_color,
                        thickness=-1,
                    )

                cv2.putText(
                    overlay,
                    text,
                    (bg_x1, bg_y2 - baseline),
                    font,
                    font_scale,
                    txt_color,
                    base_thickness,
                    cv2.LINE_AA,
                )

            return overlay
        except Exception as exc:
            self._toast(f"[YOLO] 推理异常: {exc}")
            return frame_bgr

    def _run_yolo_recognition(self, cam_index: int) -> int:
        frame = self.last_frame_bgrs.get(cam_index)
        if frame is None:
            self._toast(f"摄像头{cam_index} 未捕获画面")
            return 0xFF

        if not self._ensure_yolo_model():
            return 0xFF

        try:
            results = self.yolo_model(frame, conf=float(self.yolo_conf), verbose=False)
        except Exception as exc:
            self._toast(f"[YOLO] 推理异常: {exc}")
            return 0xFF

        if not results:
            return 0xFF

        res = results[0]
        boxes = getattr(res, "boxes", None)
        if boxes is None:
            return 0xFF

        names = getattr(res, "names", None) or getattr(self.yolo_model, "names", None) or {}
        best_score = -1.0
        best_label: Optional[str] = None
        best_cls_id: int = -1

        for box in boxes:
            score = float(box.conf[0]) if box.conf is not None else 0.0
            if score < float(self.yolo_conf):
                continue

            cls_id = int(box.cls[0]) if box.cls is not None else -1
            label = str(cls_id)
            if isinstance(names, dict) and cls_id in names:
                label = str(names[cls_id])
            elif isinstance(names, (list, tuple)) and 0 <= cls_id < len(names):
                label = str(names[cls_id])

            if score > best_score:
                best_score = score
                best_label = label
                best_cls_id = cls_id

        if best_label is None:
            return 0xFF

        mapped = self.class_map.get(best_label)
        if mapped is None and best_cls_id >= 0:
            mapped = self.class_map.get(str(best_cls_id))

        if mapped is None:
            print(f"[MODBUS] 未找到类别映射: {best_label}")
            return 0

        return int(mapped) & 0xFFFF

    # ---------- Modbus ----------
    def _publish_modbus_result(self, addr: int, value: int):
        model = getattr(self, "modbus_model", None)
        if not model:
            return
        model.set_register(addr, int(value) & 0xFFFF)
        self._refresh_modbus_status()

    def _refresh_modbus_status(self):
        status_text = "未启动"
        if getattr(self, "modbus_error", None):
            status_text = f"错误: {self.modbus_error}"
        elif getattr(self, "modbus_server", None):
            status_text = f"{self.modbus_host}:{self.modbus_port}"

        if getattr(self, "lbl_modbus_status", None) is not None:
            self.lbl_modbus_status.setText(status_text)

        reg_val_1 = None
        reg_val_2 = None
        model = getattr(self, "modbus_model", None)
        if model:
            try:
                vals1 = model.read(RESULT_REGISTER_ADDR_1, 1)
                reg_val_1 = vals1[0] if vals1 else None
                vals2 = model.read(RESULT_REGISTER_ADDR_2, 1)
                reg_val_2 = vals2[0] if vals2 else None
            except Exception:
                reg_val_1 = None
                reg_val_2 = None

        reg_text_1 = "—" if reg_val_1 is None else f"{reg_val_1} 0x{int(reg_val_1) & 0xFFFF:04X}"
        reg_text_2 = "—" if reg_val_2 is None else f"{reg_val_2} 0x{int(reg_val_2) & 0xFFFF:04X}"
        if getattr(self, "lbl_modbus_reg1", None) is not None:
            self.lbl_modbus_reg1.setText(reg_text_1)
        if getattr(self, "lbl_modbus_reg2", None) is not None:
            self.lbl_modbus_reg2.setText(reg_text_2)

    def _handle_recognition_request(self, cam_index: int):
        reg_addr = RESULT_REGISTER_ADDR_1 if cam_index == 1 else RESULT_REGISTER_ADDR_2
        result_value = self._run_yolo_recognition(cam_index)
        self._publish_modbus_result(reg_addr, result_value)
        if self.modbus_model:
            self.modbus_model.set_register(TRIGGER_REGISTER_ADDR, 0)

    def _on_modbus_write(self, addr: int, value: int):
        if addr == TRIGGER_REGISTER_ADDR and value in (1, 2):
            print(f"[MODBUS] 收到触发请求 {value}")
            self.modbus_trigger_sig.emit(int(value))

    @QtCore.pyqtSlot(int)
    def _on_modbus_trigger(self, cam_index: int):
        self._handle_recognition_request(int(cam_index))

    def _stop_modbus_server(self):
        if not getattr(self, "modbus_server", None):
            return
        try:
            self.modbus_server.shutdown()
        except Exception:
            pass
        try:
            self.modbus_server.server_close()
        except Exception:
            pass
        self.modbus_server = None
        self._refresh_modbus_status()

    def _start_modbus_server(self, host: Optional[str] = None):
        if host is not None:
            self.modbus_host = host
        self._stop_modbus_server()
        self.modbus_error = None
        try:
            self.modbus_server = start_modbus_server(
                self.modbus_host, self.modbus_port, self.modbus_model, on_write=self._on_modbus_write
            )
        except Exception as exc:
            print(f"[MODBUS] 启动失败: {exc}")
            self.modbus_server = None
            self.modbus_error = str(exc)
        self._refresh_modbus_status()

    def _toast(self, text: str, ms: int = 2200):
        self.status_label.setText(text)
        self.toast_timer.start(ms)

    # ---------- 摄像头控制 ----------
    def _start_camera(self):
        self.stop_camera()
        src = self._current_source()
        self.active_slots = []

        if src == "hik":
            self._refresh_hik_devices()
            if not self.hik_devices:
                self._toast("未发现可用的海康相机")
                self._update_video_visibility()
                return

            max_slots = min(2, len(self.hik_devices))
            for slot in range(1, max_slots + 1):
                try:
                    grabber = HikGrabber(self, device_index=slot - 1)
                except Exception as exc:
                    self._toast(f"海康{slot}启动失败：{exc}")
                    continue
                self.grabbers[slot] = grabber
                grabber.frameSignal.connect(lambda frame, s=slot: self.on_frame(s, frame))
                grabber.infoSignal.connect(lambda msg, s=slot: self.on_info(s, msg))
                grabber.errorSignal.connect(self._on_grabber_error)
                grabber.start()
                self.active_slots.append(slot)

            self._update_video_visibility()
            return

        grabber = UsbGrabber(self, index=int(self.usb_index))
        self.grabbers[1] = grabber
        grabber.frameSignal.connect(lambda frame, s=1: self.on_frame(s, frame))
        grabber.infoSignal.connect(lambda msg, s=1: self.on_info(s, msg))
        grabber.errorSignal.connect(self._on_grabber_error)
        grabber.start()
        self.active_slots.append(1)
        self._update_video_visibility()

    @QtCore.pyqtSlot(str)
    def _on_grabber_error(self, message: str):
        self._toast(message)

    def reopen_camera(self):
        self._start_camera()

    def stop_camera(self):
        for grabber in list(self.grabbers.values()):
            if grabber and grabber.isRunning():
                try:
                    grabber.stop()  # type: ignore[attr-defined]
                except Exception:
                    pass
                grabber.wait(1000)
        self.grabbers.clear()
        self.active_slots = []
        self._update_video_visibility()

    # ---------- 信号槽 ----------
    @QtCore.pyqtSlot(int, np.ndarray)
    def on_frame(self, cam_index: int, frame_bgr: np.ndarray):
        t = time.time()
        last_ts = self._last_paint_ts.get(cam_index, 0.0)
        if (t - last_ts) < (1.0 / UI_PAINT_FPS):
            return
        self._last_paint_ts[cam_index] = t

        if frame_bgr is None or frame_bgr.size == 0:
            return

        self.last_frame_bgrs[cam_index] = frame_bgr
        if cam_index == 1:
            self.last_frame_bgr = frame_bgr

        frame_to_show = frame_bgr

        if getattr(self, "chk_yolo", None) is not None and self.chk_yolo.isChecked():
            self.yolo_enabled = True
            if self._ensure_yolo_model():
                frame_to_show = self._apply_yolo(frame_bgr)
        else:
            self.yolo_enabled = False

        rgb = cv2.cvtColor(frame_to_show, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QtGui.QImage(rgb.data, w, h, ch * w, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg)
        lbl = self._video_label_for(cam_index)
        if not lbl:
            return
        scaled = pix.scaled(
            lbl.size(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.FastTransformation,
        )
        lbl.setPixmap(scaled)

    @QtCore.pyqtSlot(int, str)
    def on_info(self, cam_index: int, s: str):
        cam_label = self.lbl_cam1 if cam_index == 1 else self.lbl_cam2
        fps_label = self.lbl_fps1 if cam_index == 1 else self.lbl_fps2

        if s.startswith("[INFO]"):
            cam_label.setText(f"摄像头{cam_index}: " + s.replace("[INFO]", "").strip())
        elif s.startswith("[FPS]"):
            fps_label.setText(f"FPS{cam_index}: " + s.replace("[FPS]", "").strip())
        elif s.startswith("[HIK]") or s.startswith("[USB]"):
            self._toast(s)

    def closeEvent(self, e):
        self._stop_modbus_server()
        self.stop_camera()
        super().closeEvent(e)


def main():
    app = QtWidgets.QApplication(sys.argv)
    try:
        app.setWindowIcon(QIcon(resource_path(APP_ICON)))
    except Exception:
        pass
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
