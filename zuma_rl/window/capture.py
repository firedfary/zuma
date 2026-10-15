import ctypes
from ctypes import wintypes
from typing import Optional, Tuple
import numpy as np
import win32con

user32 = ctypes.windll.user32
gdi32 = ctypes.windll.gdi32
kernel32 = ctypes.windll.kernel32

# Configure 64-bit Windows ctypes argtypes and restypes to prevent handle truncation
user32.OpenDesktopW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
user32.OpenDesktopW.restype = wintypes.HANDLE
user32.SetThreadDesktop.argtypes = [wintypes.HANDLE]
user32.SetThreadDesktop.restype = wintypes.BOOL
user32.GetDC.argtypes = [wintypes.HWND]
user32.GetDC.restype = wintypes.HDC
user32.ReleaseDC.argtypes = [wintypes.HWND, wintypes.HDC]
user32.ReleaseDC.restype = ctypes.c_int
user32.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
user32.GetClientRect.restype = wintypes.BOOL
user32.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.POINT)]
user32.ClientToScreen.restype = wintypes.BOOL
user32.IsWindow.argtypes = [wintypes.HWND]
user32.IsWindow.restype = wintypes.BOOL
user32.PrintWindow.argtypes = [wintypes.HWND, wintypes.HDC, wintypes.UINT]
user32.PrintWindow.restype = wintypes.BOOL

# GDI32 64-bit signatures
gdi32.CreateCompatibleDC.argtypes = [wintypes.HDC]
gdi32.CreateCompatibleDC.restype = wintypes.HDC
gdi32.CreateCompatibleBitmap.argtypes = [wintypes.HDC, ctypes.c_int, ctypes.c_int]
gdi32.CreateCompatibleBitmap.restype = wintypes.HBITMAP
gdi32.SelectObject.argtypes = [wintypes.HDC, wintypes.HGDIOBJ]
gdi32.SelectObject.restype = wintypes.HGDIOBJ
gdi32.BitBlt.argtypes = [wintypes.HDC, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, wintypes.HDC, ctypes.c_int, ctypes.c_int, wintypes.DWORD]
gdi32.BitBlt.restype = wintypes.BOOL
gdi32.GetDIBits.argtypes = [wintypes.HDC, wintypes.HBITMAP, wintypes.UINT, wintypes.UINT, wintypes.LPVOID, ctypes.c_void_p, wintypes.UINT]
gdi32.GetDIBits.restype = ctypes.c_int
gdi32.DeleteObject.argtypes = [wintypes.HGDIOBJ]
gdi32.DeleteObject.restype = wintypes.BOOL
gdi32.DeleteDC.argtypes = [wintypes.HDC]
gdi32.DeleteDC.restype = wintypes.BOOL

def ensure_default_desktop(desktop_name: str = "Default") -> bool:
    """
    Ensure the current thread is attached to the interactive desktop (Default).
    This is necessary when running under service or isolated virtual desktops.
    """
    try:
        DESKTOP_ALL = 0x01FF
        hdesk = user32.OpenDesktopW(desktop_name, 0, False, DESKTOP_ALL)
        if hdesk:
            res = user32.SetThreadDesktop(hdesk)
            return bool(res)
    except Exception as e:
        print(f"[capture] Failed to switch desktop: {e}")
    return False

def find_zuma_window(window_class: str = "MainWindow", title_keywords: list = None) -> Optional[int]:
    """
    Find the Zuma window HWND using EnumDesktopWindows on the interactive desktop.
    """
    ensure_default_desktop("Default")
    if title_keywords is None:
        title_keywords = ["祖玛", "zuma", "Zuma"]

    found_hwnd = None
    WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)

    def enum_cb(h, l):
        nonlocal found_hwnd
        cls_buf = ctypes.create_unicode_buffer(256)
        user32.GetClassNameW(h, cls_buf, 256)
        cls_name = cls_buf.value
        
        # Check window class
        if cls_name == window_class:
            title_buf = ctypes.create_unicode_buffer(256)
            user32.GetWindowTextW(h, title_buf, 256)
            title = title_buf.value
            for kw in title_keywords:
                if kw in title:
                    found_hwnd = h
                    return False
        return True

    DESKTOP_ALL = 0x01FF
    hdesk = user32.OpenDesktopW("Default", 0, False, DESKTOP_ALL)
    if hdesk:
        user32.EnumDesktopWindows(hdesk, WNDENUMPROC(enum_cb), 0)
    
    # Fallback: direct FindWindowW
    if not found_hwnd:
        for kw in title_keywords:
            h = user32.FindWindowW(window_class, None)
            if h:
                found_hwnd = h
                break
                
    return found_hwnd

class ZumaScreenCapture:
    def __init__(self, hwnd: Optional[int] = None, expected_w: int = 640, expected_h: int = 480):
        ensure_default_desktop("Default")
        self.expected_w = expected_w
        self.expected_h = expected_h
        self.hwnd = hwnd or find_zuma_window()
        if not self.hwnd:
            print("[capture] Warning: Zuma window not found during initialization.")

    def update_hwnd(self) -> Optional[int]:
        self.hwnd = find_zuma_window()
        return self.hwnd

    def is_window_valid(self) -> bool:
        if not self.hwnd:
            return False
        return bool(user32.IsWindow(self.hwnd))

    def get_client_rect(self) -> Tuple[int, int]:
        if not self.is_window_valid():
            return 0, 0
        rect = wintypes.RECT()
        user32.GetClientRect(self.hwnd, ctypes.byref(rect))
        return rect.right - rect.left, rect.bottom - rect.top

    def get_screen_position(self) -> Tuple[int, int]:
        if not self.is_window_valid():
            return 0, 0
        pt = wintypes.POINT(0, 0)
        user32.ClientToScreen(self.hwnd, ctypes.byref(pt))
        return pt.x, pt.y

    def capture_frame(self) -> Optional[np.ndarray]:
        """
        Capture the Zuma window client area.
        Returns:
            np.ndarray of shape (H, W, 3) in RGB order, or None if failed.
        """
        if not self.is_window_valid():
            self.update_hwnd()
            if not self.is_window_valid():
                return None

        w, h = self.get_client_rect()
        if w <= 0 or h <= 0:
            w, h = self.expected_w, self.expected_h

        hwnd_dc = user32.GetDC(self.hwnd)
        if not hwnd_dc:
            return None

        mem_dc = gdi32.CreateCompatibleDC(hwnd_dc)
        hbitmap = gdi32.CreateCompatibleBitmap(hwnd_dc, w, h)
        old_bmp = gdi32.SelectObject(mem_dc, hbitmap)

        # PW_CLIENTONLY = 1 (Captures only client area without title bar / borders)
        success = user32.PrintWindow(self.hwnd, mem_dc, 1)
        if not success:
            # Fallback 1: BitBlt directly from client DC
            success = bool(gdi32.BitBlt(mem_dc, 0, 0, w, h, hwnd_dc, 0, 0, win32con.SRCCOPY))
            if not success:
                # Fallback 2: PW_RENDERFULLCONTENT = 2
                success = user32.PrintWindow(self.hwnd, mem_dc, 2)
                if not success:
                    user32.PrintWindow(self.hwnd, mem_dc, 0)

        # Read bitmap bits into numpy array
        class BITMAPINFOHEADER(ctypes.Structure):
            _fields_ = [
                ('biSize', wintypes.DWORD), ('biWidth', wintypes.LONG), ('biHeight', wintypes.LONG),
                ('biPlanes', wintypes.WORD), ('biBitCount', wintypes.WORD), ('biCompression', wintypes.DWORD),
                ('biSizeImage', wintypes.DWORD), ('biXPelsPerMeter', wintypes.LONG), ('biYPelsPerMeter', wintypes.LONG),
                ('biClrUsed', wintypes.DWORD), ('biClrImportant', wintypes.DWORD)
            ]

        class BITMAPINFO(ctypes.Structure):
            _fields_ = [('bmiHeader', BITMAPINFOHEADER), ('bmiColors', wintypes.DWORD * 3)]

        bmi = BITMAPINFO()
        bmi.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        bmi.bmiHeader.biWidth = w
        bmi.bmiHeader.biHeight = -h  # top-down DIB
        bmi.bmiHeader.biPlanes = 1
        bmi.bmiHeader.biBitCount = 32
        bmi.bmiHeader.biCompression = 0

        buffer = ctypes.create_string_buffer(w * h * 4)
        gdi32.GetDIBits(mem_dc, hbitmap, 0, h, buffer, ctypes.byref(bmi), 0)

        # Cleanup GDI handles
        gdi32.SelectObject(mem_dc, old_bmp)
        gdi32.DeleteObject(hbitmap)
        gdi32.DeleteDC(mem_dc)
        user32.ReleaseDC(self.hwnd, hwnd_dc)

        arr = np.frombuffer(buffer, dtype=np.uint8).reshape((h, w, 4))
        # Convert BGRA to RGB
        rgb = arr[:, :, [2, 1, 0]].copy()
        
        # If capture was black (e.g. window obscured or DWM composition issue), fallback to Desktop DC BitBlt
        if rgb.mean() < 1.0:
            rgb_fallback = self._capture_desktop_dc(w, h)
            if rgb_fallback is not None and rgb_fallback.mean() > 1.0:
                return rgb_fallback

        return rgb

    def _capture_desktop_dc(self, w: int, h: int) -> Optional[np.ndarray]:
        sx, sy = self.get_screen_position()
        screen_dc = user32.GetDC(0)
        if not screen_dc:
            return None

        mem_dc = gdi32.CreateCompatibleDC(screen_dc)
        hbitmap = gdi32.CreateCompatibleBitmap(screen_dc, w, h)
        old_bmp = gdi32.SelectObject(mem_dc, hbitmap)

        gdi32.BitBlt(mem_dc, 0, 0, w, h, screen_dc, sx, sy, win32con.SRCCOPY)

        class BITMAPINFOHEADER(ctypes.Structure):
            _fields_ = [
                ('biSize', wintypes.DWORD), ('biWidth', wintypes.LONG), ('biHeight', wintypes.LONG),
                ('biPlanes', wintypes.WORD), ('biBitCount', wintypes.WORD), ('biCompression', wintypes.DWORD),
                ('biSizeImage', wintypes.DWORD), ('biXPelsPerMeter', wintypes.LONG), ('biYPelsPerMeter', wintypes.LONG),
                ('biClrUsed', wintypes.DWORD), ('biClrImportant', wintypes.DWORD)
            ]
        class BITMAPINFO(ctypes.Structure):
            _fields_ = [('bmiHeader', BITMAPINFOHEADER), ('bmiColors', wintypes.DWORD * 3)]

        bmi = BITMAPINFO()
        bmi.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        bmi.bmiHeader.biWidth = w
        bmi.bmiHeader.biHeight = -h
        bmi.bmiHeader.biPlanes = 1
        bmi.bmiHeader.biBitCount = 32

        buf = ctypes.create_string_buffer(w * h * 4)
        gdi32.GetDIBits(mem_dc, hbitmap, 0, h, buf, ctypes.byref(bmi), 0)

        gdi32.SelectObject(mem_dc, old_bmp)
        gdi32.DeleteObject(hbitmap)
        gdi32.DeleteDC(mem_dc)
        user32.ReleaseDC(0, screen_dc)

        arr = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 4))
        return arr[:, :, [2, 1, 0]].copy()
