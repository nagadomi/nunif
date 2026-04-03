import wx
from wx import glcanvas
from OpenGL import GL
import torch
import threading
import time
import sys
import os
from collections import deque
import ctypes


POLLING_INTERVAL = 1.0 / 240.0


class _CUDART:
    def __init__(self, device_id=0):
        torch_dir = os.path.dirname(torch.__file__)
        site_packages = os.path.dirname(torch_dir)
        candidates = [
            os.path.join(torch_dir, "lib"),
            os.path.join(site_packages, "nvidia", "cuda_runtime", "lib"),
            os.path.join(site_packages, "nvidia", "cuda_runtime", "bin"),  # Windows
        ]
        cudart_path = None
        for lib_dir in candidates:
            if not os.path.exists(lib_dir):
                continue
            for f in os.listdir(lib_dir):
                if sys.platform == "win32":
                    if f.startswith("cudart64") and f.endswith(".dll"):
                        cudart_path = os.path.join(lib_dir, f)
                        break
                else:
                    if f.startswith("libcudart") and ".so" in f:
                        cudart_path = os.path.join(lib_dir, f)
                        break
            if cudart_path:
                break
        if not cudart_path:
            raise RuntimeError("Could not find cudart in torch/lib or nvidia/cuda_runtime/lib")

        try:
            if sys.platform == "win32":
                self.lib = ctypes.WinDLL(cudart_path)
            else:
                self.lib = ctypes.CDLL(cudart_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load {cudart_path}: {e}")

        # API Definitions
        self.lib.cudaSetDevice.argtypes = [ctypes.c_int]
        self.lib.cudaGraphicsGLRegisterBuffer.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint, ctypes.c_uint
        ]
        self.lib.cudaGraphicsUnregisterResource.argtypes = [ctypes.c_void_p]
        self.lib.cudaGraphicsMapResources.argtypes = [
            ctypes.c_int, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p
        ]
        self.lib.cudaGraphicsUnmapResources.argtypes = [
            ctypes.c_int, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p
        ]
        self.lib.cudaGraphicsResourceGetMappedPointer.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_void_p,
        ]
        self.lib.cudaMemcpy.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int
        ]
        self.lib.cudaSetDevice(device_id)

    def register_buffer(self, pbo_id):
        resource = ctypes.c_void_p()
        # 1 = cudaGraphicsRegisterFlagsNone
        res = self.lib.cudaGraphicsGLRegisterBuffer(ctypes.byref(resource), pbo_id, 1)
        if res != 0:
            raise RuntimeError(f"cudaGraphicsGLRegisterBuffer failed: {res}")
        return resource

    def unregister_resource(self, resource):
        self.lib.cudaGraphicsUnregisterResource(resource)

    def memcpy_d2d(self, dst_ptr, src_ptr, size):
        # 3 = cudaMemcpyDeviceToDevice
        res = self.lib.cudaMemcpy(dst_ptr, src_ptr, size, 3)
        if res != 0:
            raise RuntimeError(f"cudaMemcpy failed: {res}")

    def map_resource(self, resource):
        res = self.lib.cudaGraphicsMapResources(1, ctypes.byref(resource), None)
        if res != 0:
            raise RuntimeError(f"cudaGraphicsMapResources failed: {res}")

        ptr = ctypes.c_void_p()
        size = ctypes.c_size_t()
        res = self.lib.cudaGraphicsResourceGetMappedPointer(ctypes.byref(ptr), ctypes.byref(size), resource)
        if res != 0:
            self.lib.cudaGraphicsUnmapResources(1, ctypes.byref(resource), None)
            raise RuntimeError(f"cudaGraphicsResourceGetMappedPointer failed: {res}")
        return ptr.value

    def unmap_resource(self, resource):
        self.lib.cudaGraphicsUnmapResources(1, ctypes.byref(resource), None)


class GLCanvas(glcanvas.GLCanvas):
    def __init__(self, parent, width, height,
                 use_cuda=False, device_id=0,
                 uncap_fps=False, polling_interval=POLLING_INTERVAL):
        attribs = [
            glcanvas.WX_GL_RGBA,
            glcanvas.WX_GL_DOUBLEBUFFER,
        ]
        super().__init__(parent, attribList=attribs, size=(width, height))
        self.context = glcanvas.GLContext(self)
        self.initialized = False
        self.closed = False
        self.fps_counter = deque(maxlen=120)

        self.tex_id = None
        self.pbo = None
        self.tex_w = width
        self.tex_h = height
        self.frame = None
        self.ready_event = None

        self.use_cuda = use_cuda
        self.device_id = device_id
        self.polling_interval = polling_interval
        self.cuda_resource = None
        self._cudart = None

        if uncap_fps:
            self.Bind(wx.EVT_IDLE, self.on_idle)
        self.Bind(wx.EVT_PAINT, self.on_paint)
        self.Bind(wx.EVT_SIZE, self.on_resize)
        self.Bind(wx.EVT_ERASE_BACKGROUND, lambda e: None)

    def init_gl(self, evt=None):
        self.SetCurrent(self.context)
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glClearColor(0, 0, 0, 1)
        GL.glViewport(0, 0, *self.GetClientSize())

        self.tex_id = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.tex_id)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGB8, self.tex_w, self.tex_h, 0,
                        GL.GL_RGB, GL.GL_UNSIGNED_BYTE, None)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

        self.pbo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, self.pbo)
        GL.glBufferData(GL.GL_PIXEL_UNPACK_BUFFER, self.tex_w * self.tex_h * 4, None, GL.GL_STREAM_DRAW)
        GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, 0)

        if self.use_cuda:
            try:
                self._cudart = _CUDART(self.device_id)
                self.cuda_resource = self._cudart.register_buffer(self.pbo)
            except Exception as e:
                print(f"Failed to initialize CUDA-GL Interop: {e}", file=sys.stderr)
                self.use_cuda = False

        self.initialized = True

    def delete_gl(self):
        if not self.initialized:
            return
        self.initialized = False
        self.SetCurrent(self.context)

        if self.cuda_resource:
            self._cudart.unregister_resource(self.cuda_resource)
            self.cuda_resource = None

        if self.tex_id:
            GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
            GL.glDeleteTextures([self.tex_id])
            self.tex_id = None
        if self.pbo:
            GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, 0)
            GL.glDeleteBuffers(1, [self.pbo])
            self.pbo = None

    def destroy(self):
        self.closed = True
        self.delete_gl()

    def update_frame(self, frame, ready_event=None):
        if self.closed:
            return
        if not self.initialized:
            # skip
            self.Refresh()
            return

        assert (
            frame.ndim == 3 and
            frame.shape[0] == 3 and
            frame.shape[1] == self.tex_h and
            frame.shape[2] == self.tex_w and
            frame.dtype == torch.float32
        )
        self.frame = frame
        self.ready_event = ready_event
        self.Refresh()

    def set_tex(self):
        if self.frame is None:
            return False

        frame = self.frame
        ready_event = self.ready_event
        c, h, w = frame.shape
        self.tex_w = w
        self.tex_h = h

        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, self.pbo)

        if ready_event is not None and frame.is_cuda:
            torch.cuda.current_stream(device=frame.device).wait_event(ready_event)

        if self.use_cuda and frame.is_cuda:
            # Ensure the frame is on the same device as the OpenGL PBO
            if frame.get_device() != self.device_id:
                frame = frame.to(f"cuda:{self.device_id}")

            # Zero-copy transfer using CUDA-GL Interop
            # 1. Convert to uint8 on GPU
            frame = frame.permute(1, 2, 0).contiguous()
            frame = (frame.clamp(0, 1) * 255).to(torch.uint8)

            # 2. Map PBO to CUDA and copy
            ptr = self._cudart.map_resource(self.cuda_resource)
            try:
                self._cudart.memcpy_d2d(ptr, frame.data_ptr(), frame.nbytes)
            finally:
                self._cudart.unmap_resource(self.cuda_resource)
        else:
            # Fallback to CPU transfer
            frame = frame.permute(1, 2, 0).contiguous()
            frame = (frame.clamp(0, 1) * 255).to(torch.uint8).detach().cpu().numpy()

            ptr = GL.glMapBuffer(GL.GL_PIXEL_UNPACK_BUFFER, GL.GL_WRITE_ONLY)
            ctypes.memmove(ptr, frame.ctypes.data, frame.nbytes)
            GL.glUnmapBuffer(GL.GL_PIXEL_UNPACK_BUFFER)

        GL.glBindTexture(GL.GL_TEXTURE_2D, self.tex_id)
        GL.glTexSubImage2D(GL.GL_TEXTURE_2D, 0, 0, 0, w, h, GL.GL_RGB, GL.GL_UNSIGNED_BYTE, None)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, 0)

        self.frame = None
        self.ready_event = None
        self.fps_counter.append(time.perf_counter())

        return True

    def draw(self):
        W, H = self.GetClientSize()
        GL.glViewport(0, 0, W, H)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)

        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        GL.glOrtho(0, W, H, 0, -1, 1)
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glLoadIdentity()

        GL.glEnable(GL.GL_TEXTURE_2D)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.tex_id)

        scale = min(W / self.tex_w, H / self.tex_h)
        draw_w = self.tex_w * scale
        draw_h = self.tex_h * scale
        x0 = (W - draw_w) / 2
        y0 = (H - draw_h) / 2
        x1 = x0 + draw_w
        y1 = y0 + draw_h

        GL.glBegin(GL.GL_QUADS)

        GL.glTexCoord2f(0, 0)
        GL.glVertex2f(x0, y0)

        GL.glTexCoord2f(1, 0)
        GL.glVertex2f(x1, y0)

        GL.glTexCoord2f(1, 1)
        GL.glVertex2f(x1, y1)

        GL.glTexCoord2f(0, 1)
        GL.glVertex2f(x0, y1)

        GL.glEnd()

        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        GL.glDisable(GL.GL_TEXTURE_2D)

    def on_idle(self, evt):
        if self.closed:
            return
        if not self.initialized:
            return
        if not self.render():
            if self.polling_interval > 0.0:
                time.sleep(self.polling_interval)
        evt.RequestMore()

    def on_paint(self, evt):
        if self.closed:
            return
        if not self.initialized:
            self.init_gl()
        self.render()

    def render(self):
        self.SetCurrent(self.context)
        if self.set_tex():
            self.draw()
            self.SwapBuffers()
            return True
        return False

    def on_resize(self, evt):
        if self.closed:
            return
        if self.initialized:
            self.SetCurrent(self.context)
            w, h = self.GetClientSize()
            GL.glViewport(0, 0, max(1, w), max(1, h))
        evt.Skip()
        self.Refresh()

    def get_fps(self):
        if self.closed:
            return 0.0

        diff = []
        prev = None
        for t in self.fps_counter:
            if prev is not None:
                diff.append(t - prev)
            prev = t
        if diff:
            fps = 1.0 / (sum(diff) / len(diff))
        else:
            fps = 0.0
        return fps


class LocalViewerWindow(wx.Frame):
    def __init__(self, width, height, size=(960, 540),
                 use_cuda=False, device_id=0,
                 uncap_fps=False, polling_interval=POLLING_INTERVAL):
        super().__init__(None, title="iw3-desktop: Local Viewer",
                         size=size, style=wx.DEFAULT_FRAME_STYLE | wx.CLIP_CHILDREN)
        self.canvas = GLCanvas(self, width=width, height=height,
                               use_cuda=use_cuda, device_id=device_id,
                               uncap_fps=uncap_fps, polling_interval=polling_interval)

        self.Bind(wx.EVT_CLOSE, self.on_close)
        self.Bind(wx.EVT_CHAR_HOOK, self.on_char)

    def toggle_fullscreen(self):
        is_full = self.IsFullScreen()
        self.ShowFullScreen(not is_full, style=wx.FULLSCREEN_ALL)

    def escape_fullscreen(self):
        self.ShowFullScreen(False, style=wx.FULLSCREEN_ALL)

    def on_char(self, evt):
        code = evt.GetKeyCode()
        if code == wx.WXK_ESCAPE and self.IsFullScreen():
            self.toggle_fullscreen()
        elif code == wx.WXK_F11:
            self.toggle_fullscreen()
        else:
            evt.Skip()

    def update_frame(self, frame, ready_event=None):
        if not self.canvas.closed:
            self.canvas.update_frame(frame, ready_event=ready_event)

    def get_fps(self):
        return self.canvas.get_fps()

    def on_close(self, evt):
        self.canvas.destroy()
        evt.Skip()

    def is_closed(self):
        return self.canvas.closed


class LocalViewer():
    def __init__(self, lock, width, height,
                 use_cuda=False, device_id=0,
                 uncap_fps=False, polling_interval=POLLING_INTERVAL,
                 **_unsupported_kwargs):
        self.width = width
        self.height = height
        self.lock = lock
        self.op_lock = threading.RLock()
        self.window = None
        self.initialized = False
        self.last_frame_time = 0
        self.use_cuda = use_cuda
        self.device_id = device_id
        self.uncap_fps = uncap_fps
        self.polling_interval = polling_interval

    def stop(self):
        with self.op_lock:
            if self.window is not None:
                window = self.window
                self.window = None
                if not window.is_closed():
                    wx.CallAfter(window.Close)

    def _start(self):
        with self.op_lock:
            if self.window is None:
                self.window = LocalViewerWindow(width=self.width, height=self.height,
                                                use_cuda=self.use_cuda, device_id=self.device_id,
                                                uncap_fps=self.uncap_fps, polling_interval=self.polling_interval)
                self.window.Show()
                self.initialized = True

    def start(self):
        """
        Standard start method for use when wx.App is already running on main thread.
        This is used on Linux with wx.Yield() pattern, and on Windows when called from worker thread.
        """
        if not wx.GetApp():
            raise RuntimeError("wx.App is not initialized")
        wx.CallAfter(self._start)

    def set_frame_data(self, frame_data):
        with self.op_lock:
            if len(frame_data) == 2:
                frame, frame_time = frame_data
                ready_event = None
            else:
                frame, frame_time, ready_event = frame_data
            if self.window is not None and self.last_frame_time < frame_time:
                wx.CallAfter(self.window.update_frame, frame, ready_event)
                self.last_frame_time = frame_time

    def get_fps(self):
        if self.window is not None:
            return self.window.get_fps()
        else:
            return 0.0

    def is_closed(self):
        with self.op_lock:
            if self.window:
                return self.window.is_closed()
            else:
                if self.initialized:
                    return True
                else:
                    return False


def run_local_viewer_cli(worker_callback):
    """
    Platform-aware entry point for running local viewer from CLI.

    Handles platform-specific requirements:
    - Linux: Calls worker directly, which creates wx.App and uses wx.Yield() pattern
    - Windows: Creates wx.App on main thread, runs worker in background thread

    Args:
        worker_callback: Function to run (the processing loop). Should call iw3_desktop_main.
                        On Windows, should NOT create wx.App (init_wxapp=False)
                        On Linux, SHOULD create wx.App (init_wxapp=True)

    This function blocks until the GUI is closed.
    """
    if sys.platform != "win32":
        # Linux: Original wx.Yield() pattern works fine
        # Worker will create wx.App and use wx.Yield() to pump events
        return worker_callback()

    # Windows: Need wx event loop on main thread
    app = wx.App()

    # Create a hidden dummy frame to keep the app alive
    # Without this, app.MainLoop() exits immediately before worker can create the real window
    dummy_frame = wx.Frame(None)
    dummy_frame.Hide()

    # Track worker state
    worker_started = threading.Event()
    worker_exception = [None]

    def worker_target():
        try:
            worker_started.set()
            worker_callback()
        except Exception as e:
            worker_exception[0] = e
            print(f"LocalViewer worker error: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
        finally:
            # Exit the event loop when worker finishes
            wx.CallAfter(app.ExitMainLoop)

    worker_thread = threading.Thread(target=worker_target, daemon=False)

    def start_worker():
        worker_thread.start()
        if not worker_started.wait(timeout=5.0):
            print("Warning: Worker thread did not start within 5 seconds", file=sys.stderr)

    # Start worker after event loop begins
    wx.CallAfter(start_worker)

    # Run event loop on main thread (Windows requirement)
    app.MainLoop()

    # Cleanup
    dummy_frame.Destroy()

    # Wait for worker to finish
    if worker_thread and worker_thread.is_alive():
        worker_thread.join(timeout=5.0)

    # Re-raise any exception from worker
    if worker_exception[0]:
        raise worker_exception[0]


def _test():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--cuda", action="store_true", help="use cudaMemcpy")
    parser.add_argument("--size", choices=["hd", "4k", "8k"], default="4k", help="frame size")
    parser.add_argument("--fps", type=int, default=2000, help="frame interval")
    args = parser.parse_args()

    # 4K(SBS)
    # on RTX 3070 Ti Linux,
    #    with --cuda: 293FPS
    # without --cuda:  44FPS
    if args.size == "hd":
        W, H = (1920 * 2, 1080)
    elif args.size == "4k":
        W, H = (3840 * 2, 2160)
    elif args.size == "8k":
        W, H = (7680 * 2, 4320)

    def main():
        lock = threading.RLock()
        server = LocalViewer(lock, width=W, height=H, use_cuda=args.cuda, uncap_fps=True, polling_interval=0)
        server.start()
        time.sleep(2)
        frames = torch.rand(4, 3, H, W).cuda()
        torch.cuda.synchronize()
        time.sleep(1)
        for i in range(300):
            if server.is_closed():
                break
            frame = frames[i % frames.shape[0]]
            server.set_frame_data((frame, time.perf_counter()))
            time.sleep(1 / args.fps)

        print(f"{W}x{H}, {round(server.get_fps(), 2)}FPS")
        print("CTRL+C to exit")
        server.stop()

    def start():
        threading.Thread(target=main).start()

    app = wx.App()
    frame = wx.Frame(None)  # noqa
    wx.CallAfter(start)
    app.MainLoop()


if __name__ == '__main__':
    _test()
