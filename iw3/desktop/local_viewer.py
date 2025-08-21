import wx
from wx import glcanvas
from OpenGL import GL
import torch
import threading
import time
from collections import deque
import ctypes


class GLCanvas(glcanvas.GLCanvas):
    def __init__(self, parent, width, height):
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

        self.initialized = True

    def delete_gl(self):
        if not self.initialized:
            return
        self.initialized = False
        self.SetCurrent(self.context)
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

    def update_frame(self, frame):
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
        self.Refresh()

    def set_tex(self):
        if self.frame is None:
            return

        frame = self.frame
        c, h, w = frame.shape
        self.tex_w = w
        self.tex_h = h

        frame = frame.permute(1, 2, 0).contiguous()
        frame = (frame.clamp(0, 1) * 255).to(torch.uint8).detach().cpu().numpy()

        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)

        GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, self.pbo)
        ptr = GL.glMapBuffer(GL.GL_PIXEL_UNPACK_BUFFER, GL.GL_WRITE_ONLY)
        ctypes.memmove(ptr, frame.ctypes.data, frame.nbytes)
        GL.glUnmapBuffer(GL.GL_PIXEL_UNPACK_BUFFER)

        GL.glBindTexture(GL.GL_TEXTURE_2D, self.tex_id)
        GL.glTexSubImage2D(GL.GL_TEXTURE_2D, 0, 0, 0, w, h, GL.GL_RGB, GL.GL_UNSIGNED_BYTE, None)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

        self.frame = None
        self.fps_counter.append(time.perf_counter())

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

    def on_paint(self, evt):
        if self.closed:
            return
        if not self.initialized:
            self.init_gl()

        self.SetCurrent(self.context)
        self.set_tex()
        self.draw()
        self.SwapBuffers()

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
    def __init__(self, width, height, size=(960, 540)):
        super().__init__(None, title="iw3-desktop: Local Viewer", size=size, style=wx.DEFAULT_FRAME_STYLE | wx.CLIP_CHILDREN)
        self.canvas = GLCanvas(self, width=width, height=height)

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

    def update_frame(self, frame):
        if not self.canvas.closed:
            self.canvas.update_frame(frame)

    def get_fps(self):
        return self.canvas.get_fps()

    def on_close(self, evt):
        self.canvas.destroy()
        evt.Skip()

    def is_closed(self):
        return self.canvas.closed


class LocalViewer():
    def __init__(self, lock, width, height, **_unsupported_kwargs):
        self.width = width
        self.height = height
        self.lock = lock
        self.op_lock = threading.RLock()
        self.window = None
        self.initialized = False
        self.last_frame_time = 0

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
                self.window = LocalViewerWindow(width=self.width, height=self.height)
                self.window.Show()
                self.initialized = True

    def start(self):
        if not wx.GetApp():
            raise RuntimeError("wx.App is not initialized")
        wx.CallAfter(self._start)

    def set_frame_data(self, frame_data):
        with self.op_lock:
            frame, frame_time = frame_data
            if self.window is not None and self.last_frame_time < frame_time:
                wx.CallAfter(self.window.update_frame, frame)
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


def _test():
    W = 3840
    H = 2160
    FPS = 60

    def main():
        lock = threading.RLock()
        server = LocalViewer(lock, width=W, height=H)
        server.start()
        frames = torch.rand(8, 3, H, W).cuda()
        for i in range(300):
            frame = frames[i % frames.shape[0]]
            server.set_frame_data((frame, time.time()))
            time.sleep(1 / FPS)
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
