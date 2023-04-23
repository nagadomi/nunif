from tkinter import *
from tkinter import filedialog, LabelFrame
from tkinter.ttk import *
import tempfile, os, platform, base64, subprocess, threading

MAX_INT = 2147483647 ; MAX_INT_BIT = 9223372036854775807

root = Tk()
root.title("waifu2x gui")
try:
    root.iconphoto(False, PhotoImage(file=str(os.getcwd()) + "/waifu2x/gui/favicon.png"))
except:
    root.iconphoto(False, PhotoImage(file="./favicon.png"))
root.geometry("1000x400")

la = LabelFrame(root, text="input image")
la.place(x=0, y=0, height=40)

def open():
    global load_f
    load_f = filedialog.askopenfilename(initialdir='', title='input image', filetypes=(('png files', '*.png'), ('jpg files', '*.jpg'), ('all files', '*.*')))
    Label(la, text=load_f, background="white").pack(side="left", anchor="n")

my_btn = Button(la, text='select file', command=open).pack(side="left", anchor="n")

laa = LabelFrame(root, text="output image")
laa.place(x=0, y=40, height=40)

def save():
    global save_f
    save_f = filedialog.asksaveasfilename(initialdir='', initialfile='', title="output image", filetypes=(('png files', '*.png'), ('webp files', '*.webp'), ('jpeg files', '*.jpeg')))
    Label(laa, text=save_f, background="white").pack(side="left", anchor="n")

asd = Button(laa, text="select file", command=save).pack(side="left", anchor="n")

l1 = Labelframe(root, text="Noise & Upscaling")
l1.place(x=0, y=80)

RI_N = StringVar()

Label(l1, text="scale = scale2x, noise_scale = noise_scale2x").pack()

Combobox(l1, values=["scale", "scale2x", "scale4x", "noise", "noise_scale", "noise_scale2x", "noise_scale4x"], textvariable=RI_N, state="readonly").pack(side="left", anchor="n")

l1_1 = Labelframe(root, text="Noise Level")
l1_1.place(x=250, y=80)

DL = IntVar()

Label(l1_1, text="This is a noise level only use with noise options").pack()

Combobox(l1_1, values=[0, 1, 2, 3], textvariable=DL, state="readonly").pack(side="left", anchor="n")

l3 = Labelframe(root, text="Tile")
l3.place(x=0, y=140)

TSTV = IntVar()

Spinbox(l3, from_=0, to=MAX_INT, textvariable=TSTV).pack(side="left", anchor="n")

l5 = Labelframe(root, text="Batch")
l5.place(x=170, y=140)

BSV = IntVar()

Spinbox(l5, from_=0, to=MAX_INT, textvariable=BSV).pack(side="left", anchor="n")

l4 = Labelframe(root, text="TTA")
l4.place(x=0, y=200)

TTAC = IntVar()

Checkbutton(l4, text="TTA", variable=TTAC, offvalue=0, onvalue=1).pack(side="left", anchor="n")

l6 = Labelframe(root, text="AMP")
l6.place(x=50, y=200)

AMPC = IntVar()

Checkbutton(l6, text="AMP", variable=AMPC, offvalue=0, onvalue=1).pack(side="left", anchor="n")

l7 = LabelFrame(root, text="Image Library")
l7.place(x=105, y=200)

ILV = StringVar()

ILCG = Combobox(l7, values=["pil", "wand"], textvariable=ILV, state="readonly").pack(side="left", anchor="n")

l8 = LabelFrame(root, text="Depth")
l8.place(x=340, y=140)

Label(l8, text="bit-depth of output image. enabled only with wand").pack()

DV = IntVar()

Spinbox(l8, from_=0, to=MAX_INT, textvariable=DV).pack(side="left", anchor="n")

l8 = LabelFrame(root, text="Format")
l8.place(x=275, y=200)

FV = StringVar()

FC = Combobox(l8, values=["png", "webp", "jpeg"], textvariable=FV, state="readonly").pack(side="left", anchor="n")

l9 = LabelFrame(root, text="Model Dir")
l9.place(x=445, y=200)

MV = StringVar()

ME = Entry(l9, textvariable=MV)
ME.pack(side="left", anchor="n")
ME.insert(index=0, string="If this statement is left as is or no value is given, the default model position will be selected.")

l11 = Labelframe(root, text="GPU")
l11.pack(side="bottom")

Label(l11, text="GPU device ids. -1 for CPU").pack()

GPUV = IntVar()

Spinbox(l11, from_=-1, to=MAX_INT, textvariable=GPUV).pack(side="left", anchor="n")

l10 = LabelFrame(root, text="Start")
l10.pack()

def VariableGetVariable():
    """A function to get the user value entered in tkinter

    Returns:
        file_load
        file_save
        noise_upscaling
        noise_level
        tile
        batch
        depth
        tta
        amp
        lib
        format
        dir
    """
    file_load: str = load_f
    file_save: str = save_f
    #
    noise_upscaling: int = RI_N.get()
    noise_level: int = DL.get()
    tile: int = TSTV.get()
    batch: int = BSV.get()
    depth: int = DV.get()
    tta: int = TTAC.get()
    amp: int = AMPC.get()
    lib: str = ILV.get()
    format: str = FV.get()
    dir: str = MV.get()
    gpu: int = GPUV.get()
    #
    return file_load, file_save, noise_upscaling, noise_level, tile, batch, depth, tta, amp, lib, format, dir, gpu

class CmdThread (threading.Thread):
   def __init__(self, command, textvar):
        threading.Thread.__init__(self)
        self.command = command
        self.textvar = textvar

   def run(self):
        proc = subprocess.Popen(self.command, stdout=subprocess.PIPE)
        while not proc.poll():
            data = proc.stdout.readline()
            if data:
                print(data)
                self.textvar.set(data)
            else:
                break

def sc():

    logtext = StringVar()
    loglabel = Label(root, text=lambda:logtext.get(), textvariable=logtext)
    loglabel.pack()

    file_load, file_save, noise_upscaling, noise_level, tile, batch, depth, tta, amp, lib, format, dir, gpu = VariableGetVariable()

    print(file_load, file_save, noise_upscaling, noise_level, tile, batch, depth, tta, amp, lib, format, dir, gpu)

    file_load = f"--input {file_load} "; file_save = f"--output {file_save} "; noise_upscaling = f"--method {noise_upscaling} "; noise_level = f"--noise-level {noise_level} "; tile = f"--tile-size {tile} "; batch = f"--batch-size {batch} "; depth = f"--depth {depth} "; lib = f"--image-lib {lib} "; format = f"--format {format} "; gpu = f"--gpu {gpu} "
    #
    if tta == 1:
        tta = f"--tta "
    else:
        tta = ""
    #
    if amp == 1:
        amp = ""
    else:
        amp = f"--disable-amp "
    #
    if dir == "If this statement is left as is or no value is given, the default model position will be selected.":
        dir = ""
    else:
        dir = f"--model-dir {dir} "
    #

    op_l = [file_load, file_save, noise_upscaling, noise_level, tile, batch, depth, tta, amp, lib, format, dir]

    if platform.system() in "Windows":
        commands = "python -m waifu2x.cli "
    else:
        commands = "DEBUG=1 python3 -m waifu2x.cli "

    for op in op_l:
        commands += str(op)
    
    print(commands)

    commands = commands.split()

    thread = CmdThread(command=commands, textvar=logtext)
    thread.start()

st = Button(l10, text="start", command= lambda: sc()).pack()

def main():
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("Exit")
