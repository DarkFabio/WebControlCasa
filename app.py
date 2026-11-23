#!/usr/bin/env python3
"""
Domotica casa - NiceGUI + Siemens S7-1200 (snap7) + webcam USB (OpenCV)

Installazione (Debian):
    sudo apt install python3-venv libsnap7-1 libgl1
    python3 -m venv venv && . venv/bin/activate
    pip install nicegui python-snap7 opencv-python-headless
    python domotica.py        ->  http://IP_DEL_PC:8080

Lato PLC (TIA Portal):
  - i DB usati devono avere "Accesso ottimizzato al blocco" DISATTIVATO
  - CPU > Protezione > "Consenti accesso PUT/GET dal partner remoto" ATTIVO
"""
import asyncio
import logging
import threading
import time

import cv2
import numpy as np
from fastapi.responses import StreamingResponse
from nicegui import app, run, ui

try:
    import snap7
    from snap7.util import get_bool, get_int, set_bool, set_int
except Exception:  # snap7 non installato: l'app parte lo stesso
    snap7 = None

log = logging.getLogger("domotica")
logging.basicConfig(level=logging.INFO)

# =====================================================================
#  CONFIGURAZIONE
# =====================================================================
PLC_IP, PLC_RACK, PLC_SLOT = "192.168.1.100", 0, 1
POLL_SECONDS = 1.0
PORT = 8080

# Indirizzo bit = (numero DB, byte, bit)  ->  es. DB1.DBX0.0 = (1, 0, 0)
# "write": bit che scrive il pulsante; "read": bit che dice se la luce è accesa.
# mode: "pulse" = scrive 1 per 0.3 s e poi 0 (il PLC fa il toggle)
#       "toggle" = scrive il contrario dello stato letto (read)
LIGHT_GROUPS = {
    "Esterno": ("terrace", [
        ("Luce scale",    "stairs",     dict(write=(4, 2, 2), read=(4, 2, 2))),
        ("Luce terrazza", "deck",       dict(write=(4, 3, 4), read=(4, 3, 4))),
    ]),
    "Zona giorno": ("weekend", [
        ("Luce sala",     "chair",      dict(write=(4, 2, 0), read=(4, 2, 0))),
        ("Luce sala 1",   "chair",      dict(write=(5, 1, 4), read=(5, 1, 4))),
        ("Luce sala 2",   "chair",      dict(write=(4, 2, 6), read=(4, 2, 6))),
        ("Luce cucina",   "kitchen",    dict(write=(4, 2, 4), read=(4, 2, 4))),
        ("Luce corridoio", "meeting_room", dict(write=(5, 1, 0), read=(5, 1, 0))),
    ]),
    "Zona notte": ("bed", [
        ("Luce camera sx", "bed",       dict(write=(3, 1, 0), read=(3, 1, 0))),
        ("Luce camera dx", "bed",       dict(write=(5, 1, 7), read=(5, 1, 7))),
        ("Luce studio",    "desk",      dict(write=(5, 1, 6), read=(5, 1, 6))),
        ("Luce bagno",     "bathtub",   dict(write=(3, 1, 1), read=(3, 1, 1))),
    ]),
    "Servizi": ("home_repair_service", [
        ("Luce lavanderia", "local_laundry_service", dict(write=(5, 1, 3), read=(5, 1, 3))),
        ("Luce solaio",     "roofing",  dict(write=(3, 1, 2), read=(3, 1, 2))),
    ]),
    "Garage": ("garage", [
        ("Luce garage",       "garage",   dict(write=(1, 1, 0), read=(1, 1, 0))),
        ("Luce bagno garage", "bathtub",  dict(write=(1, 1, 1), read=(1, 1, 1))),
        ("Luce banco garage", "handyman", dict(write=(1, 1, 3), read=(1, 1, 3))),
    ]),
}
DEFAULT_MODE = "toggle"

# Fari: variabile INT (DB, byte) = minuti di accensione. Il PLC fa il resto.
# Ogni pressione aggiunge FARI_STEP minuti (max FARI_MAX).
FARI = [
    ("Faro nord",  (101, 0)),
    ("Faro sud",   (101, 1)),
    ("Faro est",   (101, 2)),
    ("Faro ovest", (101, 3)),
]
FARI_STEP, FARI_MAX = 1, 60

# SCENE. Ogni scena può avere (tutte le chiavi sono opzionali, combinabili):
#   all_off : True            -> l'app spegne tutte le luci accese
#   on      : [nomi luci]     -> l'app accende quelle attualmente spente
#   off     : [nomi luci]     -> l'app spegne quelle attualmente accese
#   plc_bit : (DB, byte, bit) -> impulso su un bit del PLC (la logica la fa il PLC)
# I nomi in on/off devono coincidere con quelli di LIGHT_GROUPS.
SCENES = [
    dict(name="Spegni tutto", icon="power_settings_new", color="red-8", all_off=True),
    dict(name="Sera", icon="nights_stay", color="indigo-7",
         on=["Luce sala 1", "Luce corridoio"], off=["Luce cucina", "Luce sala 2"]),
    dict(name="Cena", icon="restaurant", color="orange-9",
         on=["Luce cucina", "Luce sala 2"], off=["Luce sala", "Luce sala 1"]),
    dict(name="Notte", icon="bedtime", color="deep-purple-8", plc_bit=(1, 5, 0)),
    dict(name="Garage", icon="garage", color="teal-8",
         on=["Luce garage", "Luce banco garage"]),
]
SCENE_PULSE_SECONDS = 0.5

# Telecamere USB: (nome, indice /dev/videoN). Quelle assenti vengono ritentate
# ogni 5 secondi senza bloccare nulla. Puoi aggiungerne quando le colleghi.
CAMERAS = [
    ("Ingresso", 0),
    ("Giardino", 2),
    ("Garage",   4),
]
CAM_W, CAM_H, CAM_FPS = 640, 480, 8
# =====================================================================


# ------------------------------- PLC ---------------------------------
class PLC:
    def __init__(self):
        self.lock = threading.Lock()
        self.client = snap7.client.Client() if snap7 else None
        self.connected = False
        self.lights: dict[str, bool | None] = {}
        self.fari: dict[str, int | None] = {}

    def _connect(self):
        if not self.client:
            return False
        try:
            try:
                self.client.disconnect()
            except Exception:
                pass
            self.client.connect(PLC_IP, PLC_RACK, PLC_SLOT)
            self.connected = self.client.get_connected()
        except Exception as e:
            self.connected = False
            log.warning("PLC non raggiungibile: %s", e)
        return self.connected

    def _read_bit(self, addr):
        db, byte, bit = addr
        return get_bool(self.client.db_read(db, byte, 1), 0, bit)

    def _write_bit(self, addr, value):
        db, byte, bit = addr
        data = self.client.db_read(db, byte, 1)
        set_bool(data, 0, bit, value)
        self.client.db_write(db, byte, data)

    def poll(self):
        """Legge tutto (eseguito in thread). Mai eccezioni verso l'esterno."""
        with self.lock:
            if not self.connected and not self._connect():
                self._mark_unknown()
                return
            try:
                for _, (_, items) in LIGHT_GROUPS.items():
                    for name, _, cfg in items:
                        self.lights[name] = self._read_bit(cfg["read"])
                for name, (db, byte) in FARI:
                    self.fari[name] = get_int(self.client.db_read(db, byte, 2), 0)
            except Exception as e:
                log.warning("Errore lettura PLC: %s", e)
                self.connected = False
                self._mark_unknown()

    def _mark_unknown(self):
        for k in list(self.lights):
            self.lights[k] = None
        for k in list(self.fari):
            self.fari[k] = None

    def press_light(self, name, cfg):
        with self.lock:
            if not self.connected and not self._connect():
                return False
            try:
                mode = cfg.get("mode", DEFAULT_MODE)
                if mode == "toggle":
                    state = self._read_bit(cfg["read"])
                    self._write_bit(cfg["write"], not state)
                else:
                    self._write_bit(cfg["write"], True)
                    time.sleep(0.3)
                    self._write_bit(cfg["write"], False)
                return True
            except Exception as e:
                log.warning("Errore scrittura PLC: %s", e)
                self.connected = False
                return False

    def run_scene(self, scene):
        """Esegue una scena. Ritorna il numero di azioni fatte, o None se PLC offline."""
        with self.lock:
            if not self.connected and not self._connect():
                return None
            try:
                lights = {n: cfg for _, (_, items) in LIGHT_GROUPS.items() for n, _, cfg in items}
                on = list(scene.get("on", []))
                off = list(lights) if scene.get("all_off") else list(scene.get("off", []))
                for n in on + off:
                    if n not in lights:
                        log.warning("Scena '%s': luce sconosciuta '%s'", scene["name"], n)

                pulse_bits = []
                for n in on:
                    cfg = lights.get(n)
                    if cfg and self._read_bit(cfg["read"]) is False:
                        if cfg.get("mode", DEFAULT_MODE) == "toggle":
                            self._write_bit(cfg["write"], True)
                        else:
                            pulse_bits.append(cfg["write"])
                for n in off:
                    cfg = lights.get(n)
                    if cfg and self._read_bit(cfg["read"]) is True:
                        if cfg.get("mode", DEFAULT_MODE) == "toggle":
                            self._write_bit(cfg["write"], False)
                        else:
                            pulse_bits.append(cfg["write"])
                if scene.get("plc_bit"):
                    pulse_bits.append(scene["plc_bit"])

                # tutti gli impulsi insieme: una sola attesa per l'intera scena
                for addr in pulse_bits:
                    self._write_bit(addr, True)
                if pulse_bits:
                    time.sleep(SCENE_PULSE_SECONDS)
                    for addr in pulse_bits:
                        self._write_bit(addr, False)
                return len(pulse_bits)
            except Exception as e:
                log.warning("Errore scena: %s", e)
                self.connected = False
                return None

    def add_faro_time(self, name, db, byte, delta):
        """delta > 0 aggiunge minuti, delta == 0 azzera."""
        with self.lock:
            if not self.connected and not self._connect():
                return False
            try:
                cur = get_int(self.client.db_read(db, byte, 2), 0)
                new = 0 if delta == 0 else min(cur + delta, FARI_MAX)
                data = bytearray(2)
                set_int(data, 0, new)
                self.client.db_write(db, byte, data)
                return True
            except Exception as e:
                log.warning("Errore scrittura faro: %s", e)
                self.connected = False
                return False


plc = PLC()


# ----------------------------- TELECAMERE ----------------------------
def _placeholder(text):
    img = np.zeros((CAM_H, CAM_W, 3), np.uint8)
    img[:] = (40, 40, 40)
    cv2.putText(img, text, (30, CAM_H // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (160, 160, 160), 2)
    return cv2.imencode(".jpg", img)[1].tobytes()


class Camera(threading.Thread):
    def __init__(self, name, index):
        super().__init__(daemon=True)
        self.name, self.index = name, index
        self.jpeg = _placeholder("Non disponibile")
        self.online = False

    def run(self):
        cap = None
        while True:
            try:
                if cap is None or not cap.isOpened():
                    self.online = False
                    self.jpeg = _placeholder("Non disponibile")
                    cap = cv2.VideoCapture(self.index, cv2.CAP_V4L2)
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
                    if not cap.isOpened():
                        cap.release()
                        cap = None
                        time.sleep(5)
                        continue
                ok, frame = cap.read()
                if not ok:
                    cap.release()
                    cap = None
                    time.sleep(2)
                    continue
                self.jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes()
                self.online = True
                time.sleep(1 / CAM_FPS)
            except Exception as e:
                log.warning("Camera %s: %s", self.name, e)
                try:
                    cap and cap.release()
                except Exception:
                    pass
                cap = None
                time.sleep(5)


cameras = [Camera(n, i) for n, i in CAMERAS]
for c in cameras:
    c.start()


@app.get("/video/{cam_id}")
async def video(cam_id: int):
    if not 0 <= cam_id < len(cameras):
        return StreamingResponse(iter([]), status_code=404)

    async def gen():
        cam = cameras[cam_id]
        while True:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + cam.jpeg + b"\r\n"
            await asyncio.sleep(1 / CAM_FPS)

    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


# --------------------------------- UI --------------------------------
ON_COLOR, OFF_COLOR, UNK_COLOR = "amber-6", "blue-grey-8", "red-10"


@ui.page("/")
def main_page():
    ui.add_head_html(
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">'
        '<meta name="apple-mobile-web-app-capable" content="yes">'
    )
    ui.dark_mode(True)

    light_buttons = {}   # name -> (button, icon)
    faro_labels = {}     # name -> label

    # Header con stato PLC
    with ui.header().classes("items-center justify-between bg-slate-900"):
        ui.label("Casa").classes("text-xl font-bold")
        plc_badge = ui.badge("PLC ...").props("rounded")

    async def on_light(name, cfg):
        ok = await run.io_bound(plc.press_light, name, cfg)
        if not ok:
            ui.notify("PLC non raggiungibile", type="negative")
        await refresh()

    async def on_faro(name, db, byte, delta):
        ok = await run.io_bound(plc.add_faro_time, name, db, byte, delta)
        if not ok:
            ui.notify("PLC non raggiungibile", type="negative")
        await refresh()

    async def on_scene(scene):
        res = await run.io_bound(plc.run_scene, scene)
        if res is None:
            ui.notify("PLC non raggiungibile", type="negative")
        else:
            ui.notify(f"Scena «{scene['name']}» eseguita", type="positive")
        await refresh()

    with ui.tabs().classes("w-full") as tabs:
        t_luci = ui.tab("Luci", icon="lightbulb")
        t_scene = ui.tab("Scene", icon="auto_awesome")
        t_fari = ui.tab("Fari", icon="flare")
        t_cam = ui.tab("Camere", icon="videocam")

    with ui.tab_panels(tabs, value=t_luci).classes("w-full bg-transparent"):
        # ---- LUCI ----
        with ui.tab_panel(t_luci):
            for gname, (gicon, items) in LIGHT_GROUPS.items():
                with ui.card().classes("w-full mb-3"):
                    with ui.row().classes("items-center"):
                        ui.icon(gicon, size="sm")
                        ui.label(gname).classes("text-lg font-semibold")
                    with ui.grid(columns=2).classes("w-full gap-2 md:grid-cols-4"):
                        for name, icon, cfg in items:
                            b = ui.button(
                                name.replace("Luce ", "").capitalize(),
                                icon=icon,
                                on_click=lambda n=name, c=cfg: on_light(n, c),
                            ).props("no-caps size=lg").classes("w-full h-20")
                            light_buttons[name] = (b, icon)

        # ---- SCENE ----
        with ui.tab_panel(t_scene):
            with ui.grid(columns=2).classes("w-full gap-3 md:grid-cols-4"):
                for sc in SCENES:
                    ui.button(
                        sc["name"], icon=sc.get("icon", "auto_awesome"),
                        on_click=lambda s=sc: on_scene(s),
                    ).props(f"no-caps size=lg color={sc.get('color', 'primary')}").classes("w-full h-24")

        # ---- FARI ----
        with ui.tab_panel(t_fari):
            ui.label(f"Ogni pressione aggiunge {FARI_STEP} min (max {FARI_MAX}).").classes("text-grey-5")
            with ui.grid(columns=1).classes("w-full gap-3 md:grid-cols-2"):
                for name, (db, byte) in FARI:
                    with ui.card().classes("w-full"):
                        ui.label(name).classes("text-lg font-semibold")
                        lbl = ui.label("--").classes("text-3xl font-bold")
                        faro_labels[name] = lbl
                        with ui.row().classes("w-full"):
                            ui.button(f"+{FARI_STEP} min", icon="add",
                                      on_click=lambda n=name, d=db, b=byte: on_faro(n, d, b, FARI_STEP)
                                      ).props("no-caps color=amber-8").classes("grow")
                            ui.button("Spegni", icon="power_settings_new",
                                      on_click=lambda n=name, d=db, b=byte: on_faro(n, d, b, 0)
                                      ).props("no-caps color=blue-grey-8")

        # ---- TELECAMERE ----
        with ui.tab_panel(t_cam):
            if not cameras:
                ui.label("Nessuna telecamera configurata")
            with ui.grid(columns=1).classes("w-full gap-3 md:grid-cols-2"):
                for i, cam in enumerate(cameras):
                    with ui.card().classes("w-full p-2"):
                        ui.label(cam.name).classes("font-semibold")
                        ui.interactive_image(f"/video/{i}").classes("w-full")

    async def refresh():
        await run.io_bound(plc.poll)
        plc_badge.text = "PLC online" if plc.connected else "PLC offline"
        plc_badge.props(f"color={'green' if plc.connected else 'red'}")
        for name, (btn, icon) in light_buttons.items():
            st = plc.lights.get(name)
            color = UNK_COLOR if st is None else (ON_COLOR if st else OFF_COLOR)
            btn.props(f"color={color}")
            btn.props(f"icon={icon}")
        for name, lbl in faro_labels.items():
            v = plc.fari.get(name)
            lbl.text = "--" if v is None else (f"{v} min" if v > 0 else "Spento")

    ui.timer(POLL_SECONDS, refresh)


ui.run(host="0.0.0.0", port=PORT, title="Casa", dark=True, reload=False, show=False)
