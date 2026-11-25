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

import numpy as np
from fastapi.responses import StreamingResponse
from nicegui import app, background_tasks, run, ui
import os
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

import cv2


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
        ("Luce tv", "tv", dict(write=(5, 1, 0), read=(5, 1, 0))),
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
PULSE_SECONDS = 0.2

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
    dict(name="Spegni tutto", icon="power_settings_new", color="red-8", plc_bit=(110, 0, 0)),
    dict(name="Sera", icon="nights_stay", color="indigo-7",
         on=["Luce sala 1", "Luce corridoio"]),
    dict(name="Cena", icon="restaurant", color="orange-9",
         on=["Luce cucina", "Luce sala 2"]),
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
VIEW_TIMEOUT = 10   # la webcam si ferma se nessuno la guarda da questi secondi
# =====================================================================


# ------------------------------- PLC ---------------------------------
def _build_ranges():
    """Per ogni DB calcola l'intervallo di byte da leggere con UNA sola richiesta."""
    r = {}

    def add(db, start, size):
        lo, hi = r.get(db, (start, start + size))
        r[db] = (min(lo, start), max(hi, start + size))

    for _, (_, items) in LIGHT_GROUPS.items():
        for _, _, cfg in items:
            add(cfg["read"][0], cfg["read"][1], 1)
    for _, (db, byte) in FARI:
        add(db, byte, 2)
    return {db: (lo, hi - lo) for db, (lo, hi) in r.items()}


READ_RANGES = _build_ranges()


class PLC:
    poll_ms = 0

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
                t0 = time.time()
                blocks = {db: (start, self.client.db_read(db, start, size))
                          for db, (start, size) in READ_RANGES.items()}
                for _, (_, items) in LIGHT_GROUPS.items():
                    for name, _, cfg in items:
                        db, byte, bit = cfg["read"]
                        start, data = blocks[db]
                        self.lights[name] = get_bool(data, byte - start, bit)
                for name, (db, byte) in FARI:
                    start, data = blocks[db]
                    self.fari[name] = get_int(data, byte - start)
                self.poll_ms = int((time.time() - t0) * 1000)
            except Exception as e:
                log.warning("Errore lettura PLC: %s", e)
                self.connected = False
                self._mark_unknown()

    def _mark_unknown(self):
        for k in list(self.lights):
            self.lights[k] = None
        for k in list(self.fari):
            self.fari[k] = None

    def _write_bits(self, addrs, value):
        """Scrive più bit raggruppando per byte: una sola lettura+scrittura per byte."""
        groups = {}
        for db, byte, bit in addrs:
            groups.setdefault((db, byte), []).append(bit)
        for (db, byte), bits in groups.items():
            data = self.client.db_read(db, byte, 1)
            for b in bits:
                set_bool(data, 0, b, value)
            self.client.db_write(db, byte, data)

    def _pulse(self, addrs):
        """Impulso su uno o più bit. Il lock NON è tenuto durante l'attesa."""
        try:
            with self.lock:
                if not self.connected and not self._connect():
                    return False
                self._write_bits(addrs, True)
            time.sleep(PULSE_SECONDS)
            with self.lock:
                self._write_bits(addrs, False)
            return True
        except Exception as e:
            log.warning("Errore impulso PLC: %s", e)
            self.connected = False
            return False

    def press_light(self, name, cfg):
        if cfg.get("mode", DEFAULT_MODE) != "toggle":
            return self._pulse([cfg["write"]])
        with self.lock:
            if not self.connected and not self._connect():
                return False
            try:
                self._write_bit(cfg["write"], not self._read_bit(cfg["read"]))
                return True
            except Exception as e:
                log.warning("Errore scrittura PLC: %s", e)
                self.connected = False
                return False

    def _scene_plan(self, scene):
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

                return pulse_bits
            except Exception as e:
                log.warning("Errore scena: %s", e)
                self.connected = False
                return None

    def run_scene(self, scene):
        """Esegue una scena. Ritorna il numero di impulsi, o None se PLC offline."""
        pulses = self._scene_plan(scene)
        if pulses is None:
            return None
        if pulses and not self._pulse(pulses):
            return None
        return len(pulses)

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


IP_RASPBERRY = "192.168.1.32"
CAMERAS = [
    ("Telecamera 1", f"rtsp://{IP_RASPBERRY}:8554/cam1"),
    ("Telecamera 2", f"rtsp://{IP_RASPBERRY}:8554/cam2"),
]

class Camera(threading.Thread):
    def __init__(self, name, source):
        super().__init__(daemon=True)
        self.name, self.source = name, source
        self.jpeg = _placeholder("Non disponibile")
        self.online = False
        self.last_view = 0.0

    def run(self):
        cap = None
        while True:
            try:
                if time.time() - self.last_view > VIEW_TIMEOUT:
                    if cap is not None:
                        cap.release()
                        cap = None
                        self.jpeg = _placeholder("Avvio...")
                    self.online = False
                    time.sleep(0.3)
                    continue

                if cap is None or not cap.isOpened():
                    self.online = False
                    self.jpeg = _placeholder("Connessione...")
                    cap = cv2.VideoCapture(
                        self.source, cv2.CAP_FFMPEG,
                        [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000,
                         cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000],
                    )
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
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

                if frame.shape[1] != CAM_W or frame.shape[0] != CAM_H:
                    frame = cv2.resize(frame, (CAM_W, CAM_H))

                self.jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes()
                self.online = True
            except Exception as e:
                log.warning("Camera %s: %s", self.name, e)
                try:
                    cap and cap.release()
                except Exception:
                    pass
                cap = None
                time.sleep(5)


cameras = [Camera(n, s) for n, s in CAMERAS]
for c in cameras:
    c.start()
    time.sleep(0.5)

@app.get("/video/{idx}")
async def video_feed(idx: int):
    if idx < 0 or idx >= len(cameras):
        return StreamingResponse(iter([]), status_code=404)
    cam = cameras[idx]

    async def stream():
        while True:
            cam.last_view = time.time()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + cam.jpeg + b"\r\n"
            )
            await asyncio.sleep(1 / CAM_FPS)

    return StreamingResponse(
        stream(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# --------------------------------- UI --------------------------------
pending = {}      # stato "ottimistico" subito dopo un click: nome -> (valore, scadenza)
poll_now = None   # evento per forzare una lettura veloce dopo un'azione


async def poll_loop():
    """Un solo ciclo di lettura per tutta l'app, indipendente dai browser aperti."""
    while True:
        await run.io_bound(plc.poll)
        delay = POLL_SECONDS if plc.connected else 3.0
        try:
            await asyncio.wait_for(poll_now.wait(), delay)
            poll_now.clear()
            await asyncio.sleep(0.15)   # tempo al PLC per aggiornare i bit di stato
        except asyncio.TimeoutError:
            pass


async def _start():
    global poll_now
    poll_now = asyncio.Event()
    background_tasks.create(poll_loop())


app.on_startup(_start)

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
        cur = plc.lights.get(name)
        if cur is not None:   # aggiorna subito il colore, senza aspettare il PLC
            pending[name] = (not cur, time.time() + 2.5)
            apply_state()
        ok = await run.io_bound(plc.press_light, name, cfg)
        if not ok:
            pending.pop(name, None)
            ui.notify("PLC non raggiungibile", type="negative")
        poll_now.set()

    async def on_faro(name, db, byte, delta):
        ok = await run.io_bound(plc.add_faro_time, name, db, byte, delta)
        if not ok:
            ui.notify("PLC non raggiungibile", type="negative")
        poll_now.set()

    async def on_scene(scene):
        res = await run.io_bound(plc.run_scene, scene)
        if res is None:
            ui.notify("PLC non raggiungibile", type="negative")
        else:
            ui.notify(f"Scena «{scene['name']}» eseguita", type="positive")
        poll_now.set()

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
            cam_imgs = []
            with ui.grid(columns=1).classes("w-full gap-3 md:grid-cols-2"):
                for i, cam in enumerate(cameras):
                    with ui.card().classes("w-full p-2"):
                        ui.label(cam.name).classes("font-semibold")
                        img = ui.image("").classes("w-full")
                        cam_imgs.append((img, f"/video/{i}"))

            def on_tab_change(e):
                active = (e.value == "Camere")
                for img, url in cam_imgs:
                    img.set_source(url if active else "")

            tabs.on_value_change(on_tab_change)

    last = {}

    def apply_state():
        """Applica alla pagina i dati in cache: nessun accesso al PLC, quindi istantanea."""
        txt = f"PLC online · {plc.poll_ms} ms" if plc.connected else "PLC offline"
        if last.get("badge") != txt:
            last["badge"] = txt
            plc_badge.text = txt
            plc_badge.props(f"color={'green' if plc.connected else 'red'}")
        now = time.time()
        for name, (btn, _) in light_buttons.items():
            st = plc.lights.get(name)
            p = pending.get(name)
            if p:
                if st == p[0] or now > p[1]:
                    pending.pop(name, None)
                else:
                    st = p[0]
            if last.get(name, "?") != st:
                last[name] = st
                btn.props(f"color={UNK_COLOR if st is None else (ON_COLOR if st else OFF_COLOR)}")
        for name, lbl in faro_labels.items():
            v = plc.fari.get(name)
            t = "--" if v is None else (f"{v} min" if v > 0 else "Spento")
            if lbl.text != t:
                lbl.text = t

    ui.timer(0.3, apply_state)


ui.run(host="0.0.0.0", port=PORT, title="Casa", dark=True, reload=False, show=False)