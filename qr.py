#!/usr/bin/env python3
"""
qr.py — SmartShelf Identity Node (Raspberry Pi)
================================================
Scans QR codes via Pi camera, looks up users in Firestore,
manages sessions on the cabinet document.

SSD1306 128x64 OLED (I2C) wiring:
  VCC  -> Pin  1  (3.3V)
  GND  -> Pin  6  (GND)
  SDA  -> Pin  3  (GPIO2, I2C1 SDA)
  SCL  -> Pin  5  (GPIO3, I2C1 SCL)
  Enable I2C in raspi-config -> Interface Options -> I2C
  Install: pip install luma.oled Pillow

Flow:
  1. Scan QR code -> look up user in Firestore
  2. Check cooldowns (4h for recipients) and bans (1-week for violations)
  3. Write activeSession to cabinet/cabinet_001
  4. For recipients: 10-second OLED countdown; motion detection skips the
     countdown and waits for stillness to capture a clear photo
  5. On second scan by same user OR timeout -> close session
  6. First violation = warning only; second violation = 1-week ban
"""

from picamera2 import Picamera2
import cv2
import firebase_admin
from firebase_admin import credentials, firestore
import os
import time
import threading
from datetime import datetime, timezone, timedelta

# ---------- SSD1306 OLED ----------
try:
    from luma.core.interface.serial import i2c
    from luma.oled.device import ssd1306
    from PIL import Image, ImageDraw, ImageFont
    _serial = i2c(port=1, address=0x3C)
    oled = ssd1306(_serial, width=128, height=64)
    # Try to load a bitmap font; fall back to PIL default
    try:
        _font_lg = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14
        )
        _font_md = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11
        )
        _font_sm = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 9
        )
    except Exception:
        _font_lg = ImageFont.load_default()
        _font_md = _font_lg
        _font_sm = _font_lg
    HAS_OLED = True
except Exception as _oled_err:
    HAS_OLED = False
    oled = None
    print(f"[OLED] Not available: {_oled_err}", flush=True)

# Shared display state — updated from main thread, read by display thread
display_state = {
    "screen": "idle",       # idle | scanning | session | denied | unknown | goodbye
    "name": "",
    "role": "",
    "score": 0,
    "countdown": 0,         # seconds left (recipient only)
    "message": "",          # short sub-message (deny reason etc.)
    "capacity": "",         # e.g. "14/20"
}
_display_lock = threading.Lock()

# ---------- Config ----------
SERVICE_ACCOUNT_KEY_PATH = os.path.expanduser('~/firestone/auth.json')
CABINET_ID = os.getenv("CABINET_ID", "cabinet_001")
SESSION_TIMEOUT = 120          # seconds before auto-close
RECIPIENT_COOLDOWN_HOURS = 4   # recipients can visit every 4 hours
BAN_DURATION_DAYS = 7          # 1-week ban for violations (2nd+ offense)
COOLDOWN = 2                   # QR scan debounce seconds
RECIPIENT_COUNTDOWN = 10       # OLED countdown seconds before auto-capture
RESULT_POLL_TIMEOUT = 30       # seconds to wait for Gemini analysis result

# ---------- Firebase ----------
cred = credentials.Certificate(SERVICE_ACCOUNT_KEY_PATH)
firebase_admin.initialize_app(cred)
db = firestore.client()

# ---------- Camera ----------
def _init_camera(retries=10, delay=3):
    """Try to acquire the camera, retrying if another process holds it."""
    for attempt in range(1, retries + 1):
        try:
            cam = Picamera2()
            cfg = cam.create_preview_configuration(
                main={"size": (480, 360)},
                buffer_count=1
            )
            cam.configure(cfg)
            cam.start()
            return cam
        except RuntimeError as exc:
            print(f"[Camera] Attempt {attempt}/{retries} failed: {exc}", flush=True)
            if attempt < retries:
                print(f"[Camera] Retrying in {delay}s...", flush=True)
                time.sleep(delay)
    raise RuntimeError("Could not acquire camera after multiple attempts. "
                       "Kill any other process using it and try again.")

picam2 = _init_camera()

# ---------- QR Detector ----------
detector = cv2.QRCodeDetector()

# ---------- Session state ----------
active_session = None          # dict with uid, role, displayName, etc.
session_timer = None           # threading.Timer for auto-close
session_score_deltas = []      # accumulated score changes
_session_lock = threading.RLock()  # prevents timer thread / main thread races


def log(msg=""):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ==================================================================
# OLED DISPLAY
# ==================================================================

def _set_display(**kwargs):
    """Thread-safe update of the shared display state."""
    with _display_lock:
        display_state.update(kwargs)


def _render_frame():
    """Render one frame to the OLED based on current display_state."""
    with _display_lock:
        state = dict(display_state)  # shallow copy to avoid holding lock during draw

    img = Image.new("1", (128, 64), 0)  # 0 = black background
    draw = ImageDraw.Draw(img)
    screen = state["screen"]

    if screen == "idle":
        # Header bar
        draw.rectangle((0, 0, 127, 15), fill=1)
        draw.text((4, 1), "SmartShelf", font=_font_md, fill=0)
        # Main prompt
        draw.text((14, 20), "Scan QR to begin", font=_font_md, fill=1)
        # Capacity
        if state["capacity"]:
            draw.text((4, 53), f"Cabinet: {state['capacity']}", font=_font_sm, fill=1)
        # Online dot
        draw.ellipse((118, 54, 126, 62), fill=1)

    elif screen == "scanning":
        draw.rectangle((0, 0, 127, 15), fill=1)
        draw.text((4, 1), "SmartShelf", font=_font_md, fill=0)
        draw.text((30, 28), "Looking up...", font=_font_md, fill=1)

    elif screen == "session":
        role = state["role"]
        name = state["name"][:16]  # truncate to fit
        score = state["score"]
        countdown = state["countdown"]

        # Header bar — always shows name
        draw.rectangle((0, 0, 127, 14), fill=1)
        draw.text((4, 2), name, font=_font_sm, fill=0)

        if role == "recipient" and countdown > 0:
            # Big clear countdown layout
            draw.text((4, 18), "Grab an item &", font=_font_sm, fill=1)
            draw.text((4, 29), "hold to camera", font=_font_sm, fill=1)
            # Countdown box — centered horizontally
            cstr = str(countdown)
            # box: x 44-83, y 40-62
            draw.rectangle((44, 40, 83, 62), outline=1)
            # Center the digit(s) inside the box
            cx = 52 if len(cstr) == 1 else 47
            draw.text((cx, 42), cstr, font=_font_lg, fill=1)
            # Small label above box
            draw.text((92, 47), "sec", font=_font_sm, fill=1)

        elif role == "recipient" and countdown == 0:
            draw.text((14, 22), "Analyzing item...", font=_font_sm, fill=1)
            draw.text((30, 38), "Please wait", font=_font_sm, fill=1)
            # Animated dots — use seconds modulo to cycle . / .. / ...
            dots = "." * ((int(time.time()) % 3) + 1)
            draw.text((98, 38), dots, font=_font_sm, fill=1)

        else:  # donor
            draw.text((4, 19), f"Pts: {score}", font=_font_sm, fill=1)
            draw.text((4, 32), "Add via app", font=_font_sm, fill=1)
            draw.text((4, 46), "Scan again to exit", font=_font_sm, fill=1)

    elif screen == "denied":
        draw.rectangle((0, 0, 127, 15), fill=1)
        draw.text((14, 1), "Access Denied", font=_font_md, fill=0)
        msg = state["message"]
        # Word-wrap into two lines of ~22 chars
        if len(msg) > 22:
            split = msg.rfind(" ", 0, 22)
            split = split if split > 0 else 22
            draw.text((4, 20), msg[:split], font=_font_sm, fill=1)
            draw.text((4, 32), msg[split+1:split+45], font=_font_sm, fill=1)
        else:
            draw.text((4, 24), msg, font=_font_sm, fill=1)
        draw.text((4, 50), "See staff for help", font=_font_sm, fill=1)

    elif screen == "unknown":
        draw.rectangle((0, 0, 127, 15), fill=1)
        draw.text((14, 1), "Card Unknown", font=_font_md, fill=0)
        draw.text((16, 26), "Not registered", font=_font_md, fill=1)
        draw.text((4, 50), "See staff for help", font=_font_sm, fill=1)

    elif screen == "goodbye":
        draw.text((20, 10), "Thank you,", font=_font_md, fill=1)
        name = state["name"][:16]
        draw.text((4, 26), name + "!", font=_font_lg, fill=1)
        score = state["score"]
        draw.text((4, 50), f"Score: {score}", font=_font_sm, fill=1)

    return img


def display_loop():
    """Daemon thread: renders the OLED at ~10 fps."""
    if not HAS_OLED:
        return
    while True:
        try:
            img = _render_frame()
            oled.display(img)
        except Exception as exc:
            log(f"  OLED render error: {exc}")
        time.sleep(0.1)  # 10 fps


def oled_clear():
    """Blank the display (call on shutdown)."""
    if HAS_OLED:
        try:
            oled.clear()
        except Exception:
            pass


# ==================================================================
# COOLDOWN & BAN CHECKING
# ==================================================================

def check_cooldown_and_ban(uid, role):
    """Check if a recipient is on cooldown or banned.
    Returns (allowed: bool, reason: str).
    Donors always pass.
    """
    if role == "donor":
        return True, ""

    now = datetime.now(timezone.utc)
    today_str = now.strftime("%Y-%m-%d")

    try:
        cd_ref = db.collection("userCooldowns").document(uid)
        cd_snap = cd_ref.get()

        if not cd_snap.exists:
            # First visit ever
            cd_ref.set({
                "uid": uid,
                "lastVisit": firestore.SERVER_TIMESTAMP,
                "bannedUntil": None,
                "violationCount": 0,
                "date": today_str,
            })
            return True, ""

        cd = cd_snap.to_dict()

        # Check ban
        banned_until = cd.get("bannedUntil")
        if banned_until:
            if hasattr(banned_until, 'timestamp'):
                ban_time = datetime.fromtimestamp(
                    banned_until.timestamp(), tz=timezone.utc
                )
            elif isinstance(banned_until, datetime):
                ban_time = (
                    banned_until if banned_until.tzinfo
                    else banned_until.replace(tzinfo=timezone.utc)
                )
            else:
                ban_time = None

            if ban_time and now < ban_time:
                remaining = ban_time - now
                days_left = remaining.days
                hours_left = remaining.seconds // 3600
                return False, f"Banned for {days_left}d {hours_left}h (violation penalty)"

        # Check 4-hour cooldown
        last_visit = cd.get("lastVisit")
        if last_visit:
            if hasattr(last_visit, 'timestamp'):
                lv_time = datetime.fromtimestamp(
                    last_visit.timestamp(), tz=timezone.utc
                )
            elif isinstance(last_visit, datetime):
                lv_time = (
                    last_visit if last_visit.tzinfo
                    else last_visit.replace(tzinfo=timezone.utc)
                )
            else:
                lv_time = None

            if lv_time:
                elapsed = now - lv_time
                cooldown_delta = timedelta(hours=RECIPIENT_COOLDOWN_HOURS)
                if elapsed < cooldown_delta:
                    remaining = cooldown_delta - elapsed
                    hours_left = remaining.seconds // 3600
                    mins_left = (remaining.seconds % 3600) // 60
                    return (
                        False,
                        f"Cooldown: {hours_left}h {mins_left}m remaining "
                        f"(every {RECIPIENT_COOLDOWN_HOURS}h)",
                    )

    except Exception as exc:
        log(f"  WARNING: Cooldown check failed: {exc}")
        return True, ""

    return True, ""


def apply_ban(uid, reason="photo_violation"):
    """First offense: issue a warning. Second+ offense: ban for BAN_DURATION_DAYS."""
    now = datetime.now(timezone.utc)
    try:
        cd_ref = db.collection("userCooldowns").document(uid)
        cd_snap = cd_ref.get()
        prior_violations = 0
        if cd_snap.exists:
            prior_violations = cd_snap.to_dict().get("violationCount", 0)

        if prior_violations == 0:
            # First offense — warning only, no ban
            cd_ref.set({
                "violationCount": 1,
                "lastViolation": firestore.SERVER_TIMESTAMP,
                "lastViolationReason": reason,
                "warned": True,
            }, merge=True)
            log(f"  WARNING ISSUED (1st offense): {uid} — {reason}")
        else:
            # Second+ offense — apply ban
            ban_until = now + timedelta(days=BAN_DURATION_DAYS)
            cd_ref.set({
                "bannedUntil": ban_until,
                "violationCount": firestore.Increment(1),
                "lastViolation": firestore.SERVER_TIMESTAMP,
                "lastViolationReason": reason,
            }, merge=True)
            log(f"  BAN APPLIED ({prior_violations+1} offenses): {uid} banned until {ban_until.isoformat()}")
    except Exception as exc:
        log(f"  WARNING: Could not apply violation: {exc}")


def update_last_visit(uid):
    """Update the lastVisit timestamp for cooldown tracking."""
    try:
        db.collection("userCooldowns").document(uid).set({
            "lastVisit": firestore.SERVER_TIMESTAMP,
        }, merge=True)
    except Exception as exc:
        log(f"  WARNING: Could not update lastVisit: {exc}")


# ==================================================================
# SESSION MANAGEMENT
# ==================================================================

def open_session(uid, role, display_name, score):
    """Write activeSession to the cabinet document and start timeout.
    Thread-safe: acquires _session_lock so it can't race with close_session.
    """
    global active_session, session_timer, session_score_deltas

    session_data = {
        "uid": uid,
        "role": role,
        "displayName": display_name,
        "score": score,
        "startedAt": firestore.SERVER_TIMESTAMP,
    }

    try:
        db.collection("cabinet").document(CABINET_ID).set({
            "activeSession": session_data,
        }, merge=True)

        # Write session_start event
        db.collection("events").add({
            "type": "session_start",
            "userId": uid,
            "userRole": role,
            "timestamp": firestore.SERVER_TIMESTAMP,
            "deviceId": CABINET_ID,
            "itemId": None,
            "scoreDelta": 0,
        })

        # Update lastSeen on user doc
        db.collection("users").document(uid).set({
            "lastSeen": firestore.SERVER_TIMESTAMP,
        }, merge=True)

    except Exception as exc:
        log(f"  WARNING: Could not write session: {exc}")
        return False

    with _session_lock:
        active_session = {
            "uid": uid,
            "role": role,
            "displayName": display_name,
            "score": score,
            "startedAt": time.time(),
        }
        session_score_deltas = []

        # Cancel any stale timer, start fresh one
        if session_timer:
            session_timer.cancel()
        session_timer = threading.Timer(
            SESSION_TIMEOUT, lambda: close_session("timeout")
        )
        session_timer.daemon = True
        session_timer.start()

    # Update OLED
    _set_display(screen="session", name=display_name, role=role,
                 score=score, countdown=0)

    log(f"  SESSION OPENED: {display_name} ({role})")
    if role == "recipient":
        log(f"  >> RECEIVING MODE: 10s countdown + motion capture")
        cap_thread = threading.Thread(
            target=recipient_capture_loop,
            args=(uid,),
            daemon=True,
        )
        cap_thread.start()

    return True


def close_session(reason="manual"):
    """Close the active session, commit score, clear cabinet.
    Thread-safe: uses _session_lock so the timeout timer thread
    and the main thread cannot run this simultaneously.
    """
    global active_session, session_timer, session_score_deltas

    with _session_lock:
        if not active_session:
            return

        uid          = active_session["uid"]
        display_name = active_session["displayName"]
        role         = active_session["role"]
        goodbye_score = active_session.get("score", 0) + sum(session_score_deltas)

        # Clear local state first so any re-entrant call bails immediately
        active_session      = None
        total_delta         = sum(session_score_deltas)
        session_score_deltas.clear()

        # Cancel timeout timer (no-op if already fired)
        if session_timer:
            session_timer.cancel()
            session_timer = None

    # Firestore work outside the lock (blocking I/O, but state already cleared)
    try:
        # Commit total score delta
        if total_delta != 0:
            try:
                user_ref  = db.collection("users").document(uid)
                user_snap = user_ref.get()
                if user_snap.exists:
                    current_score = user_snap.to_dict().get("score", 0)
                    new_score = max(0, current_score + total_delta)
                    user_ref.update({"score": new_score})
                    log(f"  Score: {current_score} -> {new_score} (delta: {total_delta:+d})")
            except Exception as exc:
                log(f"  WARNING: Score update failed: {exc}")

        # Write session_end event
        try:
            db.collection("events").add({
                "type": "session_end",
                "userId": uid,
                "userRole": role,
                "timestamp": firestore.SERVER_TIMESTAMP,
                "deviceId": CABINET_ID,
                "scoreDelta": total_delta,
                "closeReason": reason,
            })
        except Exception as exc:
            log(f"  WARNING: Could not write session_end event: {exc}")

        # Only delete activeSession from Firestore if no new session was opened
        # in the brief window between clearing active_session and this write.
        # If a new session WAS opened, its doc already overwrote ours — skip delete.
        with _session_lock:
            should_delete = (active_session is None)
        if should_delete:
            try:
                db.collection("cabinet").document(CABINET_ID).update({
                    "activeSession": firestore.DELETE_FIELD,
                })
            except Exception as exc:
                log(f"  WARNING: Could not clear activeSession: {exc}")

        # Update cooldown for recipients
        if role == "recipient":
            update_last_visit(uid)

    except Exception as exc:
        log(f"  ERROR in close_session: {exc}")

    log(f"  SESSION CLOSED: {display_name} (reason: {reason}, delta: {total_delta:+d})")

    # Show goodbye screen briefly, then return to idle —
    # but ONLY if no new session was opened while we were working.
    _set_display(screen="goodbye", name=display_name, score=max(0, goodbye_score))

    def _back_to_idle():
        time.sleep(3)
        with _session_lock:
            if active_session is None:   # don't overwrite a new session's screen
                _set_display(screen="idle", countdown=0)
    threading.Thread(target=_back_to_idle, daemon=True).start()


# ==================================================================
# RECIPIENT MOTION-DETECTION CAPTURE
# ==================================================================

def _do_capture(uid, frame):
    """Save captured frame locally, write a scanRequest, and return the doc ref."""
    try:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.expanduser("~/captures")
        os.makedirs(save_dir, exist_ok=True)
        img_path = os.path.join(save_dir, f"{uid}_{ts}.jpg")
        cv2.imwrite(img_path, frame)
        _ts, doc_ref = db.collection("scanRequests").add({
            "userId": uid,
            "cabinetId": CABINET_ID,
            "action": "remove",
            "status": "pending",
            "timestamp": firestore.SERVER_TIMESTAMP,
            "imagePath": img_path,
        })
        log(f"  CAPTURED: {img_path}")
        return doc_ref
    except Exception as exc:
        log(f"  WARNING: Capture failed: {exc}")
        return None


def _poll_result_and_punish(uid, doc_ref):
    """Poll the scanRequest doc until Gemini finishes and log the outcome.

    Violation is ONLY applied when the AI result explicitly reports that
    there is no food item in the frame at all (confidence == 0 or
    productName is empty/none).  An inventory mismatch is NOT a violation —
    the user took something; the record just didn't match.
    """
    deadline = time.time() + RESULT_POLL_TIMEOUT
    while time.time() < deadline:
        time.sleep(2)
        try:
            snap = doc_ref.get()
            if not snap.exists:
                break
            data = snap.to_dict()
            status = data.get("status", "")
            if status not in ("pending", "processing", ""):
                result     = data.get("result") or {}
                product    = (result.get("productName") or "").strip().lower()
                confidence = float(result.get("confidence") or 1.0)
                truly_empty = (
                    product in ("", "none", "unknown", "nothing", "no item", "empty")
                    and confidence < 0.20
                )
                if truly_empty:
                    log(f"  AI found nothing in frame (conf={confidence:.0%}) — applying violation to {uid}")
                    apply_ban(uid, reason="empty_frame")
                else:
                    log(f"  Scan complete for {uid}: '{product}' (conf={confidence:.0%}) — no violation")
                return
        except Exception as exc:
            log(f"  WARNING: Result poll failed: {exc}")
    log(f"  Result poll timed out for {uid} — no action taken")


def recipient_capture_loop(uid):
    """Simple 10-second countdown for a recipient session.

    Shows a ticking countdown on the OLED so the user has time
    to grab an item and hold it in front of the camera.
    At zero the current frame is captured and submitted for
    Gemini analysis.  A violation is only applied if the AI
    determines there is nothing in the photo.
    """
    deadline = time.time() + RECIPIENT_COUNTDOWN
    last_frame = None

    _set_display(countdown=RECIPIENT_COUNTDOWN)

    while True:
        # Abort if session ended or changed user
        with _session_lock:
            sess = active_session
        if sess is None or sess.get("uid") != uid:
            return

        last_frame = picam2.capture_array()

        now = time.time()
        remaining = max(0, int(deadline - now))
        _set_display(countdown=remaining)

        if now >= deadline:
            break

        time.sleep(0.1)  # ~10 fps is plenty for a countdown

    log("  Countdown elapsed — capturing frame")
    _set_display(countdown=0)  # flip OLED to "Analyzing..."

    doc_ref = _do_capture(uid, last_frame)
    if doc_ref:
        threading.Thread(
            target=_poll_result_and_punish,
            args=(uid, doc_ref),
            daemon=True,
        ).start()

    _set_display(screen="session", countdown=0)


# ==================================================================
# HEARTBEAT
# ==================================================================

def heartbeat_loop():
    """Update isOnline every 30 seconds and refresh capacity on OLED."""
    while True:
        try:
            db.collection("cabinet").document(CABINET_ID).set({
                "isOnline": True,
                "lastUpdated": firestore.SERVER_TIMESTAMP,
            }, merge=True)
            # Refresh capacity on idle screen
            cab = db.collection("cabinet").document(CABINET_ID).get()
            if cab.exists:
                d = cab.to_dict()
                cur = d.get("currentItemCount", 0)
                total = d.get("totalSlots", 20)
                _set_display(capacity=f"{cur}/{total}")
        except Exception as exc:
            log(f"  Heartbeat failed: {exc}")
        time.sleep(30)


# ==================================================================
# MAIN LOOP
# ==================================================================

def main():
    log("=" * 50)
    log("  SmartShelf Identity Node (Raspberry Pi)")
    log(f"  Cabinet: {CABINET_ID}")
    log(f"  Recipient cooldown: {RECIPIENT_COOLDOWN_HOURS}h")
    log(f"  Ban duration: {BAN_DURATION_DAYS} days")
    log("=" * 50)
    log("")
    log("Scanning for QR codes...")

    # Start heartbeat thread
    hb = threading.Thread(target=heartbeat_loop, daemon=True)
    hb.start()

    # Start OLED display thread
    if HAS_OLED:
        disp = threading.Thread(target=display_loop, daemon=True)
        disp.start()
        log("  OLED display started (128x64 SSD1306)")
    else:
        log("  OLED not available (running without display)")
    _set_display(screen="idle")

    last_qr = None
    last_scan_time = 0

    try:
        while True:
            frame = picam2.capture_array()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            data, bbox, _ = detector.detectAndDecode(gray)

            if data and bbox is not None:
                current_time = time.time()

                if data != last_qr or current_time - last_scan_time > COOLDOWN:
                    last_qr = data
                    last_scan_time = current_time

                    log(f"QR Detected: {data}")
                    _set_display(screen="scanning")

                    # Look up user
                    docs = (
                        db.collection("users")
                        .where("qrCode", "==", data)
                        .limit(1)
                        .stream()
                    )
                    user_doc = None
                    user_data = None
                    for d in docs:
                        user_doc = d
                        user_data = d.to_dict()

                    if not user_doc:
                        log("  No user found with this QR code")
                        _set_display(screen="unknown")
                        def _unknown_timeout():
                            time.sleep(3)
                            if not active_session:
                                _set_display(screen="idle")
                        threading.Thread(target=_unknown_timeout, daemon=True).start()
                        continue

                    uid = user_doc.id
                    display_name = user_data.get("displayName", "Unknown")
                    role = user_data.get("role", "recipient")
                    score = user_data.get("score", 0)

                    log(f"  User: {display_name} (UID: {uid})")
                    log(f"  Role: {role} | Score: {score}")

                    # Same user scans again -> sign out
                    if active_session and active_session["uid"] == uid:
                        log("  Same user scanned again -> closing session")
                        close_session("manual_signout")
                        continue

                    # Different user while session active
                    if active_session:
                        log("  Different user -> closing previous session")
                        close_session("new_user_override")

                    # Check cooldowns and bans (recipients only)
                    allowed, reason = check_cooldown_and_ban(uid, role)
                    if not allowed:
                        log(f"  ACCESS DENIED: {reason}")
                        _set_display(screen="denied", name=display_name, message=reason)
                        def _denied_timeout():
                            time.sleep(4)
                            if not active_session:
                                _set_display(screen="idle")
                        threading.Thread(target=_denied_timeout, daemon=True).start()
                        try:
                            db.collection("events").add({
                                "type": "access_denied",
                                "userId": uid,
                                "userRole": role,
                                "timestamp": firestore.SERVER_TIMESTAMP,
                                "deviceId": CABINET_ID,
                                "reason": reason,
                            })
                        except Exception:
                            pass
                        continue

                    # Open session
                    open_session(uid, role, display_name, score)

            if cv2.waitKey(1) == ord("q"):
                break

    except KeyboardInterrupt:
        log("Shutting down...")
    finally:
        if active_session:
            close_session("shutdown")
        try:
            db.collection("cabinet").document(CABINET_ID).set({
                "isOnline": False,
                "activeSession": firestore.DELETE_FIELD,
            }, merge=True)
        except Exception:
            pass
        oled_clear()
        picam2.stop()
        log("Goodbye!")


if __name__ == "__main__":
    main()
