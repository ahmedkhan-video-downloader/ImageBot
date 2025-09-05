# bot.py
"""
Image & Video Utility Bot - Ahmed Khan
Features:
- Image: enhance, remove bg, cartoon, ascii (file), watermark, pdf (multi), compress, bw, invert, rotate, sticker (static/webp)
- Video: compress, trim, to_gif, to_animated_sticker (webm)
- AI image gen (optional) via external API if API key provided (IMAGE_API_PROVIDER, IMAGE_API_KEY)
- Safe file handling, per-user session (in-memory), cleanup
- Sends files (not links) to avoid HTTP 414
"""
import os
import uuid
import shutil
import traceback
import tempfile
import subprocess
from functools import partial

import telebot
from telebot import types

import cv2
import numpy as np
from rembg import remove
from PIL import Image, ImageDraw, ImageFont, ImageOps

# Optional video libs
try:
    from moviepy.editor import VideoFileClip, vfx, concatenate_videoclips
except Exception:
    VideoFileClip = None

# Environment
TOKEN = os.getenv("BOT_TOKEN")
if not TOKEN:
    raise SystemExit("ERROR: BOT_TOKEN environment variable not set. ضع توكن البوت في متغير البيئة BOT_TOKEN")

# Optional AI image generator config (example: OpenAI / Stability / custom)
IMAGE_API_PROVIDER = os.getenv("IMAGE_API_PROVIDER", "").lower()  # e.g., "openai" or "stability"
IMAGE_API_KEY = os.getenv("IMAGE_API_KEY", "")

bot = telebot.TeleBot(TOKEN)

USER_TAG = "@AHMED_KHANA"
DEV_NOTE = " المطور"

# simple in-memory per-user state: tracks list of uploaded files and pending step params
# user_states[user_id] = {"images": [paths], "videos":[paths], "pending": {"action":..., "data":...}}
user_states = {}

# ---- Helpers ----
def tmpname(prefix="tmp", ext=""):
    return f"{prefix}_{uuid.uuid4().hex}{('.' + ext) if ext else ''}"

def safe_remove(path):
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except Exception:
        pass

def cleanup_prefix(prefixes=("tmp_", "out_")):
    for fn in os.listdir("."):
        if any(fn.startswith(p) for p in prefixes):
            safe_remove(fn)

def send_photo(chat_id, path, caption=None):
    with open(path, "rb") as f:
        bot.send_photo(chat_id, f, caption=caption)

def send_doc(chat_id, path, caption=None):
    with open(path, "rb") as f:
        bot.send_document(chat_id, f, caption=caption)

# ---- Image functions ----
def enhance_image(image_path, out=None):
    if out is None:
        out = tmpname("out_enhanced", "jpg")
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("الصورة غير صالحة")
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    enhanced = cv2.filter2D(img, -1, kernel)
    cv2.imwrite(out, enhanced)
    return out

def remove_bg_image(image_path, out=None):
    if out is None:
        out = tmpname("out_nobg", "png")
    with open(image_path, "rb") as f:
        data = f.read()
    result = remove(data)
    with open(out, "wb") as f:
        f.write(result)
    return out

def cartoonify_image(image_path, out=None):
    if out is None:
        out = tmpname("out_cartoon", "jpg")
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("الصورة غير صالحة")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(img, 9, 300, 300)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    cv2.imwrite(out, cartoon)
    return out

def image_to_ascii_file(image_path, out=None, width=100):
    if out is None:
        out = tmpname("out_ascii", "txt")
    img = Image.open(image_path).convert("L")
    aspect_ratio = img.height / img.width
    height = max(1, int(aspect_ratio * width * 0.55))
    img = img.resize((width, height))
    pixels = np.array(img)
    chars = "@%#*+=-:. "
    max_index = len(chars) - 1
    lines = []
    for row in pixels:
        line = "".join(chars[min((int(pixel) * len(chars)) // 256, max_index)] for pixel in row)
        lines.append(line)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return out

def add_watermark(image_path, out=None, text=None):
    if out is None:
        out = tmpname("out_watermark", "jpg")
    if text is None:
        text = f"{USER_TAG} - {DEV_NOTE}"
    img = Image.open(image_path).convert("RGBA")
    iw, ih = img.size
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 28)
    except:
        font = ImageFont.load_default()
    draw = ImageDraw.Draw(img)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]; th = bbox[3] - bbox[1]
    pos = (iw - tw - 12, ih - th - 12)
    overlay = Image.new('RGBA', img.size, (0,0,0,0))
    od = ImageDraw.Draw(overlay)
    od.rectangle([pos[0]-6, pos[1]-6, pos[0]+tw+6, pos[1]+th+6], fill=(0,0,0,140))
    combined = Image.alpha_composite(img, overlay)
    draw2 = ImageDraw.Draw(combined)
    draw2.text(pos, text, fill=(255,255,255,255), font=font)
    combined.convert("RGB").save(out)
    return out

def image_to_pdf(images_list, out=None):
    if out is None:
        out = tmpname("out_pdf", "pdf")
    pil_images = [Image.open(p).convert("RGB") for p in images_list]
    if not pil_images:
        raise ValueError("لا توجد صور")
    first, rest = pil_images[0], pil_images[1:]
    first.save(out, save_all=True, append_images=rest)
    return out

def compress_image(image_path, out=None, quality=75):
    if out is None:
        out = tmpname("out_compressed", "jpg")
    img = Image.open(image_path).convert("RGB")
    img.save(out, "JPEG", quality=quality)
    return out

def invert_colors(image_path, out=None):
    if out is None:
        out = tmpname("out_invert", "jpg")
    img = Image.open(image_path).convert("RGB")
    inv = ImageOps.invert(img)
    inv.save(out)
    return out

def bw_image(image_path, out=None):
    if out is None:
        out = tmpname("out_bw", "jpg")
    img = Image.open(image_path).convert("L")
    img.save(out)
    return out

def rotate_image(image_path, angle, out=None):
    if out is None:
        out = tmpname("out_rotate", "jpg")
    img = Image.open(image_path).convert("RGB")
    rotated = img.rotate(angle, expand=True)
    rotated.save(out)
    return out

def image_to_sticker(image_path, out=None):
    """Create static webp sticker 512x512"""
    if out is None:
        out = tmpname("out_sticker", "webp")
    img = Image.open(image_path).convert("RGBA")
    size = 512
    img.thumbnail((size, size), Image.LANCZOS)
    bg = Image.new("RGBA", (size, size), (0,0,0,0))
    w,h = img.size
    pos = ((size-w)//2, (size-h)//2)
    bg.paste(img, pos, img)
    bg.save(out, "WEBP")
    return out

# ---- Video functions (moviepy required) ----
def compress_video(video_path, out=None, target_bitrate="800k"):
    if VideoFileClip is None:
        raise RuntimeError("moviepy غير مثبت؛ لتشغيل وظائف الفيديو ثبّت moviepy")
    if out is None:
        out = tmpname("out_video", "mp4")
    clip = VideoFileClip(video_path)
    clip.write_videofile(out, bitrate=target_bitrate, audio=True, threads=0, logger=None)
    clip.close()
    return out

def video_to_gif(video_path, out=None, fps=15):
    if VideoFileClip is None:
        raise RuntimeError("moviepy غير مثبت")
    if out is None:
        out = tmpname("out_gif", "gif")
    clip = VideoFileClip(video_path).subclip(0, min(10, VideoFileClip(video_path).duration))
    clip.write_gif(out, fps=fps)
    clip.close()
    return out

def video_to_animated_sticker(video_path, out=None):
    """
    Create a short webm animation (vp8) compatible to send as document (Telegram
    doesn't accept webm via send_sticker in bots reliably). We'll send as document.
    """
    if VideoFileClip is None:
        raise RuntimeError("moviepy غير مثبت")
    if out is None:
        out = tmpname("out_anim", "webm")
    clip = VideoFileClip(video_path)
    # limit length to 5 seconds
    duration = min(5, clip.duration)
    sub = clip.subclip(0, duration)
    # resize to max 512
    sub = sub.resize(width=512)
    sub.write_videofile(out, codec="libvpx", audio=False, logger=None, threads=0)
    clip.close()
    return out

# ---- Save incoming files ----
def save_photo_from_message(msg):
    file_info = bot.get_file(msg.photo[-1].file_id)
    data = bot.download_file(file_info.file_path)
    fname = tmpname("tmp_img", "jpg")
    with open(fname, "wb") as f:
        f.write(data)
    return fname

def save_video_from_message(msg):
    file_info = bot.get_file(msg.video.file_id)
    data = bot.download_file(file_info.file_path)
    fname = tmpname("tmp_vid", "mp4")
    with open(fname, "wb") as f:
        f.write(data)
    return fname

# ---- Keyboards & UI ----
def keyboard():
    markup = types.ReplyKeyboardMarkup(row_width=2, resize_keyboard=True)
    buttons = [
        types.KeyboardButton("تحسين الصورة"),
        types.KeyboardButton("إزالة الخلفية"),
        types.KeyboardButton("تحويل إلى كرتونية"),
        types.KeyboardButton("تحويل إلى ASCII"),
        types.KeyboardButton("إضافة علامة مائية"),
        types.KeyboardButton("تحويل إلى PDF"),
        types.KeyboardButton("ضغط الصورة"),
        types.KeyboardButton("أبيض وأسود"),
        types.KeyboardButton("عكس الألوان"),
        types.KeyboardButton("تدوير الصورة"),
        types.KeyboardButton("تحويل إلى ملصق"),
        types.KeyboardButton("ضغط الفيديو"),
        types.KeyboardButton("تحويل فيديو إلى GIF"),
        types.KeyboardButton("تحويل فيديو إلى ملصق متحرك (webm)"),
        types.KeyboardButton("توليد صورة بالذكاء الاصطناعي")
    ]
    markup.add(*buttons)
    return markup

@bot.message_handler(commands=['start','help'])
def cmd_start(m):
    user_states.pop(m.from_user.id, None)
    bot.reply_to(m, f"👋 مرحباً! أنا بوت أحمد خان. كيف أقدر أخدمك اليوم؟\nأرسل صورة أو فيديو ثم اختر العملية.", reply_markup=keyboard())

# handle photo(s)
@bot.message_handler(content_types=['photo'])
def on_photo(m):
    try:
        fname = save_photo_from_message(m)
        uid = m.from_user.id
        st = user_states.setdefault(uid, {"images": [], "videos": [], "pending": None})
        st["images"].append(fname)
        bot.reply_to(m, f"✅ تم حفظ الصورة (#{len(st['images'])}). يمكنك إرسال المزيد أو اختيار عملية.", reply_markup=keyboard())
    except Exception as e:
        bot.reply_to(m, f"❌ خطأ أثناء حفظ الصورة: {e}")

# handle video
@bot.message_handler(content_types=['video'])
def on_video(m):
    try:
        fname = save_video_from_message(m)
        uid = m.from_user.id
        st = user_states.setdefault(uid, {"images": [], "videos": [], "pending": None})
        st["videos"].append(fname)
        bot.reply_to(m, f"✅ تم حفظ الفيديو (#{len(st['videos'])}). اختر العملية.", reply_markup=keyboard())
    except Exception as e:
        bot.reply_to(m, f"❌ خطأ أثناء حفظ الفيديو: {e}")

# main action handler
@bot.message_handler(func=lambda m: True)
def handle_action(m):
    uid = m.from_user.id
    st = user_states.get(uid)
    if not st or (not st["images"] and not st["videos"]):
        bot.reply_to(m, "⚠️ أرسل صورة أو فيديو أولاً ثم اختر العملية.", reply_markup=keyboard())
        return

    action = m.text.strip()
    try:
        # Image actions (use last uploaded image by default)
        if action == "تحسين الصورة":
            inp = st["images"][-1]
            out = enhance_image(inp)
            send_photo(m.chat.id, out, caption=f"✅ تم التحسين\nالمطور: {USER_TAG}")

        elif action == "إزالة الخلفية":
            inp = st["images"][-1]
            out = remove_bg_image(inp)
            send_photo(m.chat.id, out, caption=f"🖼️ تمت إزالة الخلفية\nالمطور: {USER_TAG}")

        elif action == "تحويل إلى كرتونية":
            inp = st["images"][-1]
            out = cartoonify_image(inp)
            send_photo(m.chat.id, out, caption=f"🎨 تحويل إلى كرتونية\nالمطور: {USER_TAG}")

        elif action == "تحويل إلى ASCII":
            inp = st["images"][-1]
            out = image_to_ascii_file(inp, width=120)
            send_doc(m.chat.id, out, caption=f"📜 تحويل إلى ASCII\nالمطور: {USER_TAG}")

        elif action == "إضافة علامة مائية":
            inp = st["images"][-1]
            out = add_watermark(inp)
            send_photo(m.chat.id, out, caption=f"💧 تم إضافة العلامة المائية\nالمطور: {USER_TAG}")

        elif action == "تحويل إلى PDF":
            # use all saved images
            images = st["images"]
            out = image_to_pdf(images)
            send_doc(m.chat.id, out, caption=f"📄 تم تحويل الصور إلى PDF\nالمطور: {USER_TAG}")

        elif action == "ضغط الصورة":
            inp = st["images"][-1]
            out = compress_image(inp, quality=70)
            send_photo(m.chat.id, out, caption=f"📉 تم ضغط الصورة\nالمطور: {USER_TAG}")

        elif action == "أبيض وأسود":
            inp = st["images"][-1]
            out = bw_image(inp)
            send_photo(m.chat.id, out, caption=f"⚪⬛ أبيض وأسود\nالمطور: {USER_TAG}")

        elif action == "عكس الألوان":
            inp = st["images"][-1]
            out = invert_colors(inp)
            send_photo(m.chat.id, out, caption=f"🌀 عكس الألوان\nالمطور: {USER_TAG}")

        elif action == "تدوير الصورة":
            inp = st["images"][-1]
            # default rotate 90; for advanced: implement follow-up prompt to choose angle
            out = rotate_image(inp, 90)
            send_photo(m.chat.id, out, caption=f"🔁 تدوير 90°\nالمطور: {USER_TAG}")

        elif action == "تحويل إلى ملصق":
            inp = st["images"][-1]
            out = image_to_sticker(inp)
            # bots can send webp as sticker file; if fails, send as document
            try:
                with open(out, "rb") as s:
                    bot.send_sticker(m.chat.id, s)
                bot.send_message(m.chat.id, f"✅ تم إنشاء الملصق\nالمطور: {USER_TAG}")
            except Exception:
                send_doc(m.chat.id, out, caption=f"✅ تم إنشاء الملصق (مرسل كملف)\nالمطور: {USER_TAG}")

        # Video actions
        elif action == "ضغط الفيديو":
            if not st["videos"]:
                bot.reply_to(m, "⚠️ لم ترسل فيديو بعد.")
                return
            inp = st["videos"][-1]
            out = compress_video(inp)
            send_doc(m.chat.id, out, caption=f"📉 تم ضغط الفيديو\nالمطور: {USER_TAG}")

        elif action == "تحويل فيديو إلى GIF":
            if not st["videos"]:
                bot.reply_to(m, "⚠️ لم ترسل فيديو بعد.")
                return
            inp = st["videos"][-1]
            out = video_to_gif(inp)
            send_doc(m.chat.id, out, caption=f"🎞️ تم تحويل الفيديو إلى GIF\nالمطور: {USER_TAG}")

        elif action == "تحويل فيديو إلى ملصق متحرك (webm)":
            if not st["videos"]:
                bot.reply_to(m, "⚠️ لم ترسل فيديو بعد.")
                return
            inp = st["videos"][-1]
            out = video_to_animated_sticker(inp)
            send_doc(m.chat.id, out, caption=f"✅ تم إنشاء الملصق المتحرك (webm)\nالمطور: {USER_TAG}")

        elif action == "توليد صورة بالذكاء الاصطناعي":
            # simple flow: if IMAGE_API_KEY provided, generate with provider
            if not IMAGE_API_KEY or not IMAGE_API_PROVIDER:
                bot.reply_to(m, "⚠️ لم يتم إعداد API للتوليد. ضع IMAGE_API_PROVIDER و IMAGE_API_KEY كمتغيرات بيئة.")
                return
            bot.reply_to(m, "🔄 جاري توليد الصورة... أرسل وصفًا للنص المطلوب.")
            st["pending"] = {"action": "ai_generate"}
            return

        else:
            bot.reply_to(m, "❓ أمر غير معروف — اختر من الأزرار أو اكتب /help.")
            return

        # after successful action: notify and cleanup outputs & user inputs
        bot.send_message(m.chat.id, f"✅ تمت العملية. المطور: {USER_TAG}")
    except Exception as e:
        tb = traceback.format_exc()
        bot.reply_to(m, f"❌ حدث خطأ أثناء المعالجة: {e}\n\n{tb}")
    finally:
        # cleanup user's input files and generated outputs matching prefixes
        imgs = st.get("images", [])
        vids = st.get("videos", [])
        for p in imgs + vids:
            safe_remove(p)
        cleanup_prefix(("tmp_img","tmp_vid","out_"))
        user_states.pop(uid, None)

# Handle follow-up messages for AI generation or parameters
@bot.message_handler(func=lambda m: True)
def followups(m):
    uid = m.from_user.id
    st = user_states.get(uid)
    if not st or not st.get("pending"):
        return  # ignore, handled elsewhere
    pending = st["pending"]
    try:
        if pending["action"] == "ai_generate":
            prompt = m.text.strip()
            bot.reply_to(m, "🔄 جاري طلب التوليد... انتظر لحظة.")
            # Example: support OpenAI image generation (user must provide key)
            if IMAGE_API_PROVIDER == "openai":
                # minimal example using openai python lib if installed and API key set
                try:
                    import openai
                    openai.api_key = IMAGE_API_KEY
                    resp = openai.Image.create(prompt=prompt, size="1024x1024", n=1)
                    b64 = resp['data'][0]['b64_json']
                    import base64
                    imgdata = base64.b64decode(b64)
                    out = tmpname("out_ai", "png")
                    with open(out, "wb") as f:
                        f.write(imgdata)
                    send_photo(m.chat.id, out, caption=f"🖼️ تم توليد الصورة بواسطة AI\nالمطور: {USER_TAG}")
                    safe_remove(out)
                except Exception as e:
                    bot.reply_to(m, f"⚠️ فشل التوليد عبر OpenAI: {e}")
            else:
                bot.reply_to(m, "⚠️ موفر توليد الصور غير مدعوم حالياً في البوت. ادعم IMAGE_API_PROVIDER=openai مع IMAGE_API_KEY.")
            st["pending"] = None
    except Exception as e:
        bot.reply_to(m, f"❌ خطأ في العملية التالية: {e}")

if __name__ == "__main__":
    print("ImageBot starting...")
    bot.infinity_polling()
