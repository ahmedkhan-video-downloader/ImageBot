# bot.py
"""
ImageBot - Ahmed Khan
Features: images (enhance, remove bg, cartoon, ascii, watermark, pdf, compress, bw, invert, rotate, sticker),
video (compress, to_gif, to_animated_sticker), AI image gen (Free Stable Diffusion), safe file handling, per-user session.
"""

import os
import uuid
import traceback
import logging
import threading
import time

import telebot
from telebot import types

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

# إعداد logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional video support
try:
    from moviepy.editor import VideoFileClip
    MOVIEPY_OK = True
except Exception:
    VideoFileClip = None
    MOVIEPY_OK = False

# Optional Stable Diffusion support
try:
    from diffusers import StableDiffusionPipeline
    import torch
    SD_AVAILABLE = True
except Exception:
    SD_AVAILABLE = False

# Optional rembg support
try:
    from rembg import remove
    REMBG_AVAILABLE = True
except ImportError as e:
    logger.warning(f"rembg not available: {e}")
    REMBG_AVAILABLE = False
    def remove(data):
        raise Exception("ميزة إزالة الخلفية غير متاحة. جرب ميزات أخرى!")

# ----- Config via env -----
BOT_TOKEN = os.getenv("BOT_TOKEN")
if not BOT_TOKEN:
    raise SystemExit("ERROR: BOT_TOKEN environment variable not set. ضع توكن البوت في متغير البيئة BOT_TOKEN")

bot = telebot.TeleBot(BOT_TOKEN)

USER_TAG = "@AHMED_KHANA"
DEV_NOTE = "بصفتي المطور"

# per-user in-memory state
user_states = {}

# ---- Utilities ----
def tmpname(prefix="tmp", ext=""):
    return f"{prefix}_{uuid.uuid4().hex}{('.' + ext) if ext else ''}"

def safe_remove(path):
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except:
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

# ---- Stable Diffusion Image Generation ----
def generate_ai_image_free(prompt, out=None):
    if out is None:
        out = tmpname("out_ai", "png")
    
    try:
        if not SD_AVAILABLE:
            raise Exception("مكتبة Stable Diffusion غير مثبتة. run: pip install diffusers transformers accelerate torch torchvision")
        
        # استخدام نموذج خفيف وسريع
        model_id = "OFA-Sys/small-stable-diffusion-v0"
        
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            safety_checker=None  # إلغاء الفحص للسرعة
        )
        
        if torch.cuda.is_available():
            pipe = pipe.to("cuda")
            logger.info("Using GPU for image generation")
        else:
            logger.info("Using CPU for image generation (سيكون أبطأ)")
        
        # توليد الصورة بمعلمات سريعة
        image = pipe(
            prompt, 
            num_inference_steps=20,
            guidance_scale=7.5,
            width=512,
            height=512
        ).images[0]
        
        image.save(out)
        logger.info(f"Successfully generated image for prompt: {prompt}")
        return out
        
    except Exception as e:
        logger.error(f"Image generation failed: {str(e)}")
        raise Exception(f"فشل التوليد: {str(e)}")

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
    
    if not REMBG_AVAILABLE:
        raise Exception("⚠️ ميزة إزالة الخلفية غير متاحة. جرب ميزات أخرى!")
    
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
        font = ImageFont.truetype("arial.ttf", 28)
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

# ---- Video functions ----
def compress_video(video_path, out=None, target_bitrate="800k"):
    if not MOVIEPY_OK:
        raise RuntimeError("moviepy غير مثبت؛ لتشغيل فيديو ثبّت moviepy و ffmpeg")
    if out is None:
        out = tmpname("out_video", "mp4")
    clip = VideoFileClip(video_path)
    clip.write_videofile(out, bitrate=target_bitrate, audio=True, threads=4, logger=None, verbose=False)
    clip.close()
    return out

def video_to_gif(video_path, out=None, fps=15, duration=6):
    if not MOVIEPY_OK:
        raise RuntimeError("moviepy غير مثبت")
    if out is None:
        out = tmpname("out_gif", "gif")
    clip = VideoFileClip(video_path).subclip(0, min(duration, VideoFileClip(video_path).duration))
    clip.write_gif(out, fps=fps, program='ffmpeg', verbose=False)
    clip.close()
    return out

def video_to_animated_sticker(video_path, out=None):
    if not MOVIEPY_OK:
        raise RuntimeError("moviepy غير مثبت")
    if out is None:
        out = tmpname("out_anim", "webm")
    clip = VideoFileClip(video_path)
    duration = min(5, clip.duration)
    sub = clip.subclip(0, duration)
    sub = sub.resize(width=512)
    sub.write_videofile(out, codec="libvpx", audio=False, logger=None, threads=4, verbose=False, bitrate="500k")
    clip.close()
    return out

# ---- Save incoming files ----
def save_photo(msg):
    file_info = bot.get_file(msg.photo[-1].file_id)
    data = bot.download_file(file_info.file_path)
    fname = tmpname("tmp_img", "jpg")
    with open(fname, "wb") as f:
        f.write(data)
    return fname

def save_video(msg):
    file_info = bot.get_file(msg.video.file_id)
    data = bot.download_file(file_info.file_path)
    fname = tmpname("tmp_vid", "mp4")
    with open(fname, "wb") as f:
        f.write(data)
    return fname

def save_document(msg):
    file_info = bot.get_file(msg.document.file_id)
    data = bot.download_file(file_info.file_path)
    ext = msg.document.file_name.split('.')[-1] if msg.document.file_name else 'jpg'
    fname = tmpname("tmp_doc", ext)
    with open(fname, "wb") as f:
        f.write(data)
    return fname

# ---- UI ----
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
    start_text = f"""
👋 مرحبًا! أنا بوت أحمد خان. 

🎨 **ميزاتي المتاحة:**
• تحسين الصور وإزالة الخلفية
• تحويل إلى كرتون وASCII
• إضافة علامة مائية وضغط الصور
• تحويل إلى PDF وملصقات
• معالجة الفيديو وتحويل إلى GIF
• 🤖 **توليد صور بالذكاء الاصطناعي (مجاني!)**

أرسل صورة أو فيديو ثم اختر العملية.
    """
    bot.reply_to(m, start_text, reply_markup=keyboard())

@bot.message_handler(content_types=['photo'])
def on_photo(m):
    try:
        fname = save_photo(m)
        uid = m.from_user.id
        st = user_states.setdefault(uid, {"images": [], "videos": [], "pending": None})
        st["images"].append(fname)
        bot.reply_to(m, f"✅ تم حفظ الصورة (#{len(st['images'])}). يمكنك إرسال المزيد أو اختيار عملية.", reply_markup=keyboard())
    except Exception as e:
        bot.reply_to(m, f"❌ خطأ أثناء حفظ الصورة: {e}")

@bot.message_handler(content_types=['video'])
def on_video(m):
    try:
        fname = save_video(m)
        uid = m.from_user.id
        st = user_states.setdefault(uid, {"images": [], "videos": [], "pending": None})
        st["videos"].append(fname)
        bot.reply_to(m, f"✅ تم حفظ الفيديو (#{len(st['videos'])}). اختر العملية.", reply_markup=keyboard())
    except Exception as e:
        bot.reply_to(m, f"❌ خطأ أثناء حفظ الفيديو: {e}")

@bot.message_handler(content_types=['document'])
def on_document(m):
    try:
        # Check if it's an image document
        if m.document.mime_type.startswith('image/'):
            fname = save_document(m)
            uid = m.from_user.id
            st = user_states.setdefault(uid, {"images": [], "videos": [], "pending": None})
            st["images"].append(fname)
            bot.reply_to(m, f"✅ تم حفظ الصورة (#{len(st['images'])}). يمكنك إرسال المزيد أو اختيار عملية.", reply_markup=keyboard())
    except Exception as e:
        bot.reply_to(m, f"❌ خطأ أثناء حفظ الملف: {e}")

# ---- Main action handler ----
@bot.message_handler(func=lambda m: True)
def handle_action(m):
    uid = m.from_user.id
    st = user_states.get(uid)
    
    # إذا كان يريد توليد صور ولم يرسل وصف بعد
    if m.text.strip() == "توليد صورة بالذكاء الاصطناعي":
        bot.reply_to(m, "🔄 أرسل وصف (prompt) للصورة التي تريد توليدها بالعربية أو الإنجليزية:")
        user_states.setdefault(uid, {"images": [], "videos": [], "pending": "ai_generate"})
        return
        
    if not st or (not st["images"] and not st["videos"]):
        bot.reply_to(m, "⚠️ أرسل صورة أو فيديو أولاً ثم اختر العملية.", reply_markup=keyboard())
        return

    action = m.text.strip()
    try:
        # image single operations use last image
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
            bot.reply_to(m, "أدخل زاوية التدوير (90, 180, 270):")
            st["pending"] = {"action": "rotate", "image": st["images"][-1]}
            return

        elif action == "تحويل إلى ملصق":
            inp = st["images"][-1]
            out = image_to_sticker(inp)
            try:
                with open(out, "rb") as s:
                    bot.send_sticker(m.chat.id, s)
                bot.send_message(m.chat.id, f"✅ تم إنشاء الملصق\nالمطور: {USER_TAG}")
            except Exception:
                send_doc(m.chat.id, out, caption=f"✅ تم إنشاء الملصق (ملف webp)\nالمطور: {USER_TAG}")

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

        else:
            bot.reply_to(m, "❓ أمر غير معروف — اختر من الأزرار أو اكتب /help.")
            return

        bot.send_message(m.chat.id, f"✅ تمت العملية. المطور: {USER_TAG}")
    except Exception as e:
        tb = traceback.format_exc()
        bot.reply_to(m, f"❌ حدث خطأ أثناء المعالجة: {e}")
        logger.error(f"Error in handle_action: {tb}")
    finally:
        # cleanup user's inputs and temporary outputs
        imgs = st.get("images", [])
        vids = st.get("videos", [])
        for p in imgs + vids:
            safe_remove(p)
        cleanup_prefix(prefixes=("tmp_", "out_"))
        user_states.pop(uid, None)

# ---- Handler for AI prompt ----
@bot.message_handler(func=lambda m: user_states.get(m.from_user.id, {}).get("pending") == "ai_generate")
def handle_ai_prompt(m):
    uid = m.from_user.id
    st = user_states.get(uid)
    
    if not st:
        return
        
    prompt = m.text.strip()
    try:
        bot.send_message(m.chat.id, "⏳ جاري توليد الصورة... (قد يستغرق 1-2 دقائق)")
        
        out_path = generate_ai_image_free(prompt)
        
        with open(out_path, 'rb') as photo:
            bot.send_photo(m.chat.id, photo, caption=f"🖼️ تم التوليد بنجاح!\nالوصف: {prompt}\n{USER_TAG}")
        
        safe_remove(out_path)
        logger.info(f"Successfully generated image for user {uid}")
        
    except Exception as e:
        error_msg = f"❌ فشل في توليد الصورة: {str(e)}"
        bot.reply_to(m, error_msg)
        logger.error(f"AI generation failed for user {uid}: {str(e)}")
    
    finally:
        # Reset user state
        st["pending"] = None

# ---- Handler for rotate prompt ----
@bot.message_handler(func=lambda m: user_states.get(m.from_user.id, {}).get("pending", {}).get("action") == "rotate")
def handle_rotate_prompt(m):
    uid = m.from_user.id
    st = user_states.get(uid)
    
    if not st or not st.get("pending"):
        return
        
    try:
        angle = int(m.text.strip())
        if angle not in [90, 180, 270]:
            bot.reply_to(m, "الزاوية يجب أن تكون 90, 180, أو 270 فقط")
            return
            
        image_path = st["pending"]["image"]
        out = rotate_image(image_path, angle)
        send_photo(m.chat.id, out, caption=f"🔁 تدوير {angle}°\nالمطور: {USER_TAG}")
        safe_remove(out)
        
    except ValueError:
        bot.reply_to(m, "رجاء أدخل رقم صحيح للزاوية")
        return
    except Exception as e:
        bot.reply_to(m, f"❌ حدث خطأ أثناء التدوير: {e}")
    finally:
        st["pending"] = None

def keep_alive():
    """إرسال نبضات حياة كل 5 دقائق"""
    while True:
        time.sleep(300)
        print("🤖 Bot is still alive...")

if __name__ == "__main__":
    # بدء thread لحفظ البوت نشط
    keep_alive_thread = threading.Thread(target=keep_alive, daemon=True)
    keep_alive_thread.start()
    
    logger.info("ImageBot starting with Stable Diffusion support...")
    if SD_AVAILABLE:
        logger.info("Stable Diffusion is available!")
    else:
        logger.warning("Stable Diffusion not available. Install: pip install diffusers transformers accelerate torch torchvision")
    
    if not REMBG_AVAILABLE:
        logger.warning("rembg not available. Background removal feature disabled.")
    
    try:
        bot.infinity_polling()
    except Exception as e:
        logger.error(f"Bot stopped: {e}")
