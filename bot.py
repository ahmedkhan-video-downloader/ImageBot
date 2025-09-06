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
from functools import partial

import telebot
from telebot import types

import cv2
import numpy as np
from rembg import remove
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

# ----- Config via env -----
BOT_TOKEN = os.getenv("BOT_TOKEN")
if not BOT_TOKEN:
    raise SystemExit("ERROR: BOT_TOKEN environment variable not set. ضع توكن البوت في متغير البيئة BOT_TOKEN")

bot = telebot.TeleBot(BOT_TOKEN)

USER_TAG = "@AHMED_KHANA"
DEV_NOTE = " المطور"

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

# ---- باقي دوال الصور (نفسها كما كانت) ----
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

# ... (جميع الدوال الأخرى تبقى كما هي بدون تغيير) ...

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
        types.KeyboardButton("توليد صورة بالذكاء الاصطناعي")  # 🤖 ميزة جديدة
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
        bot.reply_to(m, f"✅ تم حفظ الصورة (#{len(st['images'])}). يمكنك إرسال المزيد أو اختيار عملية.", reply_mup=keyboard())
    except Exception as e:
        bot.reply_to(m, f"❌ خطأ أثناء حفظ الصورة: {e}")

# ... (بقية handlers تبقى كما هي) ...

## ---- Main action handler ----
@bot.message_handler(func=lambda m: True)
def handle_action(m):
    uid = m.from_user.id
    st = user_states.get(uid)
    
    # معالجة طلب توليد الصور أولاً
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

if __name__ == "__main__":
    logger.info("ImageBot starting with Stable Diffusion support...")
    if SD_AVAILABLE:
        logger.info("Stable Diffusion is available!")
    else:
        logger.warning("Stable Diffusion not available. Install: pip install diffusers transformers accelerate torch torchvision")
    
    bot.infinity_polling()
