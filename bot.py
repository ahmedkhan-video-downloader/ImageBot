# bot.py
import os
import uuid
import shutil
import traceback
import telebot
from telebot import types
import cv2
import numpy as np
from rembg import remove
from PIL import Image, ImageDraw, ImageFont, ImageOps

# ====== إعداد BOT_TOKEN من متغير بيئة ======
TOKEN = os.getenv("BOT_TOKEN")
if not TOKEN:
    raise SystemExit("ERROR: BOT_TOKEN environment variable not set. ضع توكن البوت في متغير البيئة BOT_TOKEN")

bot = telebot.TeleBot(TOKEN)

# ثوابت
USER_TAG = "@AHMED_KHANA"
DEV_NOTE = " المطور"

# حالة مستخدم مؤقتة (in-memory). للإنتاج استخدم DB إذا أردت بقاء عبر إعادة تشغيل.
# user_states[user_id] = [list_of_saved_image_paths]
user_states = {}

# --- Utilities ---
def safe_remove(path):
    try:
        if os.path.exists(path):
            os.remove(path)
    except:
        pass

def cleanup_user_files(prefixes=("input_", "out_")):
    """حذف أي ملفات مؤقتة تبدأ بمقدمات محددة"""
    for fn in os.listdir("."):
        if any(fn.startswith(p) for p in prefixes):
            try:
                os.remove(fn)
            except:
                pass

def send_photo_file(chat_id, path, caption=None):
    with open(path, "rb") as f:
        bot.send_photo(chat_id, f, caption=caption)

def send_doc_file(chat_id, path, caption=None):
    with open(path, "rb") as f:
        bot.send_document(chat_id, f, caption=caption)

# --- حفظ الصورة المرسلة باسم فريد ---
def save_photo_from_message(message):
    file_info = bot.get_file(message.photo[-1].file_id)
    data = bot.download_file(file_info.file_path)
    fname = f"input_{uuid.uuid4().hex}.jpg"
    with open(fname, "wb") as f:
        f.write(data)
    return fname

# ====== وظائف معالجة الصور ======

def enhance_image(image_path, out_path=None):
    if out_path is None:
        out_path = f"out_enhanced_{uuid.uuid4().hex}.jpg"
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("ملف الصورة غير صالح")
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    enhanced = cv2.filter2D(img, -1, kernel)
    cv2.imwrite(out_path, enhanced)
    return out_path

def remove_bg_image(image_path, out_path=None):
    if out_path is None:
        out_path = f"out_nobg_{uuid.uuid4().hex}.png"
    with open(image_path, "rb") as f:
        data = f.read()
    result = remove(data)
    with open(out_path, "wb") as f:
        f.write(result)
    return out_path

def cartoonify_image(image_path, out_path=None):
    if out_path is None:
        out_path = f"out_cartoon_{uuid.uuid4().hex}.jpg"
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("ملف الصورة غير صالح")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(img, 9, 300, 300)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    cv2.imwrite(out_path, cartoon)
    return out_path

def image_to_ascii_file(image_path, out_path=None, width=100):
    if out_path is None:
        out_path = f"out_ascii_{uuid.uuid4().hex}.txt"
    img = Image.open(image_path).convert("L")
    # resize preserving aspect ratio to target width
    aspect_ratio = img.height / img.width
    height = int(aspect_ratio * width * 0.55)
    img = img.resize((width, max(1, height)))
    pixels = np.array(img)
    chars = "@%#*+=-:. "  # gradient
    lines = []
    for row in pixels:
        line = "".join(chars[pixel * len(chars) // 256] for pixel in row)
        lines.append(line)
    txt = "\n".join(lines)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(txt)
    return out_path

def add_watermark(image_path, out_path=None, text=None):
    if out_path is None:
        out_path = f"out_watermark_{uuid.uuid4().hex}.jpg"
    if text is None:
        text = f"{USER_TAG} - {DEV_NOTE}"
    img = Image.open(image_path).convert("RGBA")
    iw, ih = img.size
    # try load nicer font, fallback to default
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 28)
    except:
        font = ImageFont.load_default()
    draw = ImageDraw.Draw(img)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]; th = bbox[3] - bbox[1]
    pos = (iw - tw - 12, ih - th - 12)
    # draw semi-transparent box for readability
    overlay = Image.new('RGBA', img.size, (0,0,0,0))
    od = ImageDraw.Draw(overlay)
    od.rectangle([pos[0]-6, pos[1]-6, pos[0]+tw+6, pos[1]+th+6], fill=(0,0,0,120))
    combined = Image.alpha_composite(img, overlay)
    draw2 = ImageDraw.Draw(combined)
    draw2.text(pos, text, fill=(255,255,255,255), font=font)
    combined.convert("RGB").save(out_path)
    return out_path

def image_to_pdf(images_list, out_path=None):
    if out_path is None:
        out_path = f"out_pdf_{uuid.uuid4().hex}.pdf"
    pil_images = [Image.open(p).convert("RGB") for p in images_list]
    if not pil_images:
        raise ValueError("لا يوجد صور لتحويلها إلى PDF")
    first, rest = pil_images[0], pil_images[1:]
    first.save(out_path, save_all=True, append_images=rest)
    return out_path

def compress_image(image_path, out_path=None, quality=75):
    if out_path is None:
        out_path = f"out_compressed_{uuid.uuid4().hex}.jpg"
    img = Image.open(image_path).convert("RGB")
    img.save(out_path, "JPEG", quality=quality)
    return out_path

def invert_colors(image_path, out_path=None):
    if out_path is None:
        out_path = f"out_invert_{uuid.uuid4().hex}.jpg"
    img = Image.open(image_path).convert("RGB")
    inv = ImageOps.invert(img)
    inv.save(out_path)
    return out_path

def bw_image(image_path, out_path=None):
    if out_path is None:
        out_path = f"out_bw_{uuid.uuid4().hex}.jpg"
    img = Image.open(image_path).convert("L")
    img.save(out_path)
    return out_path

def rotate_image(image_path, angle, out_path=None):
    if out_path is None:
        out_path = f"out_rotate_{uuid.uuid4().hex}.jpg"
    img = Image.open(image_path).convert("RGB")
    rotated = img.rotate(angle, expand=True)
    rotated.save(out_path)
    return out_path

def to_sticker(image_path, out_path=None):
    # produce PNG 512x512 for telegram sticker
    if out_path is None:
        out_path = f"out_sticker_{uuid.uuid4().hex}.png"
    img = Image.open(image_path).convert("RGBA")
    # make square 512x512
    size = 512
    img.thumbnail((size, size), Image.ANTIALIAS)
    # place on transparent background centered
    bg = Image.new("RGBA", (size, size), (0,0,0,0))
    w,h = img.size
    pos = ((size-w)//2, (size-h)//2)
    bg.paste(img, pos, img)
    bg.save(out_path)
    return out_path

# ====== واجهة المستخدم و الأزرار ======

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
        types.KeyboardButton("تحويل إلى ملصق")
    ]
    markup.add(*buttons)
    return markup

@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    bot.reply_to(message,
                 f"👋 مرحبًا! أنا بوت أحمد خان. كيف يمكنني مساعدتك اليوم؟\nأرسل صورة ثم اختر العملية من الأزرار.",
                 reply_markup=keyboard())

# استقبال الصور (يدعم عدة صور؛ نخزنها في قائمة كل مستخدم)
@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    try:
        saved = save_photo_from_message(message)
        uid = message.from_user.id
        user_states.setdefault(uid, []).append(saved)
        count = len(user_states[uid])
        bot.reply_to(message, f"✅ تم حفظ الصورة (#{count}). يمكنك إرسال صور إضافية أو اختر عملية لتحويل الصور المحفوظة.", reply_markup=keyboard())
    except Exception as e:
        bot.reply_to(message, f"❌ حدث خطأ أثناء حفظ الصورة: {e}")

# التعامل مع الأوامر/الأزرار
@bot.message_handler(func=lambda m: True)
def handle_action(message):
    uid = message.from_user.id
    if uid not in user_states or not user_states[uid]:
        bot.reply_to(message, "⚠️ أرسل صورة أولاً (يمكنك إرسال أكثر من صورة لتحويلها إلى PDF)، ثم اختر العملية.")
        return

    action = message.text
    images = user_states.get(uid, [])
    in_path = images[-1]  # نأخذ آخر صورة لو العملية تتعامل بصورة واحدة
    try:
        if action == "تحسين الصورة":
            out = enhance_image(in_path)
            caption = f"✅ تم تحسين الصورة!\nالمطور: {USER_TAG}"
            send_photo_file(message.chat.id, out, caption=caption)

        elif action == "إزالة الخلفية":
            out = remove_bg_image(in_path)
            caption = f"🖼️ تمت إزالة الخلفية\nالمطور: {USER_TAG}"
            send_photo_file(message.chat.id, out, caption=caption)

        elif action == "تحويل إلى كرتونية":
            out = cartoonify_image(in_path)
            caption = f"🎨 تم التحويل لكرتونية\nالمطور: {USER_TAG}"
            send_photo_file(message.chat.id, out, caption=caption)

        elif action == "تحويل إلى ASCII":
            out = image_to_ascii_file(in_path, width=120)
            caption = f"📜 تحويل إلى ASCII\nالمطور: {USER_TAG}"
            send_doc_file(message.chat.id, out, caption=caption)

        elif action == "إضافة علامة مائية":
            out = add_watermark(in_path)
            caption = f"💧 تم إضافة العلامة المائية\nالمطور: {USER_TAG}"
            send_photo_file(message.chat.id, out, caption=caption)

        elif action == "تحويل إلى PDF":
            # استخدم كل الصور المرسلة من قبل المستخدم لتحويلها إلى PDF واحد
            out = image_to_pdf(images)
            caption = f"📄 تم تحويل الصور إلى PDF\nالمطور: {USER_TAG}"
            send_doc_file(message.chat.id, out, caption=caption)

        elif action == "ضغط الصورة":
            out = compress_image(in_path, quality=70)
            caption = f"📉 تم ضغط الصورة\nالمطور: {USER_TAG}"
            send_photo_file(message.chat.id, out, caption=caption)

        elif action == "أبيض وأسود":
            out = bw_image(in_path)
            caption = f"⚪⬛ فلتر أبيض وأسود\nالمطور: {USER_TAG}"
            send_photo_file(message.chat.id, out, caption=caption)

        elif action == "عكس الألوان":
            out = invert_colors(in_path)
            caption = f"🌀 عكس الألوان\nالمطور: {USER_TAG}"
            send_photo_file(message.chat.id, out, caption=caption)

        elif action == "تدوير الصورة":
            # مثال: ندوّر 90 درجة. لو تريد زوايا مختلفة نضيف حوار/خطوة إضافية
            out = rotate_image(in_path, angle=90)
            caption = f"🔁 تم تدوير الصورة 90°\nالمطور: {USER_TAG}"
            send_photo_file(message.chat.id, out, caption=caption)

        elif action == "تحويل إلى ملصق":
            out = to_sticker(in_path)
            with open(out, "rb") as s:
                bot.send_sticker(message.chat.id, s)
            bot.send_message(message.chat.id, f"✅ تم إنشاء الملصق\nالمطور: {USER_TAG}")

        else:
            bot.reply_to(message, "❓ الأمر غير معروف، اختر من الأزرار أو اكتب /help")
            return

    except Exception as e:
        tb = traceback.format_exc()
        bot.reply_to(message, f"❌ خطأ أثناء المعالجة: {e}\n\n{tb}")
    finally:
        # تنظيف الملفات الخاصة بالمستخدم
        for p in images:
            safe_remove(p)
        # تنظيف أي ملفات مؤقتة ناتجة عن المعالجة (تعتمد على prefix out_)
        cleanup_user_files(prefixes=("out_",))
        user_states.pop(uid, None)

# شغّل البوت
if __name__ == "__main__":
    print("Bot starting...")
    bot.infinity_polling()
