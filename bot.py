# bot.py
import os
import telebot
from telebot import types
import cv2
import numpy as np
from rembg import remove
from PIL import Image, ImageDraw, ImageFont

TOKEN = os.getenv("BOT_TOKEN")
if not TOKEN:
    raise SystemExit("ERROR: BOT_TOKEN environment variable not set.")

bot = telebot.TeleBot(TOKEN)
USER_TAG = "@AHMED_KHANA"
DEV_NOTE = " المطور"

# utility: safe send photo file
def send_photo_file(chat_id, path, caption=None):
    with open(path, "rb") as f:
        bot.send_photo(chat_id, f, caption=caption)

# image ops
def enhance_image(image_path, out="enhanced.jpg"):
    img = cv2.imread(image_path)
    if img is None:
        return None
    kernel = np.array([[0, -1, 0],[-1, 5,-1],[0, -1, 0]])
    enhanced = cv2.filter2D(img, -1, kernel)
    cv2.imwrite(out, enhanced)
    return out

def remove_bg(image_path, out="no_bg.png"):
    with open(image_path, "rb") as f:
        data = f.read()
    result = remove(data)
    with open(out, "wb") as f:
        f.write(result)
    return out

def cartoonify_image(image_path, out="cartoon.jpg"):
    img = cv2.imread(image_path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray,5)
    edges = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,9,9)
    color = cv2.bilateralFilter(img,9,300,300)
    cartoon = cv2.bitwise_and(color,color,mask=edges)
    cv2.imwrite(out, cartoon)
    return out

def image_to_ascii(image_path, out="ascii.txt"):
    img = Image.open(image_path).convert("L")
    pixels = np.array(img)
    chars = ["@", "#", "S", "%", "?", "*", "+", ";", ":", ",", "."]
    lines = []
    for row in pixels:
        line = "".join(chars[p//25] for p in row)
        lines.append(line)
    text = "\n".join(lines)
    with open(out, "w", encoding="utf-8") as f:
        f.write(text)
    return out

def add_watermark(image_path, out="watermarked.jpg", text=None):
    if text is None:
        text = f"{USER_TAG} - {DEV_NOTE}"
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 28)
    except:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0,0), text, font=font)
    w = bbox[2]-bbox[0]; h = bbox[3]-bbox[1]
    width, height = img.size
    pos = (width - w - 10, height - h - 10)
    # draw semi-transparent rectangle for readability
    rect_x0, rect_y0 = pos[0]-6, pos[1]-6
    rect_x1, rect_y1 = pos[0]+w+6, pos[1]+h+6
    overlay = Image.new('RGBA', img.size, (0,0,0,0))
    odraw = ImageDraw.Draw(overlay)
    odraw.rectangle((rect_x0, rect_y0, rect_x1, rect_y1), fill=(0,0,0,120))
    img = Image.alpha_composite(img.convert('RGBA'), overlay)
    draw = ImageDraw.Draw(img)
    draw.text(pos, text, fill=(255,255,255,255), font=font)
    img.convert('RGB').save(out)
    return out

def image_to_pdf(image_path, out="output.pdf"):
    img = Image.open(image_path)
    img.convert("RGB").save(out)
    return out

def compress_image(image_path, out="compressed.jpg", quality=85):
    img = Image.open(image_path)
    img.save(out, "JPEG", quality=quality)
    return out

# keyboard
def keyboard():
    markup = types.ReplyKeyboardMarkup(row_width=2, resize_keyboard=True)
    buttons = [
        types.KeyboardButton("تحسين الصورة"),
        types.KeyboardButton("إزالة الخلفية"),
        types.KeyboardButton("تحويل إلى كرتونية"),
        types.KeyboardButton("تحويل إلى ASCII"),
        types.KeyboardButton("إضافة علامة مائية"),
        types.KeyboardButton("تحويل إلى PDF"),
        types.KeyboardButton("ضغط الصورة")
    ]
    markup.add(*buttons)
    return markup

@bot.message_handler(commands=['start'])
def start(m):
    bot.reply_to(m, "👋 مرحباً! أنا بوت أحمد خان، كيف يمكنني مساعدتك؟", reply_markup=keyboard())

# save incoming photo safely with unique name to avoid race conditions
import uuid
def save_photo_from_message(message):
    file_info = bot.get_file(message.photo[-1].file_id)
    data = bot.download_file(file_info.file_path)
    filename = f"input_{uuid.uuid4().hex}.jpg"
    with open(filename, "wb") as f:
        f.write(data)
    return filename

@bot.message_handler(content_types=['photo'])
def on_photo(m):
    saved = save_photo_from_message(m)
    # store filename in user's state (simple dict) so next command uses it
    user_states[m.from_user.id] = saved
    bot.reply_to(m, "✅ تم حفظ الصورة. اختر العملية المراد تنفيذها:", reply_markup=keyboard())

# simple in-memory state (for demo; for production use persistent DB)
user_states = {}

@bot.message_handler(func=lambda msg: True)
def handle_action(message):
    uid = message.from_user.id
    if uid not in user_states:
        bot.reply_to(message, "⚠️ أرسل صورة أولاً ثم اختر العملية.")
        return
    input_path = user_states.get(uid)
    if not input_path or not os.path.exists(input_path):
        bot.reply_to(message, "⚠️ المشكلة: الصورة غير موجودة. أرسل الصورة مرة أخرى.")
        user_states.pop(uid, None)
        return

    action = message.text
    try:
        if action == "تحسين الصورة":
            out = enhance_image(input_path)
            caption = f"✅ تم تحسين الصورة!\nالمطور: {USER_TAG}"
            send_photo_file(message.chat.id, out, caption)

        elif action == "إزالة الخلفية":
            out = remove_bg(input_path)
            caption = f"🖼️ تمت إزالة الخلفية\nالمطور: {USER_TAG}"
            send_photo_file(message.chat.id, out, caption)

        elif action == "تحويل إلى كرتونية":
            out = cartoonify_image(input_path)
            caption = f"🎨 تم التحويل لكرتونية\nالمطور: {USER_TAG}"
            send_photo_file(message.chat.id, out, caption)

        elif action == "تحويل إلى ASCII":
            out = image_to_ascii(input_path)
            with open(out, "r", encoding="utf-8") as f:
                text = f.read()
            bot.send_message(message.chat.id, f"📜 ASCII:\n{text}\nالمطور: {USER_TAG}")

        elif action == "إضافة علامة مائية":
            out = add_watermark(input_path)
            caption = f"💧 تم إضافة علامة مائية\nالمطور: {USER_TAG}"
            send_photo_file(message.chat.id, out, caption)

        elif action == "تحويل إلى PDF":
            out = image_to_pdf(input_path)
            with open(out, "rb") as f:
                bot.send_document(message.chat.id, f, caption=f"📄 تم التحويل إلى PDF\nالمطور: {USER_TAG}")

        elif action == "ضغط الصورة":
            out = compress_image(input_path)
            caption = f"📉 تم ضغط الصورة\nالمطور: {USER_TAG}"
            send_photo_file(message.chat.id, out, caption)

        else:
            bot.reply_to(message, "❓ أمر غير معروف. اختر من لوحة الأزرار.")
            return

    except Exception as e:
        bot.reply_to(message, f"❌ خطأ أثناء المعالجة: {e}")

    finally:
        # cleanup user's input and any generated outputs
        try:
            if os.path.exists(input_path):
                os.remove(input_path)
        except: pass
        # remove generated files matching patterns
        for fn in os.listdir("."):
            if any(fn.endswith(s) for s in (".jpg",".png",".pdf",".txt")) and fn.startswith(("enhanced","no_bg","cartoon","watermarked","compressed","output","ascii")):
                try: os.remove(fn)
                except: pass
        user_states.pop(uid, None)

if __name__ == "__main__":
    print("Bot is starting...")
    bot.infinity_polling()
