import telebot
from telebot import types
import cv2
import numpy as np
import os
from rembg import remove
from PIL import Image, ImageEnhance, ImageFont, ImageDraw
import io

TOKEN = "8266863176:AAHkWNFZYNK2v_RAPWU3E0q0x6wY0IJqArc"
bot = telebot.TeleBot(TOKEN)

# --- الوظائف ---

def enhance_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    enhanced = cv2.filter2D(img, -1, kernel)
    output = "enhanced.jpg"
    cv2.imwrite(output, enhanced)
    return output

def remove_bg(image_path):
    with open(image_path, "rb") as f:
        input_img = f.read()
    output_img = remove(input_img)
    out_path = "no_bg.png"
    with open(out_path, "wb") as f:
        f.write(output_img)
    return out_path

def cartoonify_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray,5)
    edges = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,9,9)
    color = cv2.bilateralFilter(img,9,300,300)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    out_path = "cartoon.jpg"
    cv2.imwrite(out_path, cartoon)
    return out_path

def image_to_ascii(image_path):
    img = Image.open(image_path).convert("L")
    pixels = np.array(img)
    chars = ["@", "#", "S", "%", "?", "*", "+", ";", ":", ",", "."]
    ascii_str = ""
    for row in pixels:
        for px in row:
            ascii_str += chars[px // 25]
        ascii_str += "\n"
    out_path = "ascii.txt"
    with open(out_path, "w") as f:
        f.write(ascii_str)
    return out_path

def add_watermark(image_path, watermark_text="@AHMED_KHANA"):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 36)
    except:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0,0), watermark_text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    width, height = img.size
    position = (width - text_width - 10, height - text_height - 10)
    draw.text(position, watermark_text, fill=(255,255,255), font=font)
    out_path = "watermarked.jpg"
    img.save(out_path)
    return out_path

def image_to_pdf(image_path):
    img = Image.open(image_path)
    out_path = "output.pdf"
    img.convert("RGB").save(out_path)
    return out_path

def compress_image(image_path):
    img = Image.open(image_path)
    out_path = "compressed.jpg"
    img.save(out_path,"JPEG",quality=85)
    return out_path

# --- واجهة المستخدم ---

def create_keyboard():
    markup = types.ReplyKeyboardMarkup(row_width=2)
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
def send_welcome(message):
    bot.reply_to(message,
        f"👋 مرحباً أنا بوت أحمد خان، كيف يمكنني مساعدتك؟",
        reply_markup=create_keyboard()
    )

@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    bot.reply_to(message,"🔄 جاري حفظ الصورة ومعالجتها...")
    file_info = bot.get_file(message.photo[-1].file_id)
    downloaded_file = bot.download_file(file_info.file_path)
    image_path = "input.jpg"
    with open(image_path,'wb') as f:
        f.write(downloaded_file)
    bot.reply_to(message,"✅ الصورة تم حفظها بنجاح! اختر العملية التي تريد تطبيقها:", reply_markup=create_keyboard())

@bot.message_handler(func=lambda message: True)
def process_image_action(message):
    image_path = "input.jpg"
    if not os.path.exists(image_path):
        bot.reply_to(message,"❌ لا توجد صورة مرفوعة، أرسل صورة أولاً.")
        return

    action = message.text
    out_path = None
    caption = ""

    try:
        if action == "تحسين الصورة":
            out_path = enhance_image(image_path)
            caption = "✅ تم تحسين الصورة!\nبواسطة @AHMED_KHANA"

        elif action == "إزالة الخلفية":
            out_path = remove_bg(image_path)
            caption = "🖼️ تم إزالة الخلفية!\nبواسطة @AHMED_KHANA"

        elif action == "تحويل إلى كرتونية":
            out_path = cartoonify_image(image_path)
            caption = "🎨 تم تحويل الصورة إلى كرتونية!\nبواسطة @AHMED_KHANA"

        elif action == "تحويل إلى ASCII":
            out_path = image_to_ascii(image_path)
            with open(out_path,'r') as f:
                ascii_art = f.read()
            bot.send_message(message.chat.id, "📜 هذا هو تحويل الصورة إلى ASCII:\n"+ascii_art)
            return

        elif action == "إضافة علامة مائية":
            out_path = add_watermark(image_path)
            caption = "💧 تم إضافة العلامة المائية!\nبواسطة @AHMED_KHANA"

        elif action == "تحويل إلى PDF":
            out_path = image_to_pdf(image_path)
            with open(out_path,'rb') as f:
                bot.send_document(message.chat.id,f,caption="📄 تم تحويل الصورة إلى PDF!\nبواسطة @AHMED_KHANA")
            return

        elif action == "ضغط الصورة":
            out_path = compress_image(image_path)
            caption = "📉 تم ضغط الصورة!\nبواسطة @AHMED_KHANA"

        if out_path and os.path.exists(out_path):
            with open(out_path,'rb') as f:
                bot.send_photo(message.chat.id,f,caption=caption)
    except Exception as e:
        bot.reply_to(message,f"❌ حدث خطأ: {str(e)}")
    finally:
        # تنظيف الملفات المؤقتة
        for f in ["enhanced.jpg","no_bg.png","cartoon.jpg","watermarked.jpg","ascii.txt","output.pdf","compressed.jpg"]:
            if os.path.exists(f):
                os.remove(f)
        if os.path.exists("input.jpg"):
            os.remove("input.jpg")

bot.polling()
