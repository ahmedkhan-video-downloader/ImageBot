import telebot
from telebot import types
import cv2
import numpy as np
import os
from rembg import remove
from PIL import Image, ImageFont, ImageDraw

TOKEN = "8266863176:AAHkWNFZYNK2v_RAPWU3E0q0x6wY0IJqArc"
bot = telebot.TeleBot(TOKEN)

USER_TAG = "@AHMED_KHANA"
DEVELOPER_NOTE = " المطور"

# تحسين الصورة
def enhance_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    enhanced_image = cv2.filter2D(image, -1, kernel)
    output_path = "enhanced.jpg"
    cv2.imwrite(output_path, enhanced_image)
    return output_path

# إزالة الخلفية
def remove_bg(image_path):
    with open(image_path, "rb") as file:
        input_image = file.read()
    output_image = remove(input_image)
    output_path = "no_bg.png"
    with open(output_path, "wb") as file:
        file.write(output_image)
    return output_path

# تحويل الصورة لكرتونية
def cartoonify_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,9,9)
    color = cv2.bilateralFilter(img,9,300,300)
    cartoon = cv2.bitwise_and(color,color,mask=edges)
    output_path = "cartoon.jpg"
    cv2.imwrite(output_path, cartoon)
    return output_path

# تحويل الصورة إلى ASCII
def image_to_ascii(image_path):
    img = Image.open(image_path).convert('L')
    pixels = np.array(img)
    ascii_str = ""
    chars = ["@", "#", "S", "%", "?", "*", "+", ";", ":", ",", "."]
    for row in pixels:
        for pixel in row:
            ascii_str += chars[pixel // 25]
        ascii_str += "\n"
    output_path = "ascii.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(ascii_str)
    return output_path

# إضافة علامة مائية
def add_watermark(image_path):
    img = Image.open(image_path)
    width, height = img.size
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    text = f"{USER_TAG} - {DEVELOPER_NOTE}"
    text_width, text_height = draw.textsize(text, font)
    position = (width - text_width - 10, height - text_height - 10)
    draw.text(position, text, (255,255,255), font)
    watermark_path = "watermarked.jpg"
    img.save(watermark_path)
    return watermark_path

# تحويل الصورة إلى PDF
def image_to_pdf(image_path):
    img = Image.open(image_path)
    pdf_path = "output.pdf"
    img.convert('RGB').save(pdf_path)
    return pdf_path

# ضغط الصورة
def compress_image(image_path):
    img = Image.open(image_path)
    compressed_path = "compressed.jpg"
    img.save(compressed_path,"JPEG",quality=85)
    return compressed_path

# رسالة الترحيب
@bot.message_handler(commands=['start'])
def send_welcome(message):
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
    bot.reply_to(message, f"👋 مرحباً! أنا بوت أحمد خان، كيف يمكنني مساعدتك؟", reply_markup=markup)

# استقبال الصور
@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    file_info = bot.get_file(message.photo[-1].file_id)
    downloaded_file = bot.download_file(file_info.file_path)
    with open("input.jpg", "wb") as f:
        f.write(downloaded_file)
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
    bot.reply_to(message, "📌 اختر العملية التي تريد تطبيقها على الصورة:", reply_markup=markup)

# معالجة الصور
@bot.message_handler(func=lambda message: True)
def process_image_action(message):
    if not os.path.exists("input.jpg"):
        bot.reply_to(message, "⚠️ لم يتم إرسال صورة بعد!")
        return

    try:
        if message.text == "تحسين الصورة":
            result_path = enhance_image("input.jpg")
            if result_path:
                with open(result_path,"rb") as f:
                    bot.send_photo(message.chat.id,f,caption=f"✅ تم تحسين الصورة!\n{USER_TAG} - {DEVELOPER_NOTE}")
        
        elif message.text == "إزالة الخلفية":
            result_path = remove_bg("input.jpg")
            with open(result_path,"rb") as f:
                bot.send_photo(message.chat.id,f,caption=f"🖼️ تم إزالة الخلفية!\n{USER_TAG} - {DEVELOPER_NOTE}")

        elif message.text == "تحويل إلى كرتونية":
            result_path = cartoonify_image("input.jpg")
            if result_path:
                with open(result_path,"rb") as f:
                    bot.send_photo(message.chat.id,f,caption=f"🎨 تم تحويل الصورة إلى كرتونية!\n{USER_TAG} - {DEVELOPER_NOTE}")

        elif message.text == "تحويل إلى ASCII":
            result_path = image_to_ascii("input.jpg")
            with open(result_path,"r",encoding="utf-8") as f:
                ascii_art = f.read()
            bot.send_message(message.chat.id, f"📜 تحويل الصورة إلى ASCII:\n{ascii_art}\n{USER_TAG} - {DEVELOPER_NOTE}")

        elif message.text == "إضافة علامة مائية":
            result_path = add_watermark("input.jpg")
            with open(result_path,"rb") as f:
                bot.send_photo(message.chat.id,f,caption=f"💧 تم إضافة العلامة المائية!\n{USER_TAG} - {DEVELOPER_NOTE}")

        elif message.text == "تحويل إلى PDF":
            result_path = image_to_pdf("input.jpg")
            with open(result_path,"rb") as f:
                bot.send_document(message.chat.id,f,caption=f"📄 تم تحويل الصورة إلى PDF!\n{USER_TAG} - {DEVELOPER_NOTE}")

        elif message.text == "ضغط الصورة":
            result_path = compress_image("input.jpg")
            with open(result_path,"rb") as f:
                bot.send_photo(message.chat.id,f,caption=f"📉 تم ضغط الصورة!\n{USER_TAG} - {DEVELOPER_NOTE}")

    finally:
        # تنظيف الملفات المؤقتة
        for file in ["input.jpg","enhanced.jpg","no_bg.png","cartoon.jpg","watermarked.jpg","ascii.txt","output.pdf","compressed.jpg"]:
            if os.path.exists(file):
                os.remove(file)

# تشغيل البوت
bot.polling()
