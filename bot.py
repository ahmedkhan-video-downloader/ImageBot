import telebot
from telebot import types
import cv2
import numpy as np
import os
from rembg import remove
from PIL import Image, ImageFont, ImageDraw

# استخدام توكن من متغير بيئة
TOKEN = os.getenv("BOT_TOKEN")
bot = telebot.TeleBot(TOKEN)

# دوال المعالجة (كما سبق)
def enhance_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    enhanced_image = cv2.filter2D(image, -1, kernel)
    output_path = "enhanced.jpg"
    cv2.imwrite(output_path, enhanced_image)
    return output_path

def remove_bg(image_path):
    with open(image_path, "rb") as file:
        input_image = file.read()
    output_image = remove(input_image)
    output_path = "no_bg.png"
    with open(output_path, "wb") as file:
        file.write(output_image)
    return output_path

def cartoonify_image(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(img, 9, 300, 300)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    output_path = "cartoon.jpg"
    cv2.imwrite(output_path, cartoon)
    return output_path

def image_to_ascii(image_path):
    img = Image.open(image_path).convert('L')
    pixels = np.array(img)
    ascii_str = ""
    chars = ["@", "#", "S", "%", "?", "*", "+", ";", ":", ",", "."]
    for row in pixels:
        for pixel in row:
            ascii_str += chars[pixel // 25]
        ascii_str += '\n'
    output_path = "ascii.txt"
    with open(output_path, "w") as f:
        f.write(ascii_str)
    return output_path

def add_watermark(image_path, watermark_text="أحمد خان"):
    img = Image.open(image_path)
    width, height = img.size
    font = ImageFont.load_default()
    draw = ImageDraw.Draw(img)
    text_width, text_height = draw.textsize(watermark_text, font=font)
    position = (width - text_width - 10, height - text_height - 10)
    draw.text(position, watermark_text, (255, 255, 255), font=font)
    watermark_path = "watermarked.jpg"
    img.save(watermark_path)
    return watermark_path

def image_to_pdf(image_path):
    img = Image.open(image_path)
    pdf_path = "output.pdf"
    img.convert('RGB').save(pdf_path)
    return pdf_path

def compress_image(image_path):
    img = Image.open(image_path)
    compressed_path = "compressed.jpg"
    img.save(compressed_path, "JPEG", quality=85)
    return compressed_path

def get_markup():
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
def send_welcome(message):
    bot.reply_to(message, f"  أهلاً بك انا بوت احمد خان، {message.from_user.first_name}!\nاختر ما ترغب في فعله:", reply_markup=get_markup())

@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    bot.reply_to(message, "🔄 جاري تحميل الصورة...")
    file_info = bot.get_file(message.photo[-1].file_id)
    downloaded_file = bot.download_file(file_info.file_path)
    with open("input.jpg", 'wb') as f:
        f.write(downloaded_file)
    bot.reply_to(message, "✅ تم تحميل الصورة. اختر العملية:", reply_markup=get_markup())

@bot.message_handler(func=lambda message: True)
def process_image_action(message):
    image_path = "input.jpg"
    result_path = None

    try:
        if message.text == "تحسين الصورة":
            result_path = enhance_image(image_path)
            with open(result_path, "rb") as file:
                bot.send_photo(message.chat.id, file, caption="✅ تم تحسين الصورة!")
        elif message.text == "إزالة الخلفية":
            result_path = remove_bg(image_path)
            with open(result_path, "rb") as file:
                bot.send_photo(message.chat.id, file, caption="🖼️ تم إزالة الخلفية!")
        elif message.text == "تحويل إلى كرتونية":
            result_path = cartoonify_image(image_path)
            with open(result_path, "rb") as file:
                bot.send_photo(message.chat.id, file, caption="🎨 تم تحويل الصورة إلى كرتونية!")
        elif message.text == "تحويل إلى ASCII":
            result_path = image_to_ascii(image_path)
            with open(result_path, "r") as file:
                ascii_art = file.read()
            bot.send_message(message.chat.id, "📜 تحويل الصورة إلى ASCII:\n" + ascii_art)
        elif message.text == "إضافة علامة مائية":
            result_path = add_watermark(image_path)
            with open(result_path, "rb") as file:
                bot.send_photo(message.chat.id, file, caption="💧 تم إضافة العلامة المائية!")
        elif message.text == "تحويل إلى PDF":
            result_path = image_to_pdf(image_path)
            with open(result_path, "rb") as file:
                bot.send_document(message.chat.id, file, caption="📄 تم تحويل الصورة إلى PDF!")
        elif message.text == "ضغط الصورة":
            result_path = compress_image(image_path)
            with open(result_path, "rb") as file:
                bot.send_photo(message.chat.id, file, caption="📉 تم ضغط الصورة!")
    finally:
        for path in ["input.jpg", "enhanced.jpg", "no_bg.png", "cartoon.jpg",
                     "watermarked.jpg", "ascii.txt", "output.pdf", "compressed.jpg"]:
            if os.path.exists(path):
                os.remove(path)

bot.polling()
