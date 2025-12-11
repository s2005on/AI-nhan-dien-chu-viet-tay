import tkinter as tk
from tkinter import filedialog, Label, Frame
from tkinterdnd2 import DND_FILES, TkinterDnD
from PIL import Image, ImageTk
import cv2
import numpy as np
import tensorflow as tf

# Cấu hình bộ lọc
MIN_AREA = 200  # Diện tích tối thiểu: Bỏ qua các chấm nhỏ hơn mức này (Lọc nhiễu)
MAX_AREA = 5000 # Diện tích tối đa: Bỏ qua khung quá to (ví dụ cả tờ giấy)

try:
    model = tf.keras.models.load_model('model_so.keras')
    print("✅ Đã nạp Model MNIST!")
except:
    print("❌ Lỗi: Thiếu file model_so.keras")
    model = None

class MultiDigitApp(TkinterDnD.Tk):
    def __init__(self):
        super().__init__()
        self.title("Nhận Diện Đa Số - Code V5 (Chống Nhiễu)")
        self.geometry("900x700")
        self.configure(bg="#2c3e50")

        Label(self, text="NHẬN DIỆN SỐ VIẾT TAY (NHIỀU SỐ)", font=("Arial", 20, "bold"), bg="#2c3e50", fg="white").pack(pady=20)

        # Khung Kéo Thả
        self.drop_frame = Frame(self, bg="white", width=700, height=400, highlightthickness=2, highlightbackground="#3498db")
        self.drop_frame.pack_propagate(False)
        self.drop_frame.pack(pady=10)

        self.lbl_msg = Label(self.drop_frame, text="Kéo ảnh vào đây\n(Hỗ trợ nhận diện 2 dòng)", font=("Arial", 14), fg="gray", bg="white")
        self.lbl_msg.place(relx=0.5, rely=0.5, anchor="center")
        
        self.lbl_img = Label(self.drop_frame, bg="white")
        self.lbl_img.place(relx=0.5, rely=0.5, anchor="center")

        self.drop_frame.drop_target_register(DND_FILES)
        self.drop_frame.dnd_bind('<<Drop>>', self.drop_image)
        self.drop_frame.bind("<Button-1>", self.browse_image)

        self.lbl_result = Label(self, text="Kết quả: ...", font=("Consolas", 28, "bold"), fg="#e74c3c", bg="#2c3e50")
        self.lbl_result.pack(pady=20)
        btn_frame = Frame(self, bg="#2c3e50")
        btn_frame.pack(pady=10)

    def browse_image(self, event=None):
        f = filedialog.askopenfilename()
        if f: self.process_image(f)

    def drop_image(self, event):
        path = event.data
        if path.startswith('{') and path.endswith('}'): path = path[1:-1]
        self.process_image(path)

    # --- HÀM SẮP XẾP THÔNG MINH (ĐỌC TỪ TRÊN XUỐNG, TRÁI SANG PHẢI) ---
    def sort_contours(self, cnts, method="left-to-right"):
        bounding_boxes = [cv2.boundingRect(c) for c in cnts]
        if len(cnts) == 0: return []

        # Zip contour với khung hình chữ nhật để sắp xếp
        cnts_boxes = zip(cnts, bounding_boxes)
        
        # Sắp xếp theo trục Y trước (Dòng trên -> Dòng dưới)
        # tolerance=20: Cho phép các số lệch nhau 20 pixel vẫn tính là cùng 1 dòng
        def sort_logic(cb):
            x, y, w, h = cb[1]
            return (y // 50) * 1000 + x # Kỹ thuật Gom nhóm dòng (binning)

        cnts_boxes = sorted(cnts_boxes, key=sort_logic)
        
        return [c[0] for c in cnts_boxes]

    # --- HÀM XỬ LÝ CHÍNH ---
    def process_image(self, path):
        # 1. Hiển thị ảnh
        img_pil = Image.open(path)
        img_pil.thumbnail((600, 380))
        img_tk = ImageTk.PhotoImage(img_pil)
        self.lbl_msg.place_forget()
        self.lbl_img.config(image=img_tk)
        self.lbl_img.image = img_tk

        # 2. Xử lý ảnh OpenCV
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Lọc nhiễu (Gaussian Blur) - Giúp xóa các vết chấm li ti
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Nhị phân hóa (Thresh Otsu)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

        # Tìm Contour
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Lọc bỏ rác (Chỉ lấy contour có kích thước hợp lý)
        valid_contours = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            # Điều kiện lọc: Diện tích phải > MIN_AREA và < MAX_AREA
            # Tỷ lệ khung hình (Aspect Ratio) không được quá dẹt (để loại bỏ dòng kẻ)
            aspect_ratio = w / float(h)
            if w * h > MIN_AREA and w * h < MAX_AREA and aspect_ratio < 3:
                valid_contours.append(c)

        # Sắp xếp contour (Đọc từ trên xuống dưới, trái sang phải)
        sorted_contours = self.sort_contours(valid_contours)

        result_string = ""
        
        if model:
            for c in sorted_contours:
                x, y, w, h = cv2.boundingRect(c)
                
                # Cắt vùng số
                roi = thresh[y:y+h, x:x+w]
                
                # --- Padding (Giữ nguyên tỉ lệ số) ---
                max_side = max(w, h)
                square = np.zeros((max_side, max_side), dtype='uint8')
                dx = (max_side - w) // 2
                dy = (max_side - h) // 2
                square[dy:dy+h, dx:dx+w] = roi
                
                # Thêm viền đen dày (padding) để giống MNIST
                padding = int(max_side * 0.25)
                square = cv2.copyMakeBorder(square, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=0)
                
                # Resize chuẩn 28x28
                roi_final = cv2.resize(square, (28, 28), interpolation=cv2.INTER_AREA)
                roi_norm = roi_final.astype('float32') / 255.0
                roi_input = roi_norm.reshape(1, 28, 28, 1)

                # Dự đoán
                pred = model.predict(roi_input, verbose=0)
                digit = np.argmax(pred)
                confidence = np.max(pred)

                # Chỉ lấy nếu tự tin > 50%
                if confidence > 0.5:
                    result_string += str(digit) + " "
                    # Vẽ khung xanh lên ảnh để debug
                    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(img, str(digit), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        self.lbl_result.config(text=f"Kết quả: {result_string}")
        
        # Update lại ảnh preview với khung xanh
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_final = Image.fromarray(img_rgb)
        img_final.thumbnail((600, 380))
        img_tk_final = ImageTk.PhotoImage(img_final)
        self.lbl_img.config(image=img_tk_final)
        self.lbl_img.image = img_tk_final

if __name__ == "__main__":
    app = MultiDigitApp()
    app.mainloop()