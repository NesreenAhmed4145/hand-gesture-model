import cv2
import mediapipe as mp
import csv
import os
import glob
import math

# 1. إعداد MediaPipe لليد
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True, 
    max_num_hands=1, 
    min_detection_confidence=0.5
)

# 2. ملف البيانات الجديد
csv_file = 'hand_gesture_dataset_normalized.csv'

# 3. العناوين
header = ['label']
for i in range(21): 
    header += [f'x{i}', f'y{i}', f'z{i}'] # 21 نقطة لليد

with open(csv_file, mode='w', newline='') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(header)

def create_hand_dataset(class_folders):
    total_count = 0
        
    for class_name, folder_path in class_folders.items():
        print(f"🔄 Reading folder: {class_name}...")
        
        image_paths = []
        # الكود القديم كان يضيف التكرارات
        for ext in ['jpg', 'jpeg', 'png', 'JPG', 'PNG']: 
                image_paths.extend(glob.glob(os.path.join(folder_path, f"*.{ext}")))
        
        # 🔥 الحل السحري: هذا السطر سيمسح أي تكرار فوراً
        image_paths = list(set(image_paths))
            
        print(f"   📂 Found {len(image_paths)} images (Unique).")        
        class_count = 0
            
        with open(csv_file, mode='a', newline='') as f:
            csv_writer = csv.writer(f)
            
            for img_path in image_paths:
                image = cv2.imread(img_path)
                if image is None: continue
                
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(image_rgb)
                
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        landmarks = hand_landmarks.landmark
                        
                        row = [class_name]
                        
                        # --- 🔥 التعديل الجوهري لليد (Scale Normalization) ---
                        
                        # النقطة المرجعية (المركز): المعصم (Point 0)
                        wrist = landmarks[0]
                        
                        # نقطة القياس الثابتة: عقلة الإصبع الوسطى (Point 9)
                        # نستخدم هذه النقطة لأن مكانها لا يتغير سواء اليد مفتوحة أو مغلقة
                        middle_mcp = landmarks[9]
                        
                        # حساب حجم اليد في الصورة (المسافة بين المعصم والعقلة)
                        hand_size = math.sqrt(
                            (wrist.x - middle_mcp.x)**2 + 
                            (wrist.y - middle_mcp.y)**2
                        )
                        
                        # حماية من القسمة على صفر
                        scale_factor = hand_size if hand_size > 0.01 else 1.0

                        for lm in landmarks:
                            # 1. نطرح المعصم (عشان نثبت المكان)
                            # 2. نقسم على حجم اليد (عشان نثبت الحجم/المسافة)
                            rel_x = (lm.x - wrist.x) / scale_factor
                            rel_y = (lm.y - wrist.y) / scale_factor
                            rel_z = (lm.z - wrist.z) / scale_factor
                            
                            row.extend([rel_x, rel_y, rel_z])
                        
                        csv_writer.writerow(row)
                        class_count += 1
                    
        print(f"   ✅ Extracted: {class_count} images for {class_name}")
        total_count += class_count

    print(f"\n🎉 Done! Total rows: {total_count}")
    print(f"📁 Saved to: {csv_file}")

# --- مسارات اليد (عدليها حسب جهازك) ---
my_hand_folders = {
    'Open Hand': r'dataset_final/Open Hand',
    'Closed Hand': r'dataset_final/Closed Hand',
    'Pointing': r'dataset_final/Pointing hand',
    'Clasped Hand': r'dataset_final/Clasped Hand'

}

if __name__ == "__main__":
    create_hand_dataset(my_hand_folders)