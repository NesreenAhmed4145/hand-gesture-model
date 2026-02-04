import cv2
import mediapipe as mp
import pickle
import numpy as np
import math

# ==========================================
# ⚙️ الإعدادات (Settings)
# ==========================================
MODEL_PATH = 'hand_gesture_model.pkl'
CLASPED_DISTANCE_THRESHOLD = 120  # المسافة بالبكسل لاعتبار اليد مشبكة
MOVEMENT_THRESHOLD = 5.0          # لتجاهل الاهتزازات البسيطة جداً

# ==========================================
# 1. تحميل الموديل
# ==========================================
try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print(f"✅ Model loaded: {MODEL_PATH}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

# إعداد MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5, # قللنا الرقم قليلاً لتحسين الكشف في الفيديو
    min_tracking_confidence=0.5
)

def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ Error: Could not open video file {video_path}")
        return

    # معلومات الفيديو
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"🎞️ Processing Video: {video_path}")
    print(f"   Resolution: {width}x{height} | FPS: {fps} | Frames: {total_frames}")
    print("⏳ Please wait, analyzing...")

    # --- متغيرات لتخزين الإحصائيات ---
    stats = {
        "frames_processed": 0,
        "open_count": 0,
        "clasped_count": 0,
        "unknown_count": 0,
        "total_movement": 0.0,
        "movement_samples": 0
    }
    
    prev_wrists = [] # لتتبع الحركة

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break # انتهى الفيديو

        stats["frames_processed"] += 1
        
        # تجهيز الصورة
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        
        frame_status = "Unknown" # الحالة المبدئية لهذا الفريم
        current_wrists = []
        hands_data = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 1. استخراج البيانات
                landmarks = hand_landmarks.landmark
                wrist = landmarks[0]
                wrist_px = (int(wrist.x * width), int(wrist.y * height))
                current_wrists.append(wrist_px)

                # تجهيز الصف للموديل
                row = []
                middle_mcp = landmarks[9]
                hand_size = math.sqrt((wrist.x - middle_mcp.x)**2 + (wrist.y - middle_mcp.y)**2)
                scale = hand_size if hand_size > 0.01 else 1.0
                
                for lm in landmarks:
                    rel_x = (lm.x - wrist.x) / scale
                    rel_y = (lm.y - wrist.y) / scale
                    rel_z = (lm.z - wrist.z) / scale
                    row.extend([rel_x, rel_y, rel_z])
                
                # التوقع
                pred = model.predict(np.array([row]))[0]
                hands_data.append({"wrist": wrist_px, "label": pred})

            # 2. منطق تحديد الحالة (Open vs Clasped)
            if len(hands_data) == 2:
                # حساب المسافة بين اليدين
                x1, y1 = hands_data[0]['wrist']
                x2, y2 = hands_data[1]['wrist']
                dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                
                if dist < CLASPED_DISTANCE_THRESHOLD:
                    frame_status = "Clasped"
                else:
                    frame_status = "Open"
            
            elif len(hands_data) == 1:
                # لو يد واحدة، نعتمد على الموديل
                pred = hands_data[0]['label']
                if "Clasped" in pred: frame_status = "Clasped"
                elif "Open" in pred: frame_status = "Open"
                else: frame_status = "Open" # افتراض أن الباقي مفتوح

            # 3. حساب سرعة الحركة
            frame_movement = 0
            if len(prev_wrists) == len(current_wrists) and len(current_wrists) > 0:
                for i in range(len(current_wrists)):
                    p_x, p_y = prev_wrists[i]
                    c_x, c_y = current_wrists[i]
                    move_dist = math.sqrt((c_x - p_x)**2 + (c_y - p_y)**2)
                    frame_movement += move_dist
                
                # نأخذ المتوسط لو فيه يدين
                frame_movement /= len(current_wrists)

            if frame_movement > MOVEMENT_THRESHOLD:
                stats["total_movement"] += frame_movement
                stats["movement_samples"] += 1

            prev_wrists = current_wrists

        else:
            prev_wrists = [] # فقدنا التتبع
        
        # تسجيل الإحصائيات
        if frame_status == "Clasped": stats["clasped_count"] += 1
        elif frame_status == "Open": stats["open_count"] += 1
        else: stats["unknown_count"] += 1

        # (اختياري) عرض الفيديو أثناء التحليل - يمكنك إلغاؤه للسرعة
        cv2.putText(frame, f"Status: {frame_status}", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Analyzing Video...', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()
    
    # ==========================================
    # 📊 إصدار التقرير النهائي
    # ==========================================
    print("\n" + "="*40)
    print("       📋 FINAL ANALYSIS REPORT       ")
    print("="*40)
    
    total_valid_frames = stats["open_count"] + stats["clasped_count"]
    if total_valid_frames == 0: total_valid_frames = 1 # تجنب القسمة على صفر
    
    # 1. النسب المئوية
    open_score = (stats["open_count"] / total_valid_frames) * 100
    clasped_score = (stats["clasped_count"] / total_valid_frames) * 100
    
    print(f"🔹 Total Frames Processed: {stats['frames_processed']}")
    print(f"\n✋ Hand Posture Score:")
    print(f"   ✅ Open Hand:    {open_score:.1f}%")
    print(f"   🔒 Clasped Hand: {clasped_score:.1f}%")
    
    # 2. تحليل السرعة
    avg_speed = 0
    if stats["movement_samples"] > 0:
        avg_speed = stats["total_movement"] / stats["movement_samples"]
    
    print(f"\n🚀 Movement Analysis:")
    print(f"   Average Speed: {avg_speed:.2f} pixels/frame")
    
    # تقييم نصي للسرعة
    behavior = "Calm / Stable 😌"
    if avg_speed > 15: behavior = "High Energy / Nervous ⚡"
    elif avg_speed > 8: behavior = "Normal / Conversational 🗣️"
    
    print(f"   📝 Conclusion: The subject appears {behavior}")
    print("="*40 + "\n")

# --- تشغيل الدالة ---
# ضعي مسار الفيديو الخاص بك هنا
video_file = 'C:/Users/anesr/Downloads/interview videos/MASTER_BODY_LANGUAGE_in_JOB_INTERVIEWS_Interview_Tips_Techniques_jobinterview_720P.mp4' 
analyze_video(video_file)