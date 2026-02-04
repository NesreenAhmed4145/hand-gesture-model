import cv2
import mediapipe as mp

# اسم الصورة
image_path = 'istockphoto-1145169556-612x612.jpg' # تأكدي من الاسم والمسار

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)

image = cv2.imread(image_path)
if image is None:
    print("❌ Image not found!")
    exit()

results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

print(f"🔍 Detected Hands: {len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0}")

if results.multi_hand_landmarks:
    for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

cv2.imshow("Hand Check", image)
cv2.waitKey(0)
cv2.destroyAllWindows()