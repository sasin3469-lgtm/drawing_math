import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, Response
from flask_cors import CORS
import datetime

app = Flask(__name__)
CORS(app) # 웹페이지(Cloudflare 등)에서 접근 가능하도록 허용

# --- Mediapipe 초기화 ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# --- 전역 설정 ---
canvas = None
draw_color = (255, 0, 0)  # 초기색: 파랑 (BGR)
brush_thickness = 5
eraser_thickness = 60
prev_x, prev_y = 0, 0

# --- 녹화 설정 (파일 저장) ---
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = None # 영상 캡처 시작 시 초기화

def generate_frames():
    global canvas, draw_color, prev_x, prev_y, out
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open video device")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    canvas = np.zeros((height, width, 3), np.uint8)

    # 녹화 파일 초기화
    filename = f"recording_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.avi"
    out = cv2.VideoWriter(filename, fourcc, 20.0, (width, height))

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success: break

            # 1. 우리 눈에 편하게 미리 좌우 반전 (거울 모드)
            frame = cv2.flip(frame, 1)

            # 2. 모델에게 주기 전에 다시 한 번 반전? NO! 
            # 이미 frame이 flip 되었으므로, 모델은 반전된 이미지를 봅니다.
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb_frame)

            if result.multi_hand_landmarks:
                # 손가락 정보와 왼손/오른손 라벨을 함께 가져옵니다.
                for hand_lms, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
                    label = handedness.classification[0].label # 'Left' 또는 'Right'
                    
                    landmarks = hand_lms.landmark
                    ix, iy = int(landmarks[8].x * width), int(landmarks[8].y * height)
                    
                    # 손가락 펴짐 계산 (Y좌표)
                    fingers = []
                    for id in [8, 12, 16, 20]:
                        fingers.append(landmarks[id].y < landmarks[id-2].y)
                    
                    # 엄지 접힘 판별 (왼손/오른손 및 거울 모드 반영)
                    # label이 'Left'면 화면상 오른쪽 손, 'Right'면 화면상 왼쪽 손입니다.
                    if label == 'Left':
                        thumb_folded = landmarks[4].x < landmarks[3].x 
                    else:
                        thumb_folded = landmarks[4].x > landmarks[3].x

                    # --- 모드 로직 ---
                    if all(fingers) and not thumb_folded: # 지우개 (모든 손가락 펴짐)
                        cv2.circle(canvas, (ix, iy), eraser_thickness, (0, 0, 0), -1)
                        prev_x, prev_y = 0, 0
                    elif fingers[0] and not any(fingers[1:]) and thumb_folded: # 그리기 (검지만 펴짐)
                        if prev_x == 0 and prev_y == 0: prev_x, prev_y = ix, iy
                        cv2.line(canvas, (prev_x, prev_y), (ix, iy), draw_color, brush_thickness)
                        prev_x, prev_y = ix, iy
                    else:
                        prev_x, prev_y = 0, 0
                        num_up = sum(fingers)
                        if num_up == 2: draw_color = (0, 0, 255)
                        elif num_up == 3: draw_color = (0, 255, 0)
                        elif num_up == 4: draw_color = (255, 0, 0)
            else:
                # 손이 감지되지 않으면 이전 좌표 초기화 (갑자기 선이 튀는 것 방지)
                prev_x, prev_y = 0, 0

            # 화면 합성 (Bitwise 연산으로 깔끔하게 합침)
            img_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
            _, img_inv = cv2.threshold(img_gray, 20, 255, cv2.THRESH_BINARY_INV)
            img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
            frame = cv2.bitwise_and(frame, img_inv)
            frame = cv2.bitwise_or(frame, canvas)

            # 녹화 파일에 프레임 쓰기
            if out is not None:
                out.write(frame)

            # 웹 스트리밍용 인코딩
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'--frame\r\n')
    finally:
        cap.release()
        if out is not None:
            out.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    # 포트 5000번으로 서버 실행
    app.run(host='0.0.0.0', port=5000, debug=False)


    