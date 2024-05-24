from flask import Flask, render_template, Response, jsonify
import cv2
from ultralytics import YOLO

app = Flask(__name__)

# YOLO 모델 로드
model_path = 'D:/ai/study/fault_detection/pcbtest/runs/detect/train2/weights/best.pt'
model = YOLO(model_path)

# 카메라 초기화
camera = cv2.VideoCapture(0)  # 0은 기본 카메라를 의미

# 오류 검출 플래그 초기화
error_detected = False

def generate_frames():
    global error_detected
    while True:
        try:
            # 카메라에서 프레임 읽기
            success, frame = camera.read()
            if not success:
                print("Failed to capture image")
                break
            else:
                # YOLOv8 추론 수행
                results = model(frame)
                print("YOLOv8 inference performed")

                # 오류 검출 여부 확인
                error_detected = any(box.conf > 0.5 for box in results[0].boxes)  # 임계값 0.5 예시
                if error_detected:
                    print("Error detected")

                # 결과 시각화
                annotated_frame = results[0].plot()

                # 프레임을 JPEG 형식으로 인코딩
                ret, buffer = cv2.imencode('.jpg', annotated_frame)
                if not ret:
                    print("Failed to encode image")
                    break

                frame = buffer.tobytes()

                # 프레임을 바이트 스트림으로 변환하여 반환
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except Exception as e:
            print(f"An error occurred: {e}")
            break

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/error_status')
def error_status():
    global error_detected
    return jsonify(error_detected=error_detected)

if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(debug=True)
