import cv2
from ultralytics import YOLO
import RPi.GPIO as GPIO
from time import sleep

# YOLO 모델 로드
model_path = 'D:/ai/study/fault_detection/pcbtest/runs/detect/train2/weights/best.pt'
model = YOLO(model_path)

# 카메라 초기화
camera = cv2.VideoCapture(0)  # 0은 기본 카메라를 의미

# 오류 검출 플래그 초기화
error_detected = False


servoPin          = 12   # 서보 핀
SERVO_MAX_DUTY    = 12   # 서보의 최대(180도) 위치의 주기
SERVO_MIN_DUTY    = 3    # 서보의 최소(0도) 위치의 주기

GPIO.setmode(GPIO.BOARD)        # GPIO 설정
GPIO.setup(servoPin, GPIO.OUT)  # 서보핀 출력으로 설정

servo = GPIO.PWM(servoPin, 50)  # 서보핀을 PWM 모드 50Hz로 사용하기 (50Hz > 20ms)
servo.start(0)  # 서보 PWM 시작 duty = 0, duty가 0이면 서보는 동작하지 않는다.


def set_angle(angle):
    duty = angle / 18 + 2
    GPIO.output(servoPin, True)
    servo.ChangeDutyCycle(duty)
    sleep(1)
    GPIO.output(servoPin, False)
    servo.ChangeDutyCycle(0)

def main():
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
                    set_angle(90)  # 90도 회전
                    sleep(3)
                    set_angle(0)  # 다시 0도

                # 결과 시각화
                annotated_frame = results[0].plot()

                # 화면에 프레임 표시
                cv2.imshow('YOLOv8 Inference', annotated_frame)

                # 'q' 키를 누르면 루프를 종료합니다
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except Exception as e:
            print(f"An error occurred: {e}")
            break

    # 서보모터 정지 및 GPIO 리소스 해제
    servo.stop()
    GPIO.cleanup()

    # 카메라 자원 해제
    camera.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    print("Starting YOLOv8 camera inference...")
    main()
