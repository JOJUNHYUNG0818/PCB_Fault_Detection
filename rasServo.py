import RPi.GPIO as GPIO
import time

pin = 12

GPIO.setmode(GPIO.BCM)
GPIO.setup(pin, GPIO.OUT)
p=GPIO.PWM(pin, 50)
p.start(2.5)

try:
    while True:
        p.ChangeDutyCycle(2.5)
        print('angle:1')
        time.sleep(1)
        p.ChangeDutyCycle(7.5)
        print('angle:5')
        time.sleep(1)
        p.ChangeDutyCycle(12)
        print('angle:8')
        time.sleep(1)
except KeybordInterrupt:
    p.stop()
GPIO.cleanup()