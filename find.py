from ultralytics import YOLO
import time
import psutil

# 코드 실행 시작 시간 기록
start_time = time.time()

# 실행할 코드 작성
model = YOLO('./best.pt')

img = './picture/hard_pic2.jpg'

results=model(img, save_txt= True, save=True)
# 코드 실행 종료 시간 기록
end_time = time.time()

# 걸린 시간 계산
elapsed_time = end_time - start_time
print(f"코드 실행에 걸린 시간: {elapsed_time} 초")

# 사용된 컴퓨팅 파워 계산
cpu_usage = psutil.cpu_percent()
memory_usage = psutil.virtual_memory().percent

print(f"CPU 사용량: {cpu_usage}%")
print(f"메모리 사용량: {memory_usage}%")
