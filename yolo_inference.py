from ultralytics import YOLO
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

model = YOLO('models/best.pt')

results = model.predict('input_videos/08fd33_4.mp4', save=True, device='cuda')

print(results[0])
print('='*10)
for box in results[0].boxes:
    print(box)