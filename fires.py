import cv2
import time
from ultralytics import YOLO
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import requests
from playsound import playsound

model = YOLO("best.pt")
model = model.to("cpu")

# Replace with your DroidCam URL or any other source
droidcam_url = "http://192.168.1.74:4747/video"
cap = cv2.VideoCapture(droidcam_url)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("output_video.mp4", fourcc, 20.0, (640, 480))

previous_detections = []
threshold_box_area = 10000
time_threshold_increase = 5
time_threshold_constant = 10
last_alert_time = 0

def get_public_ip():
    response = requests.get('https://api.ipify.org?format=json')
    data = response.json()
    ip_address = data.get('ip')
    return ip_address

def get_location(ip_address):
    url = f'https://ipapi.co/{ip_address}/json/'
    response = requests.get(url)
    data = response.json()
    city = data.get('city')
    region = data.get('region')
    country = data.get('country_name')
    latitude = data.get('latitude')
    longitude = data.get('longitude')
    return city, region, country, latitude, longitude

def send_alert_with_image(message, image):
    sender_email = "amanjha2132@gmail.com"
    password = "dhrzkrgwblobotuw"
    receiver_email = "mathtrigo3@gmail.com"

    msg = MIMEMultipart()
    msg['Subject'] = "Fire Alert!"
    msg['From'] = sender_email
    msg['To'] = receiver_email

    text = MIMEText(message)
    msg.attach(text)

    image_data = cv2.imencode('.jpg', image)[1].tobytes()
    image_part = MIMEImage(image_data, name='fire_detected.jpg')
    msg.attach(image_part)

    ip_address = get_public_ip()
    city, region, country, latitude, longitude = get_location(ip_address)
    location_text = f'Location: {city}, {region}, {country}\n'
    location_text += f'Latitude: {latitude}\n'
    location_text += f'Longitude: {longitude}\n'
    location_info = MIMEText(location_text, 'plain')
    msg.attach(location_info)

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, msg.as_string())
        print("Alert email with image sent successfully!")
    except Exception as e:
        print("Error sending email:", e)


def play_siren():
    siren_file = 'C:\\Users\\user\\OneDrive\\Desktop\\semster-4\\fire_alarm.mp3'
    playsound(siren_file)

while True:
    ret, frame = cap.read()
    print('ret=', ret) 

    # Perform object detection
    results = model(frame)

    original_frame = frame.copy()

    # Extract bounding boxes, classes, names, and confidences
    for detection in results:
        boxes = results[0].boxes.xyxy.tolist()
        classes = results[0].boxes.cls.tolist()
        names = results[0].names
        confidences = results[0].boxes.conf.tolist()

        if boxes is not None and confidences is not None:
            for box, prob, class_id in zip(boxes, confidences, names):
                if model.names[class_id] in ["fire", "smoke"]:
                    x1, y1, x2, y2 = box
                    box_width = x2 - x1
                    box_height = y2 - y1
                    box_area = box_width * box_height

                    # Check for increasing bounding box:
                    if any(abs(prev_box[0] - x1) > 5 or abs(prev_box[1] - y1) > 5 or abs(prev_box[2] - x2) > 5 or abs(prev_box[3] - y2) > 5 for prev_box in previous_detections):
                        current_time = time.time()
                        if current_time - last_alert_time > time_threshold_increase:
                            image = original_frame.copy() 
                            send_alert_with_image("Fire is Detected: Bounding box increasing",image)
                            play_siren()
                            alert_triggered = True
                            last_alert_time = current_time

                    # Check for large bounding box:
                    if box_area > threshold_box_area:
                        image = original_frame.copy() 
                        send_alert_with_image("Fire is Detected: Large bounding box!",image)
                        play_siren()
                        alert_triggered = True

                    # Check for constant detection:
                    if alert_triggered:
                        current_time = time.time()
                        if current_time - last_alert_time > time_threshold_constant:
                            image = original_frame.copy() 
                            send_alert_with_image("Fire is Detected: Constant detection!",image)
                            play_siren()
                            last_alert_time = current_time

                    # Update previous detections:
                    previous_detections.append([x1, y1, x2, y2])
                    if len(previous_detections) > 10:  # Keep a limited history
                        previous_detections.pop(0)

                    # Draw bounding box and text
                    cv2.rectangle(original_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)  # Red for fire/smoke
                    text_x = x1 + 5  # Slightly offset from left edge
                    text_y = y1 + 20  # Adjust for inner placement

                    # Check if text would go beyond frame edges:
                    if text_y > original_frame.shape[0] - 5:  # Leave a small margin at the bottom
                        text_y = y1 - 15  # Place text above the box

                    # Add text label inside the bounding box:
                    cv2.putText(original_frame, f"{model.names[class_id]}: {prob:.2f}", (int(text_x), int(text_y)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)  # White text for better contrast
                    # ... (rest of the code for text positioning and drawing)

            out.write(original_frame)

        cv2.imshow("Fire Detection System", original_frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()