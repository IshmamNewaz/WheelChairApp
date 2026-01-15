#Functions
import logging, cv2, time, os, pyttsx3
from collections import Counter
from ultralytics import YOLO

def stopDefaultLogging():
    for name in logging.root.manager.loggerDict:
        t_logger = logging.getLogger(name)
        t_logger.setLevel(logging.CRITICAL)

speak = pyttsx3.init()
voices = speak.getProperty('voices')
speak.setProperty('voice', voices[3].id)
speak.setProperty('rate', 150)
def talk(text):
    speak.say(text)
    speak.runAndWait()

def imgCapture(numberOfImg, captureSource):
    imgSrc = []
    folder_path = os.path.join(os.path.dirname(__file__), "images")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    for i in range(numberOfImg):
        ret, frame = captureSource.read()
        image_path = os.path.join(folder_path, f"image_{i}.jpg")
        cv2.imwrite(image_path, frame)
        imgSrc.append(image_path)
    return imgSrc

def blackScreenDetection(frame):
    imgHeight = frame.shape[0]
    top_half = frame[0:imgHeight//2, :]
    mean_intensity = cv2.mean(top_half)[0]
    if mean_intensity < 1:
        return 1
    else:
        return 0

def decision(path, imageSize, confThres, model): # For Single Image
    inputSource = cv2.imread(path)
    if blackScreenDetection(inputSource)==0:
        results = model(
            source=inputSource, 
            imgsz=imageSize, 
            conf=confThres,
            stream=False, 
            show=False
            )
        for r in results:
            if r.boxes.cls.numel() == 0:
                print(os.path.basename(path))
                return "unknown"
            else:
                for i, cls in enumerate(r.boxes.cls):
                    class_names = r.names[int(cls)]
                    class_confidence = r.boxes.conf[i].item()*100
                    print(os.path.basename(path))
                    return class_names #, class_confidence
    else:
        return "Black Screen!"

def decisionBatch(imagePaths, imageSize, confThres, model): # For a Batch Of Images
    batch_results = []
    for path in imagePaths:
        inputSource = cv2.imread(path)
        if blackScreenDetection(inputSource) == 0:
            results = model(
                source=inputSource,
                imgsz=imageSize,
                conf=confThres,
                stream=False,
                show=False
            )
            for r in results:
                if r.boxes.cls.numel() == 0:
                    print(os.path.basename(path))
                    batch_results.append("unknown")
                else:
                    for i, cls in enumerate(r.boxes.cls):
                        class_names = r.names[int(cls)]
                        class_confidence = r.boxes.conf[i].item() * 100
                        print(os.path.basename(path))
                        batch_results.append(class_names)  # , class_confidence
        else:
            batch_results.append("Black Screen!")
    return batch_results

def most_common_result(batch_results):
    counter = Counter(batch_results)
    return counter.most_common(1)[0][0]


# Main
def main():
    numberOfImagesInBatch = 5
    cap = None
    for i in range(11):  # 0..10
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"Using camera index {i}")
            break
        cap.release()
        cap = None

    if cap is None:
        raise RuntimeError("No camera found (indices 0–10)")
    imgSize = 640
    confidenceThreshold = 0.4
    weightPath = "/home/trp/Desktop/WheelChair/RoadSignDetection/last.pt "
    model = YOLO(weightPath)

    last_common_result = None
    last_confirm_time = 0.0 

    while True:
        stopDefaultLogging()
        images = imgCapture(numberOfImagesInBatch, cap)
        batch_results = decisionBatch(images, imgSize, confidenceThreshold, model)

        for i, result in enumerate(batch_results):
            print(f"Result of image {i+1}: {result}")

        common_result = most_common_result(batch_results)
        print(f"Most common result: {common_result}\n")
        current_time = time.time()

        if common_result != last_common_result:
            last_common_result = common_result
            last_confirm_time = 0.0

        if last_common_result is not None and last_common_result != 'unknown' and current_time - last_confirm_time >= 10:
            print(f"\n{last_common_result} is confirmed")
            time.sleep(.25)
            talk(f"{last_common_result} Ahead!")
            last_confirm_time = current_time

if __name__ == "__main__":
    main()