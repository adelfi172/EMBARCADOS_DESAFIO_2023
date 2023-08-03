
import numpy as np
import cv2 as cv
from time import time
import multiprocessing
import math


# Video configs
VIDEO_PATH = "video_7_96x96.mp4"
FRAME_RATE = 28.86
RECORD_RESULT = True

# Filters configs
THRESHOLD = 32

# AVG_PIXEL = np.array((149, 178, 137))
# AVG_PIXEL = np.array((120, 137, 171))
# AVG_PIXEL = np.array((134, 150, 84))
AVG_PIXEL = np.array(((120+134+149)/3, (178+137+150)/3, (137+171+84)/3))

# AVG_PIXEL = np.array((52, 57, 32)) # background detection

# Execution configs
USE_THREADS = False
THREAD_NUMBER = 2


_COLOR_BLACK = (0, 0, 0)
_COLOR_WHITE = (255, 255, 255)

_WAIT_FRAME_TIME = int(1000 / FRAME_RATE)  # in ms

Pixel = tuple[int, int, int]


def get_frame_number(capture: cv.VideoCapture) -> int:
    return capture.get(cv.CAP_PROP_POS_FRAMES)


def get_video_shape(capture: cv.VideoCapture) -> tuple[int, int]:
    return (
        int(capture.get(cv.CAP_PROP_FRAME_WIDTH)),
        int(capture.get(cv.CAP_PROP_FRAME_HEIGHT)),
    )


def norm_thresh(pixel: Pixel) -> Pixel:
    euclidian_distance = math.dist(AVG_PIXEL, pixel)

    if euclidian_distance < THRESHOLD:
        return _COLOR_WHITE
    else:
        return _COLOR_BLACK


def handle_frame(line: np.ndarray) -> np.ndarray:
    result: np.ndarray = np.ndarray(shape=(len(line), 3))
    for i, pixel in enumerate(line):
        result[i] = norm_thresh(pixel)
        
    return result

def main() -> None:
    cap = cv.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        raise FileNotFoundError(f"Could not find video `{VIDEO_PATH}`.")

    height, width = get_video_shape(cap)

    if RECORD_RESULT:
        # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
        # Define the fps to be equal to 10. Also frame size is passed.
        cap_out = cv.VideoWriter('outpy.mp4', cv.VideoWriter_fourcc(*'DIVX'), 30, get_video_shape(cap))


    with multiprocessing.Pool(THREAD_NUMBER) as pool:
        while cap.isOpened():
            ret, original_frame = cap.read()
            if not ret:
                raise EOFError()

            # original_frame = cv.cvtColor(original_frame, cv.COLOR_RGB2HSV)

            #time_start = time()
            if USE_THREADS:
                result_frame = np.array(pool.map(handle_frame, original_frame))
            else:
                result_frame = original_frame.copy()
                for i, line in enumerate(original_frame):
                    result_frame[i] = handle_frame(line)

            kernel = np.ones((8, 8), np.uint8)
            result_frame = cv.dilate(result_frame, kernel, iterations = 1)

            kernel = np.ones((8, 8),np.uint8)
            result_frame = cv.erode(result_frame, kernel, iterations = 1)
            
            result_frame = cv.morphologyEx(result_frame, cv.MORPH_CLOSE, kernel)

            #result_frame[width // 2] = [0, 255, 0]
            left_pos = 0
            for i in reversed(range(width // 2)):
                pixel = result_frame[width // 2, i]
                if pixel[0] == 255 and pixel[1] == 255 and  pixel[2] == 255:
                    left_pos = i
                    break
            
            right_pos = 0
            for i in range(width // 2 + 1, width):
                pixel = result_frame[width // 2, i]
                if pixel[0] == 255 and pixel[1] == 255 and  pixel[2] == 255:
                    right_pos = i
                    break
            
          
            print(f"left pos: {left_pos} right pos: {right_pos}")
            print(f"left dist: {width // 2 - left_pos} right pos: {width // 2 + right_pos}")

            #result_frame = cv.circle(result_frame, center=(height // 2, width // 2), radius=0, color=(255, 0, 0), thickness=-1)
            result_frame = cv.circle(result_frame, center=(height // 2, width // 2), radius=0, color=(255, 0, 0), thickness=-1)


            #time_end = time()
            #print(
            #    f"Frame {get_frame_number(cap):03.0F}: {(time_end - time_start) * 1000:.3f}ms"
            #)

            if RECORD_RESULT:
                cap_out.write(result_frame)

            result_frame = cv.circle(result_frame, center=(left_pos, height // 2), radius=1, color=(255, 0, 255), thickness=2)
            result_frame = cv.circle(result_frame, center=(right_pos, height // 2), radius=1, color=(255, 0, 255), thickness=2)
            original_frame = cv.circle(original_frame, center=(left_pos, height // 2), radius=1, color=(255, 0, 255), thickness=2)
            original_frame = cv.circle(original_frame, center=(right_pos, height // 2), radius=1, color=(255, 0, 255), thickness=2)


            cv.namedWindow("Original", cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
            cv.imshow("Original", original_frame)
            cv.namedWindow("Result", cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
            cv.imshow("Result", result_frame)

            

            if cv.waitKey(_WAIT_FRAME_TIME) == ord("q"):
                break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
