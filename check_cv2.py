import sys
try:
    import cv2
    print(f"Successfully imported cv2 from: {cv2.__file__}")
except ImportError as e:
    print(f"Failed to import cv2: {e}")
print(f"sys.path: {sys.path}")