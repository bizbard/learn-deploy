import cv2
import numpy as np
import onnx
import onnxruntime

onnx_model = onnx.load("srcnn3.onnx")
try:
    onnx.checker.check_model(onnx_model)
except Exception:
    print("Model incorrect")
else:
    print("Model correct")

input_img = cv2.imread('face.png').astype(np.float32)
input_img = np.transpose(input_img, [2, 0, 1])
input_img = np.expand_dims(input_img, 0)
# ort_session = onnxruntime.InferenceSession("srcnn.onnx")
# ort_inputs = {'input': input_img}
# ort_output = ort_session.run(['output'], ort_inputs)[0]
#
# ort_output = np.squeeze(ort_output, 0)
# ort_output = np.clip(ort_output, 0, 255)
# ort_output = np.transpose(ort_output, [1, 2, 0]).astype(np.uint8)
# cv2.imwrite("face_ort.png", ort_output)

input_factor = np.array([1, 1, 4, 4], dtype=np.float32)
ort_session = onnxruntime.InferenceSession("srcnn3.onnx")
ort_inputs = {'input': input_img, 'factor': input_factor}
ort_output = ort_session.run(None, ort_inputs)[0]

ort_output = np.squeeze(ort_output, 0)
ort_output = np.clip(ort_output, 0, 255)
ort_output = np.transpose(ort_output, [1, 2, 0]).astype(np.uint8)
cv2.imwrite("face_ort_3.png", ort_output)