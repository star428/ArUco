# opencv-aruco

## 文件结构

```
opencv-aruco
├── calibration.py  // 利用棋盘格校准相机，生成distortion_coefficients.npy和calibration_matrix.npy
├── camera_app.py   // 屏幕截图小应用
├── detect_video.py // 检测aruco tag是否存在
├── distortion_coefficients.npy // 相机文件，用来pose_estimation用
├── generate.py  // 生成aruco图片
├── generate_chessboard.py // 生成棋盘格校准图片
├── pose_estimation.py  // 推测pose estimation
├── readme.md
├── utils.py            // 其他文件
└── calibration_matrix.npy  // 校准文件，用来pose_estimation用
```
