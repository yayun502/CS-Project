### 功能說明： 動態換臉  
程式參考於 https://gitee.com/yuhong-ldu/python-ai/tree/master/faceswap  
原程式為靜態換臉，本程式更改為動態偵測並由傳入之照片進行換臉。  
### 執行前可能需要：
1. 下載dlib的特殊套件「shape_predictor_68_face_landmarks.dat」  
   下載資訊： https://blog.csdn.net/qq_51985653/article/details/113748025  
   官方下載點： http://dlib.net/files/   
   -> 點選shape_predictor_68_face_landmarks.dat.bz2   
   -> 用7zip解壓縮成shape_predictor_68_face_landmarks.dat  
   -> 和主程式檔放在同層資料夾    
2. 可加入任意照片(置於face的資料夾當中)進行換臉功能(main中更改img_src)
### 小缺點：  
銜接仍有小縫隙，但不影響整體大致上的外觀。
### 可能遇到的錯誤與解決方案：  
錯誤訊息： (mtype == CV_8U || mtype == CV_8S)  
錯誤原因： https://blog.csdn.net/qq1261275789/article/details/106543832  
解決方案： 可參考上述網址說明，目前是調整較大圖片(也就是img_src)為小一點的size <-cv2.resize

