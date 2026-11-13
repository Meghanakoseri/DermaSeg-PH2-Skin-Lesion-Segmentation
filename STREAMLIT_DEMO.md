# DermaSeg - Streamlit Demo



The Streamlit application performs skin-lesion segmentation using a trained U-Net model. It supports image upload, prediction, visual comparison and mask downloads.



## 1. Required model checkpoints



Place these two trained checkpoint files inside `models/checkpoints/`:



- `tune_focal_tversky_lr_0005_best.pt`

- `tune_ce_lr_0005_best.pt`



The first is the proposed CE + Focal Tversky model. The second is the Cross-Entropy baseline.



The model weights are excluded from Git and must be downloaded separately from the project backup.



## 2. Install dependencies



```powershell

pip install -r requirements.txt

```



## 3. Launch the application



```powershell

python -m streamlit run app.py

```



Open http://localhost:8501 in your browser.



## 4. Using the application



1\. Select the baseline or proposed model from the sidebar.

2\. Adjust the prediction threshold if required.

3\. Upload a dermoscopic image.

4\. Optionally upload its matching ground-truth lesion mask.

5\. View the original image, ground-truth mask, predicted mask, overlap map and overlays.

6\. Inspect the prediction summary and download the predicted mask or overlay.



When a matching ground-truth mask is supplied, the application calculates per-image IoU, Dice/F1, precision and recall.



## 5. Limitations



- This is an academic research demonstration, not a medical diagnostic tool.

- Ground-truth comparison metrics are meaningful only when the uploaded mask corresponds to the uploaded image.

- The application performs inference; it does not train the model.

- Clinical use and diagnostic decisions are outside the scope of this project.



