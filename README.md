# Google Play Store Apps Analysis

Đồ án cuối kỳ môn **CSC17104 – Programming for Data Science** (Khoa Công nghệ Thông tin, Trường Đại học Khoa học Tự nhiên, ĐHQG-HCM).

Dự án thực hiện quy trình khoa học dữ liệu hoàn chỉnh trên bộ dữ liệu ứng dụng Google Play Store: từ thu thập, khám phá, tiền xử lý có kiểm soát đến đặt câu hỏi chuyên sâu, kiểm chứng độ bền và xây dựng mô hình học máy không rò rỉ dữ liệu.

---

## 1. Thành viên nhóm & Phân công công việc

| Thành viên | MSSV | Câu hỏi phụ trách | Công việc chính |
|---|---|---|---|
| Phạm Hoàng Tuấn Anh | `[Điền MSSV]` | Câu 2.1, Câu 2.2 | Xây dựng module `src/preprocessing.py`, thực hiện chính phần thu thập/khám phá dữ liệu (`01_collection_exploration.ipynb`), tiền xử lý chung và phân tích câu 2.1, 2.2 |
| Văn Đình Hiếu | `23120260` | Câu 2.3, Câu 3.1 (ML) | Khảo sát giấy phép/nhãn dữ liệu, phân tích câu 2.3, xây dựng toàn bộ quy trình học máy (`03_ml_question.ipynb`) và ghi nhận `LLM_USAGE.md` |
| Nguyễn Trung Dũng | `[Điền MSSV]` | Câu 2.4, Câu 2.5 | Phân tích câu 2.4, 2.5 và tổng hợp các phát hiện chính (Key findings) |

---

## 2. Bộ dữ liệu, Nguồn, Giấy phép & Công trình liên quan

### Thông tin dataset
* **Tên dataset:** Google Play Store Apps
* **Nền tảng & Đường dẫn:** [Kaggle – lava18/google-play-store-apps](https://www.kaggle.com/datasets/lava18/google-play-store-apps)
* **Tác giả:** Lavanya Gupta (công bố/cập nhật lần cuối tháng 02/2019 trên Kaggle)
* **Phương pháp thu thập:** Web scraping từ các trang ứng dụng trên Google Play Store.
* **File sử dụng:** `data/raw/googleplaystore.csv` (10.841 dòng gốc, 13 cột; sau tiền xử lý chung còn **10.357 dòng**, **15 cột** gồm thêm `Size_MB` và `Size_varies`).
* **Giấy phép:**
  * Trang Kaggle ghi nhận **CC BY-SA 4.0** ([Creative Commons Attribution-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-sa/4.0/)).
  * File `data/raw/license.txt` đi kèm ghi nhận **CC BY 3.0** ([Creative Commons Attribution 3.0 Unported](https://creativecommons.org/licenses/by/3.0/)).
  * Nhóm sử dụng **CC BY-SA 4.0** làm tham chiếu chính, cho phép dùng cho mục đích học tập với điều kiện ghi công tác giả Lavanya Gupta và dẫn nguồn dataset.

### Khai báo công trình công khai liên quan (Related Public Notebooks)
Trên Kaggle có nhiều notebook công khai phân tích bộ dữ liệu này (chủ yếu tại mục *Code* của dataset `lava18/google-play-store-apps`, ví dụ các bài EDA tổng quan hoặc dự đoán `Rating` của tác giả và cộng đồng). Điểm khác biệt của nhóm:
1. **Xử lý trùng lặp theo đúng đơn vị phân tích:** Thay vì bỏ mù quáng mọi dòng trùng tên hoặc giữ nguyên toàn bộ gây đếm trùng, nhóm chọn đại diện theo từng câu hỏi (cặp `App–Category` ở câu 2.1, một dòng đại diện mỗi `App` ở câu 2.2–2.5) và chia tập theo nhóm `App` ở câu ML.
2. **Độ sâu phân tích & kiểm chứng:** Không dừng ở `groupby`/`value_counts` hay biểu đồ đơn biến; mỗi câu Non-ML đều kết hợp kiểm soát biến nhiễu (phân tầng theo `Category`, `Type`, kích thước, khoảng cài đặt), lượng hóa độ bất định bằng khoảng tin cậy bootstrap 95% và kiểm tra độ bền qua nhiều ngưỡng khác nhau.
3. **Bài toán ML chống rò rỉ dữ liệu:** Dự đoán mốc phổ biến (`Installs >= 1.000.000`) chỉ từ thuộc tính mô tả sản phẩm và cấu hình kỹ thuật; loại bỏ hoàn toàn các biến hậu nghiệm (`Reviews`, `Rating`, `Last Updated`, `Current Ver`) và dùng `GroupShuffleSplit`/`GroupKFold` theo `App`.

---

## 3. Danh sách câu hỏi nghiên cứu

1. **Câu 2.1 (Phạm Hoàng Tuấn Anh — D3, D4):** Trong các ứng dụng trả phí, nhóm có giá tải đến 2 USD hay nhóm có giá trên 2 USD có tỷ lệ đạt cận dưới 10.000 lượt cài cao hơn? Thứ tự giữa hai nhóm có giống nhau khi xét chung và khi so sánh trong từng danh mục không?
2. **Câu 2.2 (Phạm Hoàng Tuấn Anh — D5, D6):** Trong các ứng dụng miễn phí có ít nhất 100 lượt đánh giá, ứng dụng có cận dưới lượt cài cao hơn có xu hướng được chấm điểm cao hơn không, và mối liên hệ mạnh đến đâu? Kết quả thay đổi thế nào khi dùng ngưỡng 50 hoặc 200 lượt đánh giá?
3. **Câu 2.3 (Văn Đình Hiếu — D2, D3, D4, D5, D6):** Các ứng dụng mới được cập nhật gần đây có tỷ lệ đạt từ 1.000.000 lượt cài trở lên cao hơn những ứng dụng lâu rồi không cập nhật không? Mối liên hệ này có còn giữ nguyên khi so sánh trong cùng danh mục và cùng loại (miễn phí hoặc trả phí) không?
4. **Câu 2.4 (Nguyễn Trung Dũng — D3, D4, D5, D6):** Trong cùng danh mục và cùng loại miễn phí/trả phí, ứng dụng dung lượng lớn hơn có lượt cài cao hơn hay thấp hơn? Mối liên hệ có giống nhau giữa các danh mục, đặc biệt giữa game và ứng dụng thường không?
5. **Câu 2.5 (Nguyễn Trung Dũng — D3, D4, D5, D6):** Ứng dụng dành cho thanh thiếu niên hoặc người lớn (`Teen`, `Mature 17+`) có điểm đánh giá khác ứng dụng cho mọi lứa tuổi (`Everyone`) không, khi so sánh trong cùng danh mục và cùng khoảng cài đặt? Kết quả có giữ khi thay đổi cách gộp nhãn và ngưỡng lượt đánh giá không?
6. **Câu 3.1 — Machine Learning (Văn Đình Hiếu):** Nếu chỉ dùng các thông tin mô tả sản phẩm và cấu hình kỹ thuật có sẵn từ đầu (không dùng `Reviews`, `Rating`), mô hình học máy có phân loại được ứng dụng đạt `Installs >= 1.000.000` tốt hơn baseline đoán lớp đa số hay không, và những đặc trưng nào quan trọng nhất?

---

## 4. Các phát hiện chính (Key Findings)

* **Thiếu `Rating` tập trung ở mốc cài đặt thấp:** Trong 10.841 bản ghi gốc có **1.474 bản ghi (13,60%)** thiếu `Rating`, chủ yếu ở các ứng dụng ít lượt cài. Sau làm sạch cơ bản, bảng ứng dụng còn **10.357 dòng**.
* **Ứng dụng trả phí trên 2 USD đạt mốc 10.000 lượt cài thường xuyên hơn (câu 2.1):** Trong 536 cặp ứng dụng–danh mục đủ điều kiện, tỷ lệ đạt mốc là **47,27%** ở nhóm giá trên 2 USD so với **25,33%** ở nhóm đến 2 USD; thứ tự này giữ nguyên trong cả 6 danh mục kiểm tra.
* **Lượt cài và điểm đánh giá có liên hệ dương yếu (câu 2.2):** Trên **5.934 ứng dụng miễn phí** có ít nhất 100 lượt đánh giá, tương quan Spearman là **0,1626** (CI 95%: `[0,1365; 0,1876]`), ổn định ở ngưỡng 50 (`0,1606`) và 200 (`0,1576`).
* **Ứng dụng cập nhật gần đây phổ biến hơn rõ rệt (câu 2.3):** Nhóm cập nhật trong 90 ngày gần nhất đạt tỷ lệ >= 1M lượt cài là **49,9%** so với **20,8%** ở nhóm cũ hơn; chênh lệch sau phân tầng `Category x Type` là **+28,1 điểm phần trăm** (CI 95%: `[26,2; 29,7]`) và dương ở cả 20 danh mục lớn (còn **+20,8 điểm phần trăm** khi kiểm soát thêm kích thước).
* **Ứng dụng dung lượng lớn hơn có lượt cài cao hơn (câu 2.4):** Trên **8.417 ứng dụng** có kích thước xác định, Spearman là **0,310** (sau phân tầng còn **0,266**, CI 95%: `[0,245; 0,283]`), mạnh nhất ở `SHOPPING` (`0,578`) và `GAME` (`0,349`).
* **Nhãn độ tuổi nội dung không tạo khác biệt rõ về điểm đánh giá (câu 2.5):** Khi so trong cùng danh mục và khoảng cài đặt trên **7.307 ứng dụng**, chênh lệch điểm giữa `Teen` + `Mature 17+` và `Everyone` là **-0,030 điểm** (CI 95%: `[-0,066; 0,006]`), không có ý nghĩa thực tiễn.
* **Dự đoán mốc 1 triệu lượt cài từ thuộc tính ban đầu (câu 3.1):** Mô hình `HistGradientBoosting` đạt **ROC-AUC 0,857**, **PR-AUC 0,790** và **F1 0,707** trên tập test (so với baseline đoán lớp đa số có ROC-AUC `0,500`, PR-AUC `0,388`, F1 `0`). Ba đặc trưng quan trọng nhất theo Permutation Importance là `Category`, `Size_MB` và `Price`.

---

## 5. Cấu trúc thư mục

```text
google-play-store-apps-analysis/
├── README.md
├── environment.yml
├── LLM_USAGE.md
├── data/
│   ├── raw/                        # Dữ liệu gốc (googleplaystore.csv, license.txt)
│   └── processed/                  # Dữ liệu sau tiền xử lý chung (apps_clean.csv)
├── src/
│   └── preprocessing.py            # Các hàm làm sạch và ép kiểu dữ liệu dùng chung
├── notebooks/
│   ├── 01_collection_exploration.ipynb
│   ├── 02_analysis.ipynb
│   └── 03_ml_question.ipynb
└── slides.pptx                     # Slide thuyết trình (xuất slides.pdf khi nộp)
```

---

## 6. Hướng dẫn cài đặt và chạy dự án

### Bước 1: Tạo môi trường Conda
```bash
conda env create -f environment.yml
conda activate google-play-store-apps-analysis
```

### Bước 2: Chuẩn bị dữ liệu
1. Tải file `googleplaystore.csv` từ [Kaggle dataset](https://www.kaggle.com/datasets/lava18/google-play-store-apps).
2. Đặt file vào đường dẫn `data/raw/googleplaystore.csv`.

### Bước 3: Chạy các notebook theo thứ tự
Mở JupyterLab (`jupyter lab`) hoặc chạy trực tiếp bằng `nbconvert` theo đúng thứ tự:
1. `notebooks/01_collection_exploration.ipynb` — Thu thập, kiểm tra chất lượng và khám phá dữ liệu.
2. `notebooks/02_analysis.ipynb` — Tiền xử lý chung (xuất `data/processed/apps_clean.csv`), phân tích 5 câu hỏi Non-ML (2.1–2.5) và tổng kết/phản tư.
3. `notebooks/03_ml_question.ipynb` — Huấn luyện, tinh chỉnh, đánh giá và giải thích mô hình Machine Learning (câu 3.1).
